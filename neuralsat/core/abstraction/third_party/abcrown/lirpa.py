'''
    adopt from beta_CROWN_solver.py of alpha-beta-CROWN
'''

from collections import defaultdict, OrderedDict
import random
import torch
import copy
import time

from .auto_LiRPA.utils import reduction_min, reduction_max, reduction_mean, reduction_sum, stop_criterion_sum, stop_criterion_min
from .auto_LiRPA import BoundedModule, BoundedTensor
from .auto_LiRPA.bound_ops import BoundRelu
from .auto_LiRPA.perturbations import *

import arguments


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func


class LiRPA:

    def __init__(self, model_ori, pred, test, device='cuda', simplify=False, in_size=(1, 3, 32, 32),
                 conv_mode='patches', deterministic=False, c=None, rhs=None):
        """
        convert pytorch model to auto_LiRPA module
        """
        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.simplify = False
        self.c = c
        self.rhs = rhs
        self.pred = pred
        self.layers = layers
        self.input_shape = in_size
        self.net = BoundedModule(net, torch.zeros(in_size, device=device), bound_opts={'relu': 'adaptive', 'deterministic': deterministic, 'conv_mode': conv_mode},
                                 device=device)
        self.net.eval()
        self.needed_A_dict = None
        self.pool = None   # For multi-process.
        self.pool_result = None
        self.pool_termination_flag = None

    
    def get_lower_bound(self, pre_lbs, pre_ubs, split, slopes=None, betas=None, history=None, layer_set_bound=True, 
                        split_history=None, single_node_split=True, intermediate_betas=None):

        """
        # (in) pre_lbs: layers list -> tensor(batch, layer shape)
        # (in) relu_mask: relu layers list -> tensor(batch, relu layer shape (view-1))
        # (in) slope: relu layers list -> tensor(batch, relu layer shape)
        # (out) lower_bounds: batch list -> layers list -> tensor(layer shape)
        # (out) masks_ret: batch list -> relu layers list -> tensor(relu layer shape)
        # (out) slope: batch list -> relu layers list -> tensor(relu layer shape)
        """
        if history is None:
            history = []

        # lp_test = arguments.Config["debug"]["lp_test"]
        assert single_node_split
        if single_node_split:
            ret = self.update_bounds_parallel(pre_lbs, 
                                              pre_ubs, 
                                              split, 
                                              slopes, 
                                              betas=betas, 
                                              early_stop=False, 
                                              history=history,
                                              layer_set_bound=layer_set_bound)

        # if get_upper_bound and single_node_split, primals have p and z values; otherwise None
        lower_bounds, upper_bounds, lAs, slopes, betas, split_history, best_intermediate_betas, primals = ret

        return [i[-1] for i in upper_bounds], [i[-1] for i in lower_bounds], None, lAs, lower_bounds, upper_bounds, slopes, split_history, betas, best_intermediate_betas, primals


    def get_relu(self, model, idx):
        # find the i-th ReLU layer
        i = 0
        for layer in model.children():
            if isinstance(layer, BoundRelu):
                i += 1
                if i == idx:
                    return layer


    """Trasfer all necessary tensors to CPU in a batch."""
    def transfer_to_cpu(self, net, non_blocking=True, opt_intermediate_beta=False):
        # Create a data structure holding all the tensors we need to transfer.
        cpu_net = lambda : None
        cpu_net.relus = [None] * len (net.relus)
        for i in range(len(cpu_net.relus)):
            cpu_net.relus[i] = lambda : None
            cpu_net.relus[i].inputs = [lambda : None]
            cpu_net.relus[i].name = net.relus[i].name

        # Transfer data structures for each relu.
        # For get_candidate_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # For get_candidate_parallel.
            cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
            cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)
        # For get_lA_parallel().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
        # For get_slope().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            # Per-neuron alpha.
            cpu_layer.alpha = OrderedDict()
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha[spec_name] = alpha.to(device='cpu', non_blocking=non_blocking)
        # For get_beta().
        for cpu_layer, layer in zip(cpu_net.relus, net.relus):
            if layer.sparse_beta is not None:
                cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)
        # For intermediate beta.
        if opt_intermediate_beta and net.best_intermediate_betas is not None:
            cpu_net.best_intermediate_betas = OrderedDict()
            for split_layer, all_int_betas_this_layer in net.best_intermediate_betas.items():
                # Single neuron split so far.
                assert 'single' in all_int_betas_this_layer
                assert 'history' not in all_int_betas_this_layer
                assert 'split' not in all_int_betas_this_layer
                cpu_net.best_intermediate_betas[split_layer] = {'single': defaultdict(dict)}
                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['lb'] = this_layer_intermediate_betas['lb'].to(device='cpu', non_blocking=non_blocking)
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['ub'] = this_layer_intermediate_betas['ub'].to(device='cpu', non_blocking=non_blocking)

        return cpu_net


    def get_candidate(self, model, lb, ub):
        # get the intermediate bounds in the current model and build self.name_dict which contains the important index
        # and model name pairs

        lower_bounds = []
        upper_bounds = []
        self.pre_relu_indices = []
        i = 0
        # build a name_dict to map layer idx in self.layers to BoundedModule
        self.name_dict = {}

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower.detach())
            upper_bounds.append(layer.inputs[0].upper.detach())
            self.name_dict[i] = layer.inputs[0].name
            self.pre_relu_indices.append(i)
            i += 1

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(1, -1).detach())
        upper_bounds.append(ub.view(1, -1).detach())

        return lower_bounds, upper_bounds, self.pre_relu_indices


    def get_candidate_parallel(self, model, lb, ub, batch, diving_batch=0):
        # get the intermediate bounds in the current model
        lower_bounds = []
        upper_bounds = []

        for layer in model.relus:
            lower_bounds.append(layer.inputs[0].lower)
            upper_bounds.append(layer.inputs[0].upper)

        # Also add the bounds on the final thing
        lower_bounds.append(lb.view(batch + diving_batch, -1).detach())
        upper_bounds.append(ub.view(batch + diving_batch, -1).detach())

        return lower_bounds, upper_bounds


    def get_mask_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None], [None]
        # get the mask of status of ReLU, 0 means inactive neurons, -1 means unstable neurons, 1 means active neurons (this is not used).
        # get lower A matrix of ReLU
        mask, lA = [], []
        for this_relu in model.relus:
            # 1 is unstable neuron, 0 is stable neuron.
            mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float()
            mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
            if this_relu.lA is not None:
                # lA.append(this_relu.lA.squeeze(0))
                lA.append(this_relu.lA.transpose(0, 1))
            else:
                # It might be skipped due to inactive neurons.
                lA.append(None)

        return mask, lA


    def get_lA_parallel(self, model):
        if len(model.relus) == 0:
            return [None]
        # get lower A matrix of ReLU
        lA = []
        for this_relu in model.relus:
            # lA.append(this_relu.lA.squeeze(0))
            lA.append(this_relu.lA.transpose(0, 1))
        return lA


    def get_beta(self, model, splits_per_example, diving_batch=0):
        # split_per_example only has half of the examples.
        batch = splits_per_example.size(0) - diving_batch
        retb = [[] for i in range(batch * 2 + diving_batch)]
        for mi, m in enumerate(model.relus):
            for i in range(batch):
                # Save only used beta, discard padding beta.
                retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                retb[i + batch].append(m.sparse_beta[i + batch, :splits_per_example[i, mi]])
            for i in range(diving_batch):
                retb[2 * batch + i].append(m.sparse_beta[2 * batch + i, :splits_per_example[batch + i, mi]])
        return retb


    def get_slope(self, model):
        if len(model.relus) == 0:
            return [None]

        # slope has size (2, spec, batch, *shape). When we save it, we make batch dimension the first.
        # spec is some intermediate layer neurons, or output spec size.
        batch_size = next(iter(model.relus[0].alpha.values())).size(2)
        ret = [defaultdict(dict) for i in range(batch_size)]
        for m in model.relus:
            for spec_name, alpha in m.alpha.items():
                # print(f'save layer {m.name} start_node {spec_name} shape {alpha.size()} norm {alpha.abs().sum()}')
                for i in range(batch_size):
                    # each slope size is (2, spec, 1, *shape).
                    ret[i][m.name][spec_name] = alpha[:,:,i:i+1,:]
        return ret


    def set_slope(self, model, slope, intermediate_refinement_layers=None, diving_batch=0):
        cleanup_intermediate_slope = isinstance(intermediate_refinement_layers, list) and len(intermediate_refinement_layers) == 0
        if cleanup_intermediate_slope:
            # Clean all intermediate betas if we are not going to refine intermeidate layer neurons anymore.
            del model.best_intermediate_betas
            for m in model.relus:
                if hasattr(m, 'single_intermediate_betas'):
                    # print(f'deleting single_intermediate_betas for {m.name}')
                    del m.single_intermediate_betas
                if hasattr(m, 'history_intermediate_betas'):
                    # print(f'deleting history_intermediate_betas for {m.name}')
                    del m.history_intermediate_betas
                if hasattr(m, 'split_intermediate_betas'):
                    # print(f'deleting split_intermediate_betas for {m.name}')
                    del m.split_intermediate_betas

        if type(slope) == list:
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[0][m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            # print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
                            if len(slope) - diving_batch > 0:
                                # Merge all slope vectors together in this batch. Size is (2, spec, batch, *shape).
                                m.alpha[spec_name] = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch)], dim=2)
                                # Duplicate for the second half of the batch.
                                m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3))).detach().requires_grad_()
                            if diving_batch > 0:
                                # create diving alpha
                                diving_alpha = torch.cat([slope[i][m.name][spec_name] for i in range(len(slope) - diving_batch, len(slope))], dim=2)
                                if diving_batch == len(slope):
                                    m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                                else:
                                    m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                                del diving_alpha
                            # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        elif type(slope) == defaultdict:
            for m in model.relus:
                for spec_name in list(m.alpha.keys()):
                    if spec_name in slope[m.name]:
                        if cleanup_intermediate_slope and spec_name != model.final_name:
                            # print(f'deleting alpha {spec_name} for layer {m.name}')
                            del m.alpha[spec_name]
                            continue
                        # Only setup the last layer slopes if no refinement is done.
                        if intermediate_refinement_layers is not None or spec_name == model.final_name:
                            # create diving alpha
                            diving_alpha = slope[m.name][spec_name]
                            assert diving_batch == diving_alpha.shape[2]
                            m.alpha[spec_name] = diving_alpha.detach().requires_grad_()
                            # else:
                            #     m.alpha[spec_name] = torch.cat([m.alpha[spec_name], diving_alpha], dim=2).detach().requires_grad_()
                            del diving_alpha
                        # print(f'load layer {m.name} start_node {spec_name} shape {m.alpha[spec_name].size()} norm {m.alpha[spec_name][:,:,0].abs().sum()} {m.alpha[spec_name][:,:,-1].abs().sum()} {m.alpha[spec_name].abs().sum()}')
                    else:
                        # This layer's alpha is not used. For example, we can drop all intermediate layer alphas.
                        del m.alpha[spec_name]
        else:
            raise NotImplementedError


    def reset_beta(self, model, batch, max_splits_per_layer=None, betas=None, diving_batch=0):
        # Recreate new beta with appropriate shape.
        for mi, m in enumerate(self.net.relus):
            # Create only the non-zero beta. For each layer, it is padded to maximal length.
            # We create tensors on CPU first, and they will be transferred to GPU after initialized.
            m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
            m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            # Load beta from history.
            # for bi in range(len(betas)):
            for bi in range(batch):
                if betas[bi] is not None:
                    # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                    valid_betas = len(betas[bi][mi])
                    m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
            # This is the beta variable to be optimized for this layer.
            m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()
            
            assert batch + diving_batch == len(betas)
            if diving_batch != 0:
                m.diving_sparse_beta = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                m.diving_sparse_beta_loc = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
                m.diving_sparse_beta_sign = torch.zeros(size=(diving_batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
                # Load diving beta from history.
                for dbi in range(diving_batch):
                    if betas[batch + dbi] is not None:
                        # First dimension of betas is batch, second dimension is relu layer, third dimension is saved betas.
                        valid_betas = len(betas[batch + dbi][mi])
                        m.diving_sparse_beta[dbi, :valid_betas] = betas[batch + dbi][mi]
                m.diving_sparse_beta = m.diving_sparse_beta.to(device=self.net.device, non_blocking=True)
                m.sparse_beta = torch.cat([m.sparse_beta, m.diving_sparse_beta], dim=0).detach().\
                            to(device=self.net.device, non_blocking=True).requires_grad_()
                del m.diving_sparse_beta


    """Main function for computing bounds after branch and bound in Beta-CROWN."""
    def update_bounds_parallel(self, pre_lb_all=None, pre_ub_all=None, split=None, slopes=None, beta=None, betas=None,
                        early_stop=True, history=None, layer_set_bound=True, shortcut=False):
        # update optimize-CROWN bounds in a parallel

        if beta is None:
            beta = arguments.Config["solver"]["beta-crown"]["beta"] # might need to set beta False in FSB node selection
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        iteration = arguments.Config["solver"]["beta-crown"]["iteration"]
        lr_alpha = arguments.Config["solver"]["beta-crown"]["lr_alpha"]
        lr_beta = arguments.Config["solver"]["beta-crown"]["lr_beta"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]

        assert not get_upper_bound

        if type(split) == list:
            decision = np.array(split)
        else:
            decision = np.array(split["decision"])
            decision = np.array([i.squeeze() for i in decision])

        batch = len(decision)

        # initial results with empty list
        ret_l = [[] for _ in range(batch * 2)]
        ret_u = [[] for _ in range(batch * 2)]
        ret_s = [[] for _ in range(batch * 2)]
        ret_b = [[] for _ in range(batch * 2)]
        new_split_history = [{} for _ in range(batch * 2)]
        # Each key is corresponding to a pre-relu layer, and each value intermediate beta values for neurons in that layer.
        best_intermediate_betas = [defaultdict(dict) for _ in range(batch * 2)] 

        # iteratively change upper and lower bound from former to later layer
        if beta:
            # count how many split nodes in each batch example (batch, num of layers)
            splits_per_example = torch.zeros(size=(batch, len(self.net.relus)), dtype=torch.int64, device='cpu', requires_grad=False)
            for bi in range(batch):
                d = decision[bi][0]
                for mi, layer_splits in enumerate(history[bi]):
                    splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.
            # This is the maximum number of split in each relu neuron for each batch.
            if batch > 0: 
                max_splits_per_layer = splits_per_example.max(dim=0)[0]

            # Create and load warmup beta.
            self.reset_beta(self.net, 
                            batch, 
                            betas=betas, 
                            max_splits_per_layer=max_splits_per_layer)  # warm start beta

            for bi in range(batch):
                # Add history splits.
                d, idx = decision[bi][0], decision[bi][1]
                # Each history element has format [[[layer 1's split location], [layer 1's split coefficients +1/-1]], [[layer 2's split location], [layer 2's split coefficients +1/-1]], ...].
                for mi, (split_locs, split_coeffs) in enumerate(history[bi]):
                    split_len = len(split_locs)
                    self.net.relus[mi].sparse_beta_sign[bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                    self.net.relus[mi].sparse_beta_loc[bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                    # Add current decision for positive splits.
                    if mi == d:
                        self.net.relus[mi].sparse_beta_sign[bi, split_len] = 1.0
                        self.net.relus[mi].sparse_beta_loc[bi, split_len] = idx
            # Duplicate split location.
            for m in self.net.relus:
                m.sparse_beta_loc = m.sparse_beta_loc.repeat(2, 1).detach()
                m.sparse_beta_loc = m.sparse_beta_loc.to(device=self.net.device, non_blocking=True)
                m.sparse_beta_sign = m.sparse_beta_sign.repeat(2, 1).detach()
            # Fixup the second half of the split (negative splits).
            for bi in range(batch):
                d = decision[bi][0]  # layer of this split.
                split_len = len(history[bi][d][0])  # length of history splits for this example in this layer.
                self.net.relus[d].sparse_beta_sign[bi + batch, split_len] = -1.0
            # Transfer tensors to GPU.
            for m in self.net.relus:
                m.sparse_beta_sign = m.sparse_beta_sign.to(device=self.net.device, non_blocking=True)

        else:
            for m in self.net.relus:
                m.beta = None

        # pre_ub_all[:-1] means pre-set bounds for all intermediate layers
        with torch.no_grad():
            # Setting the neuron upper/lower bounds with a split to 0.
            zero_indices_batch = [[] for _ in range(len(pre_lb_all) - 1)]
            zero_indices_neuron = [[] for _ in range(len(pre_lb_all) - 1)]
            for i in range(batch):
                d, idx = decision[i][0], decision[i][1]
                # We save the batch, and neuron number for each split, and will set all corresponding elements in batch.
                zero_indices_batch[d].append(i)
                zero_indices_neuron[d].append(idx)
            zero_indices_batch = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_batch]
            zero_indices_neuron = [torch.as_tensor(t).to(device=self.net.device, non_blocking=True) for t in zero_indices_neuron]

            # 2 * batch
            upper_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_ub_all[:-1]]
            lower_bounds = [torch.cat([i[:batch], i[:batch], i[batch:]], dim=0) for i in pre_lb_all[:-1]]

            # Only the last element is used later.
            pre_lb_last = torch.cat([pre_lb_all[-1][:batch], pre_lb_all[-1][:batch], pre_lb_all[-1][batch:]])
            pre_ub_last = torch.cat([pre_ub_all[-1][:batch], pre_ub_all[-1][:batch], pre_ub_all[-1][batch:]])

            new_candidate = {}
            for d in range(len(lower_bounds)):
                # for each layer except the last output layer
                if len(zero_indices_batch[d]):
                    # we set lower = 0 in first half batch, and upper = 0 in second half batch
                    lower_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d], zero_indices_neuron[d]] = 0.0
                    upper_bounds[d][:2 * batch].view(2 * batch, -1)[zero_indices_batch[d] + batch, zero_indices_neuron[d]] = 0.0
                new_candidate[self.name_dict[d]] = [lower_bounds[d], upper_bounds[d]]

        # create new_x here since batch may change
        ptb = PerturbationLpNorm(norm=self.x.ptb.norm, 
                                 eps=self.x.ptb.eps,
                                 x_L=self.x.ptb.x_L[0].expand(batch * 2, *[-1]*(self.x.ptb.x_L.ndim-1)),
                                 x_U=self.x.ptb.x_U[0].expand(batch * 2, *[-1]*(self.x.ptb.x_L.ndim-1)))
        new_x = BoundedTensor(self.x.data.expand(batch * 2, *[-1]*(self.x.data.ndim-1)), ptb)

        # c = None if self.c is None else self.c.repeat(new_x.shape[0], 1, 1)
        c = None if self.c is None else self.c.expand(new_x.shape[0], -1, -1)
        # self.net(new_x)  # batch may change, so we need to do forward to set some shapes here

        if len(slopes) > 0:
            # set slope here again
            self.set_slope(self.net, slopes)

        if shortcut:
            self.net.set_bound_opts({'optimize_bound_args': {'ob_beta': beta, 'ob_single_node_split': True,
                'ob_update_by_layer': layer_set_bound, 'ob_optimizer':optimizer}})
            with torch.no_grad():
                lb, _, = self.net.compute_bounds(x=(new_x,), 
                                                 IBP=False, 
                                                 C=c, 
                                                 method='backward',
                                                 new_interval=new_candidate, 
                                                 bound_upper=False, 
                                                 return_A=False)
            return lb

        return_A = True if get_upper_bound else False  # we need A matrix to consturct adv example
        if layer_set_bound:
            self.net.set_bound_opts({'optimize_bound_args':
                                         {'ob_beta': beta, 'ob_single_node_split': True,
                                          'ob_update_by_layer': layer_set_bound, 'ob_iteration': iteration,
                                          'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                                          'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            tmp_ret = self.net.compute_bounds(x=(new_x,), 
                                              IBP=False, 
                                              C=c, 
                                              method='CROWN-Optimized',
                                              new_interval=new_candidate, 
                                              return_A=return_A, 
                                              bound_upper=False, 
                                              needed_A_dict=self.needed_A_dict)
        else:
            # all intermediate bounds are re-calculated by optimized CROWN
            self.net.set_bound_opts(
                {'optimize_bound_args': {'ob_beta': beta, 'ob_update_by_layer': layer_set_bound,
                    'ob_iteration': iteration, 'ob_lr': arguments.Config['solver']['beta-crown']['lr_alpha'],
                    'ob_lr_beta': lr_beta, 'ob_optimizer': optimizer}})
            tmp_ret = self.net.compute_bounds(x=(new_x,), 
                                              IBP=False, 
                                              C=c, 
                                              method='CROWN-Optimized',
                                              new_interval=new_candidate, 
                                              return_A=return_A, 
                                              bound_upper=False, 
                                              needed_A_dict=self.needed_A_dict)

        if get_upper_bound:
            raise NotImplementedError
            # lb, _, A = tmp_ret
            # primal_x, ub = self.get_primal_upper_bound(A)
        else:
            lb, _ = tmp_ret
            ub = lb + 99  # dummy upper bound
            primal_x = None


        with torch.no_grad():
            # Move tensors to CPU for all elements in this batch.
            lb, ub = lb.to(device='cpu'), ub.to(device='cpu')
            transfer_net = self.transfer_to_cpu(self.net, non_blocking=False)

            lAs = self.get_lA_parallel(transfer_net)
            if len(slopes) > 0:
                ret_s = self.get_slope(transfer_net)

            if beta:
                ret_b = self.get_beta(transfer_net, splits_per_example)

            # Reorganize tensors.
            lower_bounds_new, upper_bounds_new = self.get_candidate_parallel(transfer_net, lb, ub, batch * 2)

            lower_bounds_new[-1] = torch.max(lower_bounds_new[-1], pre_lb_last.cpu())
            if not get_upper_bound:
                # Do not set to min so the primal is always corresponding to the upper bound.
                upper_bounds_new[-1] = torch.min(upper_bounds_new[-1], pre_ub_last.cpu())
            # reshape the results based on batch.
            for i in range(batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_l[i + batch] = [j[i + batch:i + batch + 1] for j in lower_bounds_new]

                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]
                ret_u[i + batch] = [j[i + batch:i + batch + 1] for j in upper_bounds_new]
            for i in range(2 * batch, 2 * batch):
                ret_l[i] = [j[i:i + 1] for j in lower_bounds_new]
                ret_u[i] = [j[i:i + 1] for j in upper_bounds_new]


        ret_p, primals = None, None
        if get_upper_bound:
            raise NotImplementedError
            # print("opt crown:", lb)
            # primal_values, integer_primals = self.get_neuron_primal(primal_x, lb=lower_bounds_new, ub=upper_bounds_new)
            # correct intermediate primal should produce the correct primal output lb
            # print("primal lb with beta:", primal_values[-1])
            # print("### Extracting primal values and mixed integers with beta for intermeidate nodes done ###")
            # exit()
            # primals = {"p": primal_values, "z": integer_primals}

        return ret_l, ret_u, lAs, ret_s, ret_b, new_split_history, best_intermediate_betas, primal_x


    def build_the_model(self, input_domain, x, stop_criterion_func=stop_criterion_sum(0)):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]

        assert not get_upper_bound
        
        self.x = x
        self.input_domain = input_domain

        slope_opt = None

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        # first get CROWN bounds
        # Reference bounds are intermediate layer bounds from initial CROWN bounds.
        lb, ub, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c, bound_upper=False)
        # print('initial CROWN bounds:', lb, ub)
        if stop_criterion_func(lb).all().item():
            return None, lb[-1], None, None, None, None, None, None, None, None, None, None
        
        self.net.set_bound_opts({'optimize_bound_args': 
                                    {'ob_iteration': init_iteration, 
                                     'ob_beta': False, 
                                     'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 
                                     'ob_optimizer': optimizer,
                                     'ob_early_stop': False, 
                                     'ob_verbose': 0,
                                     'ob_keep_best': True, 
                                     'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 
                                     'ob_init': False,
                                     'ob_loss_reduction_func': loss_reduction_func,
                                     'ob_stop_criterion_func': stop_criterion_func,
                                     'ob_lr_decay': lr_decay}})

        lb, ub = self.net.compute_bounds(x=(x,), 
                                         IBP=False, 
                                         C=self.c, 
                                         method='CROWN-Optimized', 
                                         return_A=False,
                                         bound_upper=False, 
                                         aux_reference_bounds=aux_reference_bounds)

        slope_opt = self.get_slope(self.net)[0]  # initial with one node only

        # update bounds
        # print('initial alpha-CROWN bounds:', lb, ub)
        primals, duals, mini_inp = None, None, None
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds
        mask, lA = self.get_mask_lA_parallel(self.net)

        if not self.simplify or stop_criterion_func(lb[-1]):
            history = [[[], []] for _ in range(len(self.net.relus))]
            return ub[-1], lb[-1], mini_inp, duals, primals, mask, lA, lb, ub, pre_relu_indices, slope_opt, history
            # return ub[-1], lb[-1], mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

        # for each pre-relu layer, we initial 2 lists for active and inactive split
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            raise NotImplementedError
            # self.needed_A_dict = defaultdict(set)
            # self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        return ub[-1], lb[-1], mini_inp, duals, primals, mask, lA, lb, ub, pre_relu_indices, slope_opt, history
        # return ub[-1].item(), lb[-1].item(), mini_inp, duals, primals, mask[0], lA[0], lb, ub, pre_relu_indices, slope_opt, history

    def build_the_model_with_refined_bounds(self, input_domain, x, refined_lower_bounds, refined_upper_bounds,
                                            stop_criterion_func=stop_criterion_sum(0), reference_slopes=None):

        lr_init_alpha = arguments.Config["solver"]["alpha-crown"]["lr_alpha"]
        init_iteration = arguments.Config["solver"]["alpha-crown"]["iteration"]
        share_slopes = arguments.Config["solver"]["alpha-crown"]["share_slopes"]
        no_joint_opt = arguments.Config["solver"]["alpha-crown"]["no_joint_opt"]
        optimizer = arguments.Config["solver"]["beta-crown"]["optimizer"]
        lr_decay = arguments.Config["solver"]["beta-crown"]["lr_decay"]
        loss_reduction_func = arguments.Config["general"]["loss_reduction_func"]
        get_upper_bound = arguments.Config["bab"]["get_upper_bound"]

        assert not get_upper_bound

        self.x = x
        self.input_domain = input_domain

        loss_reduction_func = reduction_str2func(loss_reduction_func)

        slope_opt = None
        if not no_joint_opt:
            ######## using bab_verification_mip_refine.py ########
            lb, ub = refined_lower_bounds, refined_upper_bounds
        primals, duals, mini_inp = None, None, None

        return_A = True if get_upper_bound else False
        if get_upper_bound:
            raise NotImplementedError
            # self.needed_A_dict = defaultdict(set)
            # self.needed_A_dict[self.net.output_name[0]].add(self.net.input_name[0])

        # first get CROWN bounds
        self.net.init_slope((self.x,), share_slopes=share_slopes, c=self.c)
        # If we already have slopes available, we initialize them.
        if reference_slopes is not None:
            for m in self.net.relus:
                for spec_name, alpha in m.alpha.items():
                    # each slope size is (2, spec, 1, *shape); batch size is 1.
                    if spec_name in reference_slopes[m.name]:
                        reference_alpha = reference_slopes[m.name][spec_name]
                        if alpha.size() == reference_alpha.size():
                            # print(f"setting alpha for layer {m.name} start_node {spec_name}")
                            alpha.data.copy_(reference_alpha)
                        else:
                            pass
                            # print(f"not setting layer {m.name} start_node {spec_name} because shape mismatch ({alpha.size()} != {reference_alpha.size()})")
        self.net.set_bound_opts({'optimize_bound_args': 
                                    {'ob_iteration': init_iteration, 
                                     'ob_beta': False, 
                                     'ob_alpha': True,
                                     'ob_alpha_share_slopes': share_slopes, 
                                     'ob_optimizer': optimizer,
                                     'ob_early_stop': False, 
                                     'ob_verbose': 0,
                                     'ob_keep_best': True, 
                                     'ob_update_by_layer': True,
                                     'ob_lr': lr_init_alpha, 
                                     'ob_init': False,
                                     'ob_loss_reduction_func': loss_reduction_func,
                                     'ob_stop_criterion_func': stop_criterion_func,
                                     'ob_lr_decay': lr_decay}})

        if no_joint_opt:
            # using init crown bounds as new_interval
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                new_interval[nd] = [layer.inputs[0].lower, layer.inputs[0].upper] ##!!
                # reference_bounds[nd] = [lb[i], ub[i]]
        else:
            # using refined bounds with init opt crown
            new_interval, reference_bounds = {}, {}
            for i, layer in enumerate(self.net.relus):
                # only refined with the second relu layer
                #if i>=2: break
                nd = layer.inputs[0].name
                # print(i, nd, lb[i].shape)
                new_interval[nd] = [lb[i], ub[i]]
                # reference_bounds[nd] = [lb[i], ub[i]]
        ret = self.net.compute_bounds(x=(x,), 
                                      IBP=False, 
                                      C=self.c, 
                                      method='crown-optimized', 
                                      return_A=return_A,
                                      new_interval=new_interval, 
                                      bound_upper=False, 
                                      needed_A_dict=self.needed_A_dict)
        if return_A:
            lb, ub, A = ret
        else:
            lb, ub = ret

        # print("alpha-CROWN with fixed intermediate bounds:", lb, ub)
        slope_opt = self.get_slope(self.net)[0]
        lb, ub, pre_relu_indices = self.get_candidate(self.net, lb, lb + 99)  # primals are better upper bounds

        mask, lA = self.get_mask_lA_parallel(self.net)
        history = [[[], []] for _ in range(len(self.net.relus))]

        if get_upper_bound:
            raise NotImplementedError
            # print("opt crown:", lb[-1])
            # primal_x, ub_x = self.get_primal_upper_bound(A)
            # print("### Extracting primal values for inputs done ###")

            # # get the primal values for intermediate layers
            # primal_values, integer_primals = self.get_neuron_primal(primal_x, lb, ub)
            # # correct intermediate primal should produce the correct primal output lb
            # print("primal lb:", primal_values[-1])
            # print("### Extracting primal values and mixed integers for intermeidate nodes done ###")
        
            # primals = {"p": primal_values, "z": integer_primals}
        return ub[-1], lb[-1], mini_inp, duals, primals, mask, lA, lb, ub, pre_relu_indices, slope_opt, history

