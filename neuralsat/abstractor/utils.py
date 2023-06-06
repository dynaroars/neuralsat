from collections import defaultdict, OrderedDict
import gurobipy as grb
import torch

from auto_LiRPA.bound_ops import BoundRelu, BoundOptimizableActivation
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor

from .params import get_branching_opt_params


def new_slopes(slopes, keep_name):
    new_slope = {}
    for relu_layer, alphas in slopes.items():
        new_slope[relu_layer] = {}
        if keep_name in alphas:
            new_slope[relu_layer][keep_name] = alphas[keep_name]
    return new_slope


def get_unstable_neurons(masks):
    total_unstables = 0
    for i, layer_mask in enumerate([mask[0:1] for mask in masks]):
        layer_unstables = int(torch.sum(layer_mask).item())
        print(f'Layer {i}: size {layer_mask.shape[1:]}, unstable {layer_unstables}')
        total_unstables += layer_unstables
    print(f'-----------------\nTotal number of unstable neurons: {total_unstables}\n-----------------\n')
    return total_unstables


def get_slope(self, model, only_final=False):
    if len(model.perturbed_optimizable_activations) == 0:
        return {}
    ret = {}
    kept_layer_names = [self.net.final_name]
    for m in model.perturbed_optimizable_activations:
        ret[m.name] = {}
        for spec_name, alpha in m.alpha.items():
            if not only_final or spec_name in kept_layer_names:
                ret[m.name][spec_name] = alpha # (2, spec, batch, *shape)
    return ret


def set_slope(self, model, slope, set_all=False):
    kept_layer_names = [self.net.final_name]
    if isinstance(slope, defaultdict):
        for m in model.perturbed_optimizable_activations:
            for spec_name in list(m.alpha.keys()):
                if spec_name in slope[m.name]:
                    # setup the last layer slopes if no refinement.
                    if spec_name in kept_layer_names or set_all:
                        slope_len = slope[m.name][spec_name].size(2)
                        if slope_len > 0:
                            m.alpha[spec_name] = slope[m.name][spec_name]
                            # 2 * batch
                            m.alpha[spec_name] = m.alpha[spec_name].repeat(1, 1, 2, *([1] * (m.alpha[spec_name].ndim - 3))).detach().requires_grad_()
                else:
                    # layer's alpha is not used
                    del m.alpha[spec_name]
    else:
        raise NotImplementedError


def reset_beta(self, batch, max_splits_per_layer, betas=None):
    for mi, m in enumerate(self.net.perturbed_optimizable_activations):
        if isinstance(m, BoundRelu):
            m.sparse_beta = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            m.sparse_beta_loc = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.int64, device='cpu', requires_grad=False)
            m.sparse_beta_sign = torch.zeros(size=(batch, max_splits_per_layer[mi]), dtype=torch.get_default_dtype(), device='cpu', requires_grad=False)
            for bi in range(batch):
                if betas is not None and betas[bi] is not None:
                    # [batch, relu layers, betas]
                    valid_betas = len(betas[bi][mi])
                    m.sparse_beta[bi, :valid_betas] = betas[bi][mi]
            m.sparse_beta = m.sparse_beta.repeat(2, 1).detach().to(device=self.net.device, non_blocking=True).requires_grad_()


def get_hidden_bounds(self, model, lb, ub):
    lower_bounds = []
    upper_bounds = []
    self.pre_relu_indices = []
    self.name_dict = {}

    for i, layer in enumerate(model.perturbed_optimizable_activations):
        lower_bounds.append(layer.inputs[0].lower.detach())
        upper_bounds.append(layer.inputs[0].upper.detach())
        self.name_dict[i] = layer.inputs[0].name
        if isinstance(layer, BoundRelu):
            self.pre_relu_indices.append(i)

    lower_bounds.append(lb.flatten(1).detach())
    upper_bounds.append(ub.flatten(1).detach())
    return lower_bounds, upper_bounds


def get_batch_hidden_bounds(self, model, lb, ub, batch):
    lower_bounds = []
    upper_bounds = []

    for layer in model.perturbed_optimizable_activations:
        lower_bounds.append(layer.inputs[0].lower)
        upper_bounds.append(layer.inputs[0].upper)

    lower_bounds.append(lb.view(batch, -1).detach())
    upper_bounds.append(ub.view(batch, -1).detach())
    return lower_bounds, upper_bounds


def get_mask_lA(self, model):
    if len(model.perturbed_optimizable_activations) == 0:
        return [None], [None]

    mask, lA = [], []
    for this_relu in model.perturbed_optimizable_activations:
        mask_tmp = torch.logical_and(this_relu.inputs[0].lower < 0, this_relu.inputs[0].upper > 0).float() # 1 is unstable, 0 is stable
        mask.append(mask_tmp.reshape(mask_tmp.size(0), -1))
        if hasattr(this_relu, 'lA') and this_relu.lA is not None:
            lA.append(this_relu.lA.transpose(0, 1))
        else:
            lA.append(None) # inactive
    return mask, lA


def get_lA(self, model, size=None, to_cpu=False):
    preserve_mask = self.net.last_update_preserve_mask
    if len(model.perturbed_optimizable_activations) == 0:
        return [None]
    # get lower A matrix of ReLU
    lA = []
    if preserve_mask is not None:
        for this_relu in model.perturbed_optimizable_activations:
            new_lA = torch.zeros([size, this_relu.lA.shape[0]] + list(this_relu.lA.shape[2:]), dtype=this_relu.lA.dtype, device=this_relu.lA.device)
            new_lA[preserve_mask] = this_relu.lA.transpose(0,1)
            lA.append(new_lA.to(device='cpu', non_blocking=False) if to_cpu else new_lA)
    else:
        for this_relu in model.perturbed_optimizable_activations:
            lA.append(this_relu.lA.transpose(0,1).to(device='cpu', non_blocking=False) if to_cpu else this_relu.lA.squeeze(0))
    return lA


def get_beta(self, model, splits_per_example):
    # split_per_example only has half of the examples.
    batch = splits_per_example.size(0)
    retb = [[] for _ in range(batch * 2)]
    
    for mi, m in enumerate(model.perturbed_optimizable_activations):
        if hasattr(m, 'sparse_beta'):
            # Save only used beta, discard padding beta.
            for i in range(batch):
                retb[i].append(m.sparse_beta[i, :splits_per_example[i, mi]])
                retb[i + batch].append(m.sparse_beta[i + batch, :splits_per_example[i, mi]])
    return retb


def set_beta(self, model, betas, histories, decision, use_beta=True):
    # iteratively change upper and lower bound from former to later layer
    if use_beta:
        batch = len(decision)
        # count split nodes 
        splits_per_example = torch.zeros((batch, len(model.relus)), dtype=torch.int64, device='cpu') # (batch, num of layers)
        for bi in range(batch):
            d = decision[bi][0]
            for mi, layer_splits in enumerate(histories[bi]):
                splits_per_example[bi, mi] = len(layer_splits[0]) + int(d == mi)  # First element of layer_splits is a list of split neuron IDs.

        # warm start beta
        self.reset_beta(batch, betas=betas, max_splits_per_layer=splits_per_example.max(dim=0)[0])

        # update new decisions
        # positive splits in 1st half
        for bi in range(batch):
            d, idx = decision[bi][0], decision[bi][1]
            for mi, (split_locs, split_coeffs) in enumerate(histories[bi]):
                split_len = len(split_locs)
                model.relus[mi].sparse_beta_sign[bi, :split_len] = torch.as_tensor(split_coeffs, device='cpu', dtype=torch.get_default_dtype())
                model.relus[mi].sparse_beta_loc[bi, :split_len] = torch.as_tensor(split_locs, device='cpu', dtype=torch.int64)
                if mi == d:
                    model.relus[mi].sparse_beta_sign[bi, split_len] = 1.0
                    model.relus[mi].sparse_beta_loc[bi, split_len] = idx
                    
        # 2 * batch
        for m in model.relus:
            m.sparse_beta_loc = m.sparse_beta_loc.repeat(2, 1).detach()
            m.sparse_beta_loc = m.sparse_beta_loc.to(device=model.device, non_blocking=True)
            m.sparse_beta_sign = m.sparse_beta_sign.repeat(2, 1).detach()
            
        # negative splits in 2nd half
        for bi in range(batch):
            d = decision[bi][0]
            split_len = len(histories[bi][d][0])
            model.relus[d].sparse_beta_sign[bi + batch, split_len] = -1.0

        for m in model.relus:
            m.sparse_beta_sign = m.sparse_beta_sign.to(device=model.device, non_blocking=True)
    else:
        splits_per_example = None
        for m in model.relus:
            m.beta = None
            
    return splits_per_example
            
            
def hidden_split_idx(self, lower_bounds, upper_bounds, decision):
    batch = len(decision)
    with torch.no_grad():
        # split at 0 for ReLU
        splitting_indices_batch = [[] for _ in range(len(lower_bounds) - 1)]
        splitting_indices_neuron = [[] for _ in range(len(lower_bounds) - 1)]
        for i in range(batch):
            d, idx = decision[i][0], decision[i][1]
            splitting_indices_batch[d].append(i)
            splitting_indices_neuron[d].append(idx)
        splitting_indices_batch = [torch.as_tensor(t).to(device=self.device, non_blocking=True) for t in splitting_indices_batch]
        splitting_indices_neuron = [torch.as_tensor(t).to(device=self.device, non_blocking=True) for t in splitting_indices_neuron]

        # 2 * batch
        tmp_ubs = [torch.cat([i[:batch], i[:batch]], dim=0) for i in upper_bounds[:-1]]
        tmp_lbs = [torch.cat([i[:batch], i[:batch]], dim=0) for i in lower_bounds[:-1]]

        # save for later
        pre_lb_last = torch.cat([lower_bounds[-1][:batch], lower_bounds[-1][:batch]])
        pre_ub_last = torch.cat([upper_bounds[-1][:batch], upper_bounds[-1][:batch]])

        new_intermediate_layer_bounds = {}
        for d in range(len(tmp_lbs)):
            if len(splitting_indices_batch[d]):
                # set active in 1st half (lower = 0)
                tmp_lbs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d], splitting_indices_neuron[d]] = 0.0
                # set inactive in 2nd half (upper = 0)
                tmp_ubs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d] + batch, splitting_indices_neuron[d]] = 0.0
                
                # non-relu activation
                # l_ = tmp_lbs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d], splitting_indices_neuron[d]]
                # u_ = tmp_ubs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d], splitting_indices_neuron[d]]
                # splitting_point = (u_ + l_) / 2
                # tmp_lbs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d], splitting_indices_neuron[d]] = splitting_point
                # tmp_ubs[d][:2 * batch].view(2 * batch, -1)[splitting_indices_batch[d] + batch, splitting_indices_neuron[d]] = splitting_point
                
            new_intermediate_layer_bounds[self.name_dict[d]] = [tmp_lbs[d], tmp_ubs[d]]
            
    return new_intermediate_layer_bounds, pre_lb_last, pre_ub_last


@torch.no_grad()
def input_split_idx(self, input_lowers, input_uppers, split_idx):
    input_lowers = input_lowers.flatten(1)
    input_uppers = input_uppers.flatten(1)

    input_lowers_cp = input_lowers.clone()
    input_uppers_cp = input_uppers.clone()

    indices = torch.arange(input_lowers_cp.shape[0])
    idx = split_idx[:, 0].long()

    input_lowers_cp_tmp = input_lowers_cp.clone()
    input_uppers_cp_tmp = input_uppers_cp.clone()

    mid = (input_lowers_cp[indices, idx] + input_uppers_cp[indices, idx]) / 2

    input_lowers_cp[indices, idx] = mid
    input_uppers_cp_tmp[indices, idx] = mid
    
    input_lowers_cp = torch.cat([input_lowers_cp, input_lowers_cp_tmp])
    input_uppers_cp = torch.cat([input_uppers_cp, input_uppers_cp_tmp])

    new_input_lowers = input_lowers_cp.reshape(-1, *self.input_shape[1:])
    new_input_uppers = input_uppers_cp.reshape(-1, *self.input_shape[1:])

    return new_input_lowers, new_input_uppers


def transfer_to_cpu(self, net, non_blocking=True, opt_intermediate_beta=False, transfer_items="all"):
    """Trasfer all necessary tensors to CPU in a batch."""

    class TMP:
        pass
    cpu_net = TMP()
    cpu_net.perturbed_optimizable_activations = [None] * len (net.perturbed_optimizable_activations)
    for i in range(len(cpu_net.perturbed_optimizable_activations)):
        cpu_net.perturbed_optimizable_activations[i] = lambda : None
        cpu_net.perturbed_optimizable_activations[i].inputs = [lambda : None]
        cpu_net.perturbed_optimizable_activations[i].name = net.perturbed_optimizable_activations[i].name

    # Transfer data structures for each relu.
    if transfer_items == "all":
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
            cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)

        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)

    if transfer_items == "all" or transfer_items == "slopes":
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):

            cpu_layer.alpha = OrderedDict()
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha[spec_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)

    if transfer_items == "all":
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            if hasattr(layer, 'sparse_beta') and layer.sparse_beta is not None:
                cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)

        if opt_intermediate_beta and net.best_intermediate_betas is not None:
            cpu_net.best_intermediate_betas = OrderedDict()
            for split_layer, all_int_betas_this_layer in net.best_intermediate_betas.items():

                assert 'single' in all_int_betas_this_layer
                assert 'history' not in all_int_betas_this_layer
                assert 'split' not in all_int_betas_this_layer
                cpu_net.best_intermediate_betas[split_layer] = {'single': defaultdict(dict)}
                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['lb'] = this_layer_intermediate_betas['lb'].to(device='cpu', non_blocking=non_blocking)
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['ub'] = this_layer_intermediate_betas['ub'].to(device='cpu', non_blocking=non_blocking)
    
    return cpu_net

    
def build_lp_solver(self, model_type, input_lower, input_upper, c):
    assert model_type in ['lp', 'mip']
    # gurobi solver
    self.net.model = grb.Model(model_type)
    self.net.model.setParam('OutputFlag', False)
    self.net.model.setParam("FeasibilityTol", 1e-7)
    # self.net.model.setParam('TimeLimit', timeout)

    # create new inputs
    new_x = BoundedTensor(input_lower, PerturbationLpNorm(x_L=input_lower, x_U=input_upper))
    # disable beta
    self.net.set_bound_opts(get_branching_opt_params()) 
    # forward to update hidden bounds
    self.net.compute_bounds(x=(new_x,), C=c, method="backward")
    # build solver
    self.net.build_solver_module(C=c, final_node_name=self.net.final_name, model_type=model_type)
    self.net.model.update()


def solve_full_assignment(self, lower_bounds, upper_bounds, rhs):
    working_model = self.net.model.copy()
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in self.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in self.net.relus]
    
    for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
        lbs, ubs = lower_bounds[relu_idx].reshape(-1), upper_bounds[relu_idx].reshape(-1)
        for neuron_idx in range(lbs.shape[0]):
            pre_var = working_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
            pre_var.lb = pre_lb = lbs[neuron_idx]
            pre_var.ub = pre_ub = ubs[neuron_idx]
            var = working_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
            # var is None if originally stable
            if var is not None:
                if pre_ub >= pre_lb >= 0:
                    var.lb = pre_lb
                    var.ub = pre_ub
                    working_model.addConstr(pre_var == var)
                elif pre_lb <= pre_ub <= 0:
                    var.lb = 0
                    var.ub = 0
                else:
                    raise ValueError(f'Exists unstable neuron at index [{relu_idx}][{neuron_idx}]: lb={pre_lb} ub={pre_ub}')
                
    working_model.update()

    feasible = True
    adv = None
    output_vars = self.net.final_node().solver_vars
    assert len(output_vars) == len(rhs), f"out shape not matching! {len(output_vars)} {len(rhs)}"
    for out_idx in range(len(output_vars)):
        objective_var = working_model.getVarByName(output_vars[out_idx].VarName)
        working_model.setObjective(objective_var, grb.GRB.MINIMIZE)
        working_model.update()
        working_model.optimize()

        if working_model.status == 2:
            # print("Gurobi all node split: feasible!")
            output_lb = objective_var.X
        else:
            # print(f"Gurobi all node split: infeasible! Model status {working_model.status}")
            output_lb = float('inf')

        if output_lb > rhs[out_idx]:
            feasible = False
            adv = None
            break

        input_vars = [working_model.getVarByName(var.VarName) for var in self.net.input_vars]
        adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)
        
    del working_model
    return feasible, adv
