from collections import defaultdict, OrderedDict
import gurobipy as grb
import numpy as np
import torch
import copy

from auto_LiRPA.bound_ops import BoundRelu, BoundOptimizableActivation
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor

from .params import get_branching_opt_params
from util.misc.check import check_solution


def new_slopes(slopes, keep_name):
    new_slope = {}
    for relu_layer, alphas in slopes.items():
        new_slope[relu_layer] = {}
        if keep_name in alphas:
            new_slope[relu_layer][keep_name] = alphas[keep_name]
    return new_slope


def get_slope(self, model):
    if len(model.perturbed_optimizable_activations) == 0:
        return {}
    slopes = {m.name: {node_name: alpha for (node_name, alpha) in m.alpha.items()}
                for m in model.perturbed_optimizable_activations} 
    return slopes


def set_slope(self, model, slope, set_all=False):
    assert isinstance(slope, defaultdict)
    for m in model.perturbed_optimizable_activations:
        for node_name in list(m.alpha.keys()):
            if node_name in slope[m.name]:
                if (node_name == self.net.final_name) or set_all:
                    slope_len = slope[m.name][node_name].size(2)
                    if slope_len > 0:
                        m.alpha[node_name] = slope[m.name][node_name]
                        m.alpha[node_name] = m.alpha[node_name].repeat(1, 1, 2, *([1] * (m.alpha[node_name].ndim - 3))).detach().requires_grad_() # 2 * batch
            else:
                # do not use alphas
                del m.alpha[node_name]


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


def get_hidden_bounds(self, model, lb):
    ub = (torch.zeros_like(lb) + np.inf).to(lb.device)
    lower_bounds = [layer.inputs[0].lower.detach() for layer in model.perturbed_optimizable_activations]
    upper_bounds = [layer.inputs[0].upper.detach() for layer in model.perturbed_optimizable_activations]
    lower_bounds.append(lb.flatten(1).detach())
    upper_bounds.append(ub.flatten(1).detach())
    self.pre_relu_indices = [i for (i, layer) in enumerate(model.perturbed_optimizable_activations) if isinstance(layer, BoundRelu)]
    self.name_dict = {i: layer.inputs[0].name for (i, layer) in enumerate(model.perturbed_optimizable_activations)}
    return lower_bounds, upper_bounds


def get_batch_hidden_bounds(self, model, lb):
    batch = len(lb)
    ub = (torch.zeros_like(lb) + np.inf).to(lb.device)
    lower_bounds = [layer.inputs[0].lower for layer in model.perturbed_optimizable_activations]
    upper_bounds = [layer.inputs[0].upper for layer in model.perturbed_optimizable_activations]
    lower_bounds.append(lb.view(batch, -1).detach())
    upper_bounds.append(ub.view(batch, -1).detach())
    return lower_bounds, upper_bounds


def get_lAs(self, model):
    if len(model.perturbed_optimizable_activations) == 0:
        return [None]
    lA = [layer.lA.transpose(0, 1) if (hasattr(layer, 'lA') and layer.lA is not None) else None 
            for layer in model.perturbed_optimizable_activations]
    return lA


def get_batch_lAs(self, model, size=None, to_cpu=False):
    if len(model.perturbed_optimizable_activations) == 0:
        return [None]

    lA = []
    preserve_mask = self.net.last_update_preserve_mask
    if preserve_mask is not None:
        for this_relu in model.perturbed_optimizable_activations:
            new_lA = torch.zeros([size, this_relu.lA.shape[0]] + list(this_relu.lA.shape[2:]), dtype=this_relu.lA.dtype, device=this_relu.lA.device)
            new_lA[preserve_mask] = this_relu.lA.transpose(0, 1)
            lA.append(new_lA.to(device='cpu', non_blocking=False) if to_cpu else new_lA)
    else:
        for this_relu in model.perturbed_optimizable_activations:
            lA.append(this_relu.lA.transpose(0, 1).to(device='cpu', non_blocking=False) if to_cpu else this_relu.lA.squeeze(0))
    return lA


def get_beta(self, model, num_splits):
    batch = num_splits.size(0)
    retb = [[] for _ in range(batch * 2)]
    for mi, m in enumerate(model.perturbed_optimizable_activations):
        if hasattr(m, 'sparse_beta'): # discard padding beta.
            for i in range(batch):
                retb[i].append(m.sparse_beta[i, :num_splits[i, mi]])
                retb[i + batch].append(m.sparse_beta[i + batch, :num_splits[i, mi]])
    return retb


def set_beta(self, model, betas, histories, decision, use_beta=True):
    if use_beta:
        batch = len(decision)
        # count split nodes 
        num_splits = torch.zeros((batch, len(model.relus)), dtype=torch.int64, device='cpu') # (batch, num of layers)
        for bi in range(batch):
            d = decision[bi][0]
            for mi, layer_splits in enumerate(histories[bi]):
                neuron_indices = layer_splits[0]
                num_splits[bi, mi] = len(neuron_indices) + int(d == mi)

        # update beta
        self.reset_beta(
            batch=batch, 
            betas=betas, 
            max_splits_per_layer=num_splits.max(dim=0)[0],
        )

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
        num_splits = None
        for m in model.relus:
            m.beta = None
            
    return num_splits
            
            
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
            
    return new_intermediate_layer_bounds


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

    split_value = (input_lowers_cp[indices, idx] + input_uppers_cp[indices, idx]) / 2

    input_lowers_cp[indices, idx] = split_value
    input_uppers_cp_tmp[indices, idx] = split_value
    
    input_lowers_cp = torch.cat([input_lowers_cp, input_lowers_cp_tmp])
    input_uppers_cp = torch.cat([input_uppers_cp, input_uppers_cp_tmp])

    new_input_lowers = input_lowers_cp.reshape(-1, *self.input_shape[1:])
    new_input_uppers = input_uppers_cp.reshape(-1, *self.input_shape[1:])

    return new_input_lowers, new_input_uppers


def transfer_to_cpu(self, net, non_blocking=True, slope_only=False):
    class TMP:
        pass
    
    cpu_net = TMP()
    cpu_net.perturbed_optimizable_activations = [None] * len (net.perturbed_optimizable_activations)
    for i in range(len(cpu_net.perturbed_optimizable_activations)):
        cpu_net.perturbed_optimizable_activations[i] = lambda : None
        cpu_net.perturbed_optimizable_activations[i].inputs = [lambda : None]
        cpu_net.perturbed_optimizable_activations[i].name = net.perturbed_optimizable_activations[i].name

    for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
        # alphas
        cpu_layer.alpha = OrderedDict()
        for node_name, alpha in layer.alpha.items():
            cpu_layer.alpha[node_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)
        # skip others
        if slope_only:
            continue
        # hidden bounds
        cpu_layer.inputs[0].lower = layer.inputs[0].lower.to(device='cpu', non_blocking=non_blocking)
        cpu_layer.inputs[0].upper = layer.inputs[0].upper.to(device='cpu', non_blocking=non_blocking)
        # lAs
        cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
        # betas
        if hasattr(layer, 'sparse_beta') and layer.sparse_beta is not None:
            cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)

    return cpu_net

    
def build_lp_solver(self, model_type, input_lower, input_upper, c, refine, intermediate_layer_bounds=None, timeout_per_neuron=None):
    assert model_type in ['lp', 'mip']

    if hasattr(self.net, 'model'): 
        if (intermediate_layer_bounds is None) and (self.last_c_lp == c or torch.equal(self.last_c_lp, c)) \
            and (self.net.model.ModelName == model_type) \
            and torch.equal(self.last_input_lower, input_lower) and torch.equal(self.last_input_upper, input_upper):
            print('[!] Reuse built LP model')
            return
        self.net.clear_solver_module(self.net.final_node())
        del self.net.model
    
    # gurobi solver
    self.net.model = grb.Model(model_type)
    self.net.model.setParam('OutputFlag', False)
    self.net.model.setParam("FeasibilityTol", 1e-5)
    # self.net.model.setParam('TimeLimit', timeout)
    if model_type == 'mip':
        self.net.model.setParam('MIPGap', 1e-2)  # Relative gap between lower and upper objective bound 
        self.net.model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between lower and upper objective bound 

    # create new inputs
    new_x = BoundedTensor(input_lower, PerturbationLpNorm(x_L=input_lower, x_U=input_upper))
    # disable beta
    
    # forward to recompute hidden bounds
    self.net.set_bound_opts(get_branching_opt_params()) 
    self.net.compute_bounds(x=(new_x,), C=c, method="backward")
    
    # self.net.set_bound_opts({'optimize_bound_args': {
    #             'enable_beta_crown': False, 
    #             'iteration': 100, 
    #             'lr_alpha': 0.1,
    #         }} ) 
    # self.net.compute_bounds(x=(new_x,), C=c, method="crown-optimized")
    
    # build solver
    self.net.build_solver_module(
        x=(new_x,), 
        C=c, 
        final_node_name=self.net.final_name, 
        model_type=model_type, 
        intermediate_layer_bounds=intermediate_layer_bounds,
        timeout_per_neuron=timeout_per_neuron,
        refine=refine,
    )
    self.net.model.update()
    self.last_c_lp = c
    self.last_input_lower = input_lower.clone()
    self.last_input_upper = input_upper.clone()


def solve_full_assignment(self, input_lower, input_upper, lower_bounds, upper_bounds, c, rhs):
    tmp_model = self.net.model.copy()
    tmp_model.update()
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in self.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in self.net.relus]
    
    for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
        lbs, ubs = lower_bounds[relu_idx].reshape(-1), upper_bounds[relu_idx].reshape(-1)
        for neuron_idx in range(lbs.shape[0]):
            pre_var = tmp_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
            pre_var.lb = pre_lb = lbs[neuron_idx]
            pre_var.ub = pre_ub = ubs[neuron_idx]
            var = tmp_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
            # var is None if originally stable
            if var is not None:
                if pre_ub >= pre_lb >= 0:
                    var.lb = pre_lb
                    var.ub = pre_ub
                    tmp_model.addConstr(pre_var == var)
                elif pre_lb <= pre_ub <= 0:
                    var.lb = 0
                    var.ub = 0
                else:
                    raise ValueError(f'Exists unstable neuron at index [{relu_idx}][{neuron_idx}]: lb={pre_lb} ub={pre_ub}')
                
    tmp_model.update()

    feasible = True
    adv = None
    output_vars = self.net.final_node().solver_vars
    assert len(output_vars) == len(rhs), f"out shape not matching! {len(output_vars)} {len(rhs)}"
    for out_idx in range(len(output_vars)):
        objective_var = tmp_model.getVarByName(output_vars[out_idx].VarName)
        tmp_model.setObjective(objective_var, grb.GRB.MINIMIZE)
        tmp_model.update()
        tmp_model.optimize()

        if tmp_model.status == 2:
            # print("Gurobi all node split: feasible!")
            output_lb = objective_var.X
        else:
            # print(f"Gurobi all node split: infeasible! Model status {tmp_model.status}")
            output_lb = float('inf')

        if output_lb > rhs[out_idx]:
            feasible = False
            adv = None
            break
        
        input_vars = [tmp_model.getVarByName(f'inp_{dim}') for dim in range(np.prod(self.input_shape))]
        adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)
        if check_solution(net=self.pytorch_model, adv=adv, cs=c, rhs=rhs, data_min=input_lower, data_max=input_upper):
            return True, adv
        
    del tmp_model
    return feasible, adv
