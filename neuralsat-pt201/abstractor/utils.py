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
from util.misc.logger import logger


def update_refined_beta(self, betas, batch):
    if betas is not None:
        if not len(betas['sparse_beta']):
            return
        self.net.set_bound_opts({'optimize_bound_args': {'enable_beta_crown': True}})
        assert len(self.net.relus) == len(betas['sparse_beta'])
        for relu_idx, relu_layer in enumerate(self.net.relus):
            relu_layer.sparse_beta = betas['sparse_beta'][relu_idx].detach().clone().repeat(batch, 1).requires_grad_() # need detach()
            relu_layer.sparse_beta_loc = betas['sparse_beta_loc'][relu_idx].clone().repeat(batch, 1)
            relu_layer.sparse_beta_sign = betas['sparse_beta_sign'][relu_idx].clone().repeat(batch, 1)
    

def new_slopes(slopes, keep_name):
    new_slope = {}
    for relu_layer, alphas in slopes.items():
        new_slope[relu_layer] = {}
        if keep_name in alphas:
            new_slope[relu_layer][keep_name] = alphas[keep_name]
    return new_slope


def _transfer(tensor, device=None, half=False):
    assert device in ['cpu', 'cuda']
    assert isinstance(half, bool)
    if half:
        tensor = tensor.half()
    if device:
        tensor = tensor.to(device)
    return tensor


def get_slope(self, half=True, device='cpu'):
    if len(self.net.perturbed_optimizable_activations) == 0:
        return {}
    slopes = {
        m.name: {
            node_name: _transfer(alpha, device=device, half=half) for (node_name, alpha) in m.alpha.items()
        } for m in self.net.perturbed_optimizable_activations
    } 
    return slopes


def set_slope(self, slope, set_all=False):
    assert isinstance(slope, defaultdict), print(type(slope))
    for m in self.net.perturbed_optimizable_activations:
        for node_name in list(m.alpha.keys()):
            if node_name in slope[m.name]:
                if (node_name == self.net.final_name) or set_all:
                    slope_len = slope[m.name][node_name].size(2)
                    if slope_len > 0:
                        m.alpha[node_name] = slope[m.name][node_name]
                        m.alpha[node_name] = m.alpha[node_name].repeat(1, 1, 2, *([1] * (m.alpha[node_name].ndim - 3))).detach().requires_grad_() # 2 * batch
                        # print('setting alpha:', m.name, node_name, m.alpha[node_name].shape, m.alpha[node_name].dtype, m.alpha[node_name].sum().item())
            else:
                # do not use alphas
                del m.alpha[node_name]


def get_hidden_bounds(self, output_lbs, device='cpu'):
    lower_bounds, upper_bounds = {}, {}
    output_ubs = output_lbs + torch.inf
    
    # get hidden bounds
    for layer in self.net.layers_requiring_bounds:
        lower_bounds[layer.name] = _transfer(layer.lower.detach(), device=device)
        upper_bounds[layer.name] = _transfer(layer.upper.detach(), device=device)
    
    # add output bounds
    lower_bounds[self.net.final_name] = _transfer(output_lbs.flatten(1).detach(), device=device)
    upper_bounds[self.net.final_name] = _transfer(output_ubs.flatten(1).detach(), device=device)
    
    return lower_bounds, upper_bounds


def get_lAs(self, size=None, device='cpu'):
    lAs = {}
    for node in self.net.get_splittable_activations():
        lA = getattr(node, 'lA', None)
        if lA is None:
            continue
        preserve_mask = self.net.last_update_preserve_mask
        if preserve_mask is not None:
            assert size is not None
            new_lA = torch.zeros([size, lA.shape[0]] + list(lA.shape[2:]), dtype=lA.dtype, device=lA.device)
            new_lA[preserve_mask] = lA.transpose(0, 1)
            lA = new_lA
        else:
            lA = lA.transpose(0, 1)
        lAs[node.name] = _transfer(lA, device=device)
    return lAs


def get_beta(self, num_splits, device='cpu'):
    ret = []
    for i in range(len(num_splits)):
        betas = {k: _transfer(self.net[k].sparse_betas[0].val[i, :num_splits[i][k]], device) for k in num_splits[i]}
        ret.append(betas)
    return ret


def reset_beta(self, batch, max_splits_per_layer, betas=None, bias=False):
    for layer_name in max_splits_per_layer:
        layer = self.net[layer_name]
        start_nodes = []
        for act in self.net.split_activations[layer_name]:
            start_nodes.extend(list(act[0].alpha.keys()))
        shape = (batch, max_splits_per_layer[layer_name])
        if betas is not None and betas[0] is not None and layer_name in betas[0]:
            betas_ = [(betas[bi][layer_name] if betas[bi] is not None else None) for bi in range(batch)]
        else:
            betas_ = [None for _ in range(batch)]
        # set betas
        self.net.reset_beta(layer, shape, betas_, bias=bias, start_nodes=list(set(start_nodes)))


def _copy_history(history):
    assert history is not None
    ret = {}
    for k, v in history.items():
        if isinstance(v[0], torch.Tensor):
            ret[k] = v
        elif isinstance(v[0], list):
            ret[k] = tuple(v[i].copy() for i in range(len(v)))
        else:
            ret[k] = tuple(copy.deepcopy(v[i]) for i in range(len(v)))
    return ret

def _append_tensor(tensor, value, dtype=torch.float32):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=dtype)
    size = len(tensor)
    res = torch.empty(size=(size+1,), dtype=dtype)
    res[:size] = tensor
    res[-1] = value
    return res

def update_histories(self, histories, decisions):
    double_histories = []
    batch = len(decisions)

    # double the histories
    for _ in range(2):
        for h in histories:
            double_histories.append(_copy_history(h))
    
    # add new decisions to histories
    for i, h in enumerate(double_histories):
        bi = i % batch
        l_id, n_id = decisions[bi]
        l_name = self.net.split_nodes[l_id].name
        # FIXME: check repeated decisions
        loc = _append_tensor(h[l_name][0], n_id, dtype=torch.long)
        sign = _append_tensor(h[l_name][1], +1 if i < batch else -1)
        beta = _append_tensor(h[l_name][2], 0.0) # FIXME: 0.0 is for ReLU only
        h[l_name] = (loc, sign, beta)
        
    return double_histories
    

def set_beta(self, betas, histories, decision, use_beta=True):
    if not use_beta:
        for m in self.net.splittable_activations:
            m.beta = None
        return None

    batch = len(histories)
    splits_per_example = []
    max_splits_per_layer = {}
    
    for bi in range(batch):
        splits_per_example.append({})
        for k, v in histories[bi].items():
            splits_per_example[bi][k] = len(v[0])
            max_splits_per_layer[k] = max(max_splits_per_layer.get(k, 0), splits_per_example[bi][k])

    # set old betas
    self.reset_beta(
        betas=betas,
        max_splits_per_layer=max_splits_per_layer, 
        batch=batch, 
        bias=False,
    )

    # set new betas
    for node in self.net.split_nodes:
        if node.sparse_betas is None:
            continue
        sparse_betas = node.sparse_betas if isinstance(node.sparse_betas, list) else node.sparse_betas.values()
        for sparse_beta in sparse_betas:
            sparse_beta.apply_splits(histories, node.name)
            
    return splits_per_example
            
            
@torch.no_grad()
def hidden_split_idx(self, lower_bounds, upper_bounds, decision):
    batch = len(decision)
    splitting_indices_batch = {k: [] for k in lower_bounds}
    splitting_indices_neuron = {k: [] for k in lower_bounds}
    splitting_points = {k: [] for k in lower_bounds}

    for i in range(batch):
        l_id, n_id = decision[i][0], decision[i][1]
        node = self.net.split_nodes[l_id]
        splitting_indices_batch[node.name].append(i)
        splitting_indices_neuron[node.name].append(n_id)
        splitting_points[node.name].append(0.0) # FIXME: split at 0 for ReLU
    
    # convert to tensor
    splitting_indices_batch = {k: torch.as_tensor(v).to(device=self.device, non_blocking=True) for k, v in splitting_indices_batch.items()}
    splitting_indices_neuron = {k: torch.as_tensor(v).to(device=self.device, non_blocking=True) for k, v in splitting_indices_neuron.items()}
    splitting_points = {k: torch.as_tensor(v).to(device=self.device, non_blocking=True) for k, v in splitting_points.items()}
    
    # 2 * batch
    double_upper_bounds = {k: torch.cat([v, v], dim=0) for k, v in upper_bounds.items()}
    double_lower_bounds = {k: torch.cat([v, v], dim=0) for k, v in lower_bounds.items()}

    # construct new hidden bounds
    new_intermediate_layer_bounds = {}
    for key in double_lower_bounds:
        assert len(double_lower_bounds[key]) == len(double_upper_bounds[key]) == 2 * batch
        if len(splitting_indices_batch[key]):
            # set 1st half (set lower)
            double_lower_bounds[key].view(2 * batch, -1)[splitting_indices_batch[key], splitting_indices_neuron[key]] = splitting_points[key]
            # set 2nd half (set upper)
            double_upper_bounds[key].view(2 * batch, -1)[splitting_indices_batch[key] + batch, splitting_indices_neuron[key]] = splitting_points[key]
        new_intermediate_layer_bounds[key] = [double_lower_bounds[key], double_upper_bounds[key]]
            
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

    
def build_lp_solver(self, model_type, input_lower, input_upper, c, refine, rhs=None, intermediate_layer_bounds=None, timeout=None, timeout_per_neuron=None):
    assert model_type in ['lp', 'mip']

    if hasattr(self.net, 'model'): 
        if (intermediate_layer_bounds is None) and torch.equal(self.last_c_lp, c) \
            and (self.net.model.ModelName == model_type) \
            and torch.equal(self.last_input_lower, input_lower) and torch.equal(self.last_input_upper, input_upper):
            logger.debug('[!] Reuse built LP model')
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
    lb, _ = self.net.compute_bounds(x=(new_x,), C=c, method="backward", reference_bounds=intermediate_layer_bounds)
    # print(lb)
    if rhs is not None:
        if (lb > rhs).all():
            return None
        
    
    # self.net.set_bound_opts({'optimize_bound_args': {
    #             'enable_beta_crown': False, 
    #             'iteration': 100, 
    #             'lr_alpha': 0.1,
    #         }} ) 
    # self.net.compute_bounds(x=(new_x,), C=c, method="crown-optimized")
    
    # build solver
    self.net.build_solver_module(
        x=(new_x,), 
        # intermediate_layer_bounds=intermediate_layer_bounds,
        C=c, 
        final_node_name=self.net.final_name, 
        model_type=model_type, 
        timeout=timeout,
        timeout_per_neuron=timeout_per_neuron,
        refine=refine,
    )
    self.net.model.update()
    self.last_c_lp = c
    self.last_input_lower = input_lower.clone()
    self.last_input_upper = input_upper.clone()


def solve_full_assignment(self, input_lower, input_upper, lower_bounds, upper_bounds, c, rhs):
    logger.debug('Full assignment')
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