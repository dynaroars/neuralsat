from beartype import beartype
import gurobipy as grb
import numpy as np
import typing
import torch
import math
import copy
import os

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor

from abstractor.params import get_branching_opt_params
from util.misc.check import check_solution
from util.misc.logger import logger

if typing.TYPE_CHECKING:
    import abstractor

def update_refined_beta(self: 'abstractor.abstractor.NetworkAbstractor', betas, batch):
    pass
    
@beartype
def new_input(self: 'abstractor.abstractor.NetworkAbstractor', x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(x_L <= x_U)
    return BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(self.device)


@beartype
def new_slopes(slopes: dict, keep_name: str) -> dict:
    new_slope = {}
    for relu_layer, alphas in slopes.items():
        new_slope[relu_layer] = {}
        if keep_name in alphas:
            new_slope[relu_layer][keep_name] = alphas[keep_name]
    return new_slope


@beartype
def _to_device(tensor: torch.Tensor, device: str = 'cpu', half: bool = False) -> torch.Tensor:
    assert device in ['cpu', 'cuda'] and isinstance(half, bool)
    if half:
        tensor = tensor.half()
    if device:
        tensor = tensor.to(device)
    return tensor


@beartype
def get_slope(self: 'abstractor.abstractor.NetworkAbstractor', half=True, device='cpu') -> dict:
    if len(self.net.perturbed_optimizable_activations) == 0:
        return {}
    slopes = {
        m.name: {
            node_name: _to_device(alpha, device=device, half=half) for (node_name, alpha) in m.alpha.items()
        } for m in self.net.perturbed_optimizable_activations
    } 
    return slopes


@beartype
def set_slope(self: 'abstractor.abstractor.NetworkAbstractor', slope: dict) -> None:
    for m in self.net.perturbed_optimizable_activations:
        for node_name in list(m.alpha.keys()):
            if node_name in slope[m.name]:
                slope_len = slope[m.name][node_name].size(2)
                if slope_len > 0:
                    m.alpha[node_name] = slope[m.name][node_name]
                    m.alpha[node_name] = m.alpha[node_name].repeat(1, 1, 2, *([1] * (m.alpha[node_name].ndim - 3))).detach().requires_grad_() # 2 * batch
                    # print('setting alpha:', m.name, node_name, m.alpha[node_name].shape, m.alpha[node_name].dtype, m.alpha[node_name].sum().item())
            else:
                # do not use alphas
                del m.alpha[node_name]


@beartype
def get_hidden_bounds(self: 'abstractor.abstractor.NetworkAbstractor', output_lbs: torch.Tensor, device: str = 'cpu') -> tuple[dict, dict]:
    lower_bounds, upper_bounds = {}, {}
    output_ubs = output_lbs + torch.inf
    
    # get hidden bounds
    for layer in list(set(self.net.layers_requiring_bounds + self.net.split_nodes)):
        lower_bounds[layer.name] = _to_device(layer.lower.detach(), device=device)
        upper_bounds[layer.name] = _to_device(layer.upper.detach(), device=device)
    
    # add output bounds
    lower_bounds[self.net.final_name] = _to_device(output_lbs.flatten(1).detach(), device=device)
    upper_bounds[self.net.final_name] = _to_device(output_ubs.flatten(1).detach(), device=device)

    assert len(list(set([_.shape[0] for _ in lower_bounds.values()]))) == 1, print([_.shape[0] for _ in lower_bounds.values()])
    assert len(list(set([_.shape[0] for _ in upper_bounds.values()]))) == 1, print([_.shape[0] for _ in upper_bounds.values()])
    
    return lower_bounds, upper_bounds


@beartype
def get_lAs(self: 'abstractor.abstractor.NetworkAbstractor', size: int | None = None, device: str = 'cpu') -> dict:
    lAs = {}
    # list_nodes = [n for n in self.net.nodes() if n.name == self.net.input_name[0]] if self.input_split else self.net.get_splittable_activations()
    list_nodes = [self.net[self.net.input_name[0]]] if self.input_split else self.net.get_splittable_activations()
    for node in list_nodes:
        lA = getattr(node, 'lA', None)
        if lA is None:
            continue
        lAs[node.name] = _to_device(lA.transpose(0, 1), device=device)
    return lAs


@beartype
def get_beta(self: 'abstractor.abstractor.NetworkAbstractor', num_splits: list[dict], device: str = 'cpu') -> list:
    ret = []
    for i in range(len(num_splits)):
        betas = {k: _to_device(self.net[k].sparse_betas[0].val[i, :num_splits[i][k]], device=device) for k in num_splits[i]}
        ret.append(betas)
    return ret


@beartype
def reset_beta(self: 'abstractor.abstractor.NetworkAbstractor', batch: int, max_splits_per_layer: dict, betas: list | None = None, bias: bool = False) -> None:
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


@beartype
def _copy_history(history: dict) -> dict:
    ret = {}
    for k, v in history.items():
        if isinstance(v[0], torch.Tensor):
            ret[k] = v
        elif isinstance(v[0], list):
            ret[k] = tuple(v[i].copy() for i in range(len(v)))
        else:
            ret[k] = tuple(copy.deepcopy(v[i]) for i in range(len(v)))
    return ret


@beartype
def _append_tensor(tensor: torch.Tensor | list, value: int | float | np.int64, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=dtype)
    size = len(tensor)
    res = torch.empty(size=(size+1,), dtype=dtype)
    res[:size] = tensor
    res[-1] = value
    return res


@beartype
def update_histories(self: 'abstractor.abstractor.NetworkAbstractor', histories: list[dict], decisions: list[list]) -> list[dict]:
    double_histories = []
    batch = len(decisions)

    # double the histories
    for _ in range(2):
        for h in histories:
            double_histories.append(_copy_history(h))
    
    # add new decisions to histories
    for i, h in enumerate(double_histories):
        bi = i % batch
        n_name, n_id, n_point = decisions[bi]
        # FIXME: check repeated decisions
        loc = _append_tensor(h[n_name][0], n_id, dtype=torch.long)
        sign = _append_tensor(h[n_name][1], +1 if i < batch else -1)
        beta = _append_tensor(h[n_name][2], n_point)
        h[n_name] = (loc, sign, beta)
        
    return double_histories
    

@beartype
def set_beta(self: 'abstractor.abstractor.NetworkAbstractor', betas: list, histories: list[dict]) -> list[dict]:
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
        bias=None in self.split_points,
    )

    # set new betas
    for node in self.net.split_nodes:
        if node.sparse_betas is None:
            continue
        sparse_betas = node.sparse_betas if isinstance(node.sparse_betas, list) else node.sparse_betas.values()
        for sparse_beta in sparse_betas:
            sparse_beta.apply_splits(histories, node.name)
            
    return splits_per_example
            
            
@beartype
@torch.no_grad()
def hidden_split_idx(self: 'abstractor.abstractor.NetworkAbstractor', lower_bounds: dict, upper_bounds: dict, 
                     decisions: list[list]) -> dict:
    batch = len(decisions)
    splitting_indices_batch = {k: [] for k in lower_bounds}
    splitting_indices_neuron = {k: [] for k in lower_bounds}
    splitting_points = {k: [] for k in lower_bounds}
    
    for i in range(batch):
        n_name, n_id, n_point = decisions[i]
        splitting_indices_batch[n_name].append(i)
        splitting_indices_neuron[n_name].append(n_id)
        splitting_points[n_name].append(n_point)
    
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
            if os.environ.get('NEURALSAT_ASSERT'):
                assert torch.all(double_lower_bounds[key] <= double_upper_bounds[key])
        new_intermediate_layer_bounds[key] = [double_lower_bounds[key], double_upper_bounds[key]]
            
    assert all([_[0].shape[0] == _[1].shape[0] == 2 * batch for _ in new_intermediate_layer_bounds.values()])
    return new_intermediate_layer_bounds


@beartype
@torch.no_grad()
def input_split_idx(self: 'abstractor.abstractor.NetworkAbstractor', input_lowers: torch.Tensor, input_uppers: torch.Tensor, 
                    split_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    
@beartype
def build_lp_solver(self: 'abstractor.abstractor.NetworkAbstractor', model_type: str, 
                    input_lower: torch.Tensor, input_upper: torch.Tensor, 
                    c: torch.Tensor, rhs: torch.Tensor | None = None, 
                    refine: bool = True, intermediate_layer_bounds: dict | None = None, 
                    timeout: float | None = None, timeout_per_neuron: float | None = None) -> None:
    assert model_type in ['lp', 'mip']
    # delete old LP model
    self.net._reset_solver_vars(self.net.final_node())
    if hasattr(self.net, 'model'): 
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
    new_x = self.new_input(input_lower, input_upper)
    
    # forward to recompute hidden bounds
    self.net.set_bound_opts(get_branching_opt_params()) 
    self.net.init_alpha(x=(new_x,), c=c)
    
    lb, _ = self.net.compute_bounds(x=(new_x,), C=c, method="backward", reference_bounds=intermediate_layer_bounds)
    if rhs is not None:
        if (lb > rhs).all():
            return None
        
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


@beartype
def solve_full_assignment(self: 'abstractor.abstractor.NetworkAbstractor', input_lower: torch.Tensor, input_upper: torch.Tensor, 
                          lower_bounds: dict, upper_bounds: dict, c: torch.Tensor, rhs: torch.Tensor) -> tuple[bool, torch.Tensor | None]:
    logger.debug('Full assignment')
    tmp_model = self.net.model.copy()
    tmp_model.update()
    
    # TODO: assert all activation layers are ReLU
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in self.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in self.net.relus]
    
    for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
        lbs = lower_bounds[pre_relu_name].reshape(-1)
        ubs = upper_bounds[pre_relu_name].reshape(-1)
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
        
        input_vars = [tmp_model.getVarByName(f'inp_{dim}') for dim in range(math.prod(self.input_shape))]
        adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)
        if check_solution(net=self.pytorch_model, adv=adv, cs=c, rhs=rhs, data_min=input_lower, data_max=input_upper):
            return True, adv
        
    del tmp_model
    return feasible, adv



@beartype
def compute_stability(self: 'abstractor.abstractor.NetworkAbstractor', objective):
    cs = objective.cs.to(self.device)
    rhs = objective.rhs.to(self.device)
    
    # input property
    if not torch.allclose(objective.lower_bounds.mean(dim=0), objective.lower_bounds[0], 1e-5, 1e-5):
        input_lowers = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
    else:
        input_lowers = objective.lower_bounds[0:1].to(self.device)
        input_uppers = objective.upper_bounds[0:1].to(self.device)
    
    x = self.new_input(x_L=input_lowers, x_U=input_uppers)
    
    assert self.method in ['forward', 'backward']
    with torch.no_grad():
        lb, _ = self.net.compute_bounds(
            x=(x,), 
            C=cs, 
            method=self.method, 
        )
        lower_bounds, upper_bounds = [], []
        for node in self.net.relus:
            lower_bounds.append(node.inputs[0].lower)
            upper_bounds.append(node.inputs[0].upper)
        
    n_unstable = sum([
        torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).sum().detach().cpu() 
            for j in range(len(lower_bounds))
    ])
    n_total = sum([lower_bounds[j].numel() for j in range(len(lower_bounds))])
    
    return n_total - n_unstable, n_unstable, lower_bounds, upper_bounds
