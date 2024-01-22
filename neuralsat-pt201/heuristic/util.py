from __future__ import annotations
from beartype import beartype
import typing
import torch
import copy

from abstractor.utils import _append_tensor
from solver.sat_solver import SATSolver
from util.misc.logger import logger
from util.misc.result import AbstractResults
from auto_LiRPA.bound_ops import *

if typing.TYPE_CHECKING:
    import abstractor
    import heuristic

    
@beartype
def compute_masks(lower_bounds: dict, upper_bounds: dict, device: str, non_blocking: bool = True) -> dict:
    new_masks = {
        j: torch.logical_and(
                    lower_bounds[j] < 0, 
                    upper_bounds[j] > 0).flatten(start_dim=1).to(torch.get_default_dtype()).to(device=device, non_blocking=non_blocking)
        for j in lower_bounds
    }
    return new_masks


@beartype
def _compute_babsr_scores(abstractor: 'abstractor.abstractor.NetworkAbstractor', 
                          lower_bounds: dict, upper_bounds: dict, 
                          lAs: dict, masks: dict, 
                          reduce_op: typing.Callable, 
                          batch: int, number_bounds: int) -> tuple[list, list]:
    score = []
    intercept_tb = []
    
    # last to first layer
    for layer in reversed(abstractor.net.split_nodes):
        assert len(abstractor.net.split_activations[layer.name]) == 1
        # layer data
        this_layer_mask = masks[layer.name].unsqueeze(1)
        pre_act_layer = abstractor.net.split_activations[layer.name][0][0]
        assert len(pre_act_layer.inputs) == 1
        ratio = lAs[pre_act_layer.name]
        
        # ratio
        ratio_temp_0, ratio_temp_1 = _compute_ratio(lower_bounds[layer.name], upper_bounds[layer.name])

        # intercept scores, backup scores, lower score is better
        intercept_temp = torch.clamp(ratio, max=0)
        intercept_candidate = intercept_temp * ratio_temp_1.unsqueeze(1)
        
        # (batch, neuron)
        reshaped_intercept_candidate = intercept_candidate.view(batch, number_bounds, -1) * this_layer_mask
        intercept_tb.insert(0, reshaped_intercept_candidate.mean(1)) 

        # bias
        b_temp = _get_bias_term(pre_act_layer.inputs[0], ratio)
        
        # branching scores, higher score is better
        ratio_temp_0 = ratio_temp_0.unsqueeze(1)
        bias_candidate_1 = b_temp * (ratio_temp_0 - 1)
        bias_candidate_2 = b_temp * ratio_temp_0
        bias_candidate = reduce_op(bias_candidate_1, bias_candidate_2)
        score_candidate = bias_candidate + intercept_candidate
        
        # (batch, neuron)
        score_candidate = score_candidate.abs().view(batch, number_bounds, -1) * this_layer_mask
        score.insert(0, score_candidate.mean(1)) 
    return score, intercept_tb


@beartype
def _history_to_clause(h: dict, name_mapping: dict) -> list:
    clause = []
    for lname, ldata in h.items():
        assert sum(ldata[2]) == 0 # TODO: fixme, non-ReLU might have non-zero value
        # extract data
        var_names, signs = ldata[0], ldata[1]
        # convert data
        var_names = [name_mapping[lname, int(v)] for v in var_names]
        # negation
        signs = [-int(s) for s in signs]
        # append literals
        clause += [v * s for v, s in zip(var_names, signs)]
    return clause

@beartype
def _compute_ratio(lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = upper_bound.clamp(min=0)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept

    
@beartype
def _get_bias_term(input_node, ratio: torch.Tensor) -> torch.Tensor:
    if type(input_node) == BoundConv:
        if len(input_node.inputs) > 2:
            bias = input_node.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
        else:
            bias = 0
    elif type(input_node) == BoundLinear:
        # TODO: consider if no bias
        bias = input_node.inputs[-1].param.detach()
    elif type(input_node) == BoundAdd:
        bias = 0
        for l in input_node.inputs:
            if type(l) == BoundConv:
                if len(l.inputs) > 2:
                    bias += l.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
            if type(l) == BoundBatchNormalization:
                bias += 0
            if type(l) == BoundAdd:
                for ll in l.inputs:
                    if type(ll) == BoundConv:
                        bias += ll.inputs[-1].param.detach().unsqueeze(-1).unsqueeze(-1)
    elif type(input_node) == BoundBatchNormalization:
        bias = input_node.inputs[-3].param.detach().view(-1, *([1] * (ratio.ndim - 3)))
    else: 
        raise NotImplementedError()
    
    return bias * ratio
    
    
@beartype
def update_hidden_bounds_histories(self: 'heuristic.domains_list.DomainsList', lower_bounds: dict, upper_bounds: dict, 
                                   histories: list, literal: int, batch_idx: int | torch.Tensor) -> bool:
    assert literal != 0
    # extract decision
    lid, nid = self.reversed_var_mapping[abs(literal)]

    # update histories
    history = histories[batch_idx]
    loc = _append_tensor(history[lid][0], nid, dtype=torch.long)
    sign = _append_tensor(history[lid][1], +1 if literal > 0 else -1)
    beta = _append_tensor(history[lid][2], 0.0) # FIXME: 0.0 is for ReLU only
    history[lid] = (loc, sign, beta)
    
    # update bounds
    if literal > 0: # active neuron
        lower_bounds[lid][batch_idx].flatten()[nid] = 0.0
    else: # inactive neuron
        upper_bounds[lid][batch_idx].flatten()[nid] = 0.0
    
    if upper_bounds[lid][batch_idx].flatten()[nid] < lower_bounds[lid][batch_idx].flatten()[nid]:
        return False
    return True


@beartype
def create_one(masks: dict, clauses: list, var_mapping: dict) -> tuple[SATSolver | None, list]:
    # literals
    literals_to_assign = []
    for lname, lmask in masks.items():
        for nid, nstatus in enumerate(lmask.flatten()):
            if nstatus != 0:
                literal = int(nstatus * var_mapping[lname, nid])
                literals_to_assign.append(literal)
            
    # create sat solver
    new_sat_solver = SATSolver(clauses)
    
    # assign
    if not new_sat_solver.multiple_assign(literals_to_assign): 
        return None, [] # conflict
    
    # propagation
    bcp_stat, bcp_vars = new_sat_solver.bcp()
    if not bcp_stat: 
        return None, [] # conflict
    
    return new_sat_solver, bcp_vars


@beartype
def init_sat_solver(self: 'heuristic.domains_list.DomainsList', objective_ids: torch.Tensor, remain_idx: torch.Tensor,
                    lower_bounds: dict, upper_bounds: dict, histories: list, preconditions: dict) -> torch.Tensor:
    assert torch.equal(objective_ids, torch.unique(objective_ids))
    # initial learned conflict clauses
    clauses_per_objective = {k: [_history_to_clause(c, self.var_mapping) for c in v] for k, v in preconditions.items()}
    # pprint(clauses_per_objective)
    
    # masks: 1 for active, -1 for inactive, 0 for unstable
    masks = {
        k: ((lower_bounds[k] > 0).flatten(1).int() - (upper_bounds[k] < 0).flatten(1).int()).detach().cpu() 
            for k in lower_bounds if k != self.net.final_name
    }
    
    masks_per_objective = {int(objective_id): {k: v[batch_id] for k, v in masks.items()} for (batch_id, objective_id) in enumerate(objective_ids)}
    # print(masks_per_objective)
    
    self.all_sat_solvers = []
    new_remain_idx = []
    for (batch_id, objective_id) in enumerate(objective_ids):
        if batch_id not in remain_idx:
            continue
        
        new_sat_solver, bcp_vars = create_one(
            masks=masks_per_objective[int(objective_id)], 
            clauses=clauses_per_objective[int(objective_id)], 
            var_mapping=self.var_mapping,
        )
        
        update_stats = [self.update_hidden_bounds_histories(
            lower_bounds=lower_bounds, 
            upper_bounds=upper_bounds, 
            histories=histories, 
            literal=literal, 
            batch_idx=batch_id,
        ) for literal in bcp_vars] + [True]
            
        # print(batch_id, bcp_vars)
        if (new_sat_solver is not None) and all(update_stats):
            # save
            self.all_sat_solvers.append(new_sat_solver)    
            new_remain_idx.append(batch_id)
                    
    return torch.tensor(new_remain_idx)
    
    
@beartype
def boolean_propagation(self: 'heuristic.domains_list.DomainsList', domain_params: AbstractResults, decisions: list, batch_idx: int | torch.Tensor) -> SATSolver | None:
    assert len(decisions) * 2 == len(domain_params.input_lowers) 
    assert len(decisions) * 2 == len(domain_params.sat_solvers) 
    
    # new solver
    new_sat_solver = copy.deepcopy(domain_params.sat_solvers[batch_idx])
    
    # TODO: Fixme: generalize this (do not use list)
    lid, nid = decisions[batch_idx % len(decisions)]
    lname = self.net.split_nodes[lid].name
    
    # new decision
    variable = self.var_mapping[lname, nid]
    literal = variable if batch_idx < len(decisions) else -variable

    # assign
    if not new_sat_solver.assign(literal):
        logger.debug('[!] Assign conflicted')
        return None
    
    # propagation
    bcp_stat, bcp_vars = new_sat_solver.bcp()
    if not bcp_stat: # conflict
        return None

    if len(bcp_vars):
        # print('BCP', bcp_vars, 'batch_idx', batch_idx)
        update_stats = [self.update_hidden_bounds_histories(
                            lower_bounds=domain_params.lower_bounds, 
                            upper_bounds=domain_params.upper_bounds, 
                            histories=domain_params.histories, 
                            literal=lit, 
                            batch_idx=batch_idx) 
                        for lit in bcp_vars]

        if not all(update_stats):
            logger.debug('[!] BCP assign conflicted')
            return None

    return new_sat_solver


@beartype
def save_conflict_clauses(self: 'heuristic.domains_list.DomainsList', domain_params: AbstractResults, remaining_index: torch.Tensor) -> None:
    assert domain_params.objective_ids is not None
    for i in range(len(domain_params.histories)):
        if i in remaining_index:
            continue
        self.all_conflict_clauses[int(domain_params.objective_ids[i])].append(domain_params.histories[i])