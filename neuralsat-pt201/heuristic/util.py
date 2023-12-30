import torch
import time
import copy

from abstractor.utils import _append_tensor
from solver.sat_solver import SATSolver
from util.misc.logger import logger
from auto_LiRPA.bound_ops import *

def compute_masks(lower_bounds, upper_bounds, device):
    new_masks = {
        j: torch.logical_and(
                    lower_bounds[j] < 0, 
                    upper_bounds[j] > 0).flatten(start_dim=1).to(torch.get_default_dtype()).to(device=device, non_blocking=True)
        for j in lower_bounds
    }
    return new_masks


def _compute_babsr_scores(abstractor, lower_bounds, upper_bounds, lAs, batch, masks, reduce_op, number_bounds):
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


def _history_to_clause(h, name_mapping):
    clause = []
    for lname, ldata in h.items():
        assert sum(ldata[2]) == 0 # TODO: fixme
        # extract data
        var_names, signs = ldata[0], ldata[1]
        # convert data
        var_names = [name_mapping[lname, int(v)] for v in var_names]
        # negation
        signs = [-int(s) for s in signs]
        # append literals
        clause += [v * s for v, s in zip(var_names, signs)]
    return clause


def _compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = upper_bound.clamp(min=0)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept

    
def _get_bias_term(input_node, ratio):
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
    
    
def update_hidden_bounds_histories(self, lower_bounds, upper_bounds, histories, literal, batch_idx):
    assert literal != 0
    assert lower_bounds is not None
    # extract decision
    lid, nid = self.reversed_var_mapping[abs(literal)]
    
    history = histories[batch_idx] if isinstance(histories, list) else histories

    # update histories
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


def init_sat_solver(self, lower_bounds, upper_bounds, histories, preconditions):
    # variables mapping from variable to (lid, nid)
    tic = time.time()
    
    # TODO: fix batch > 1
    assert all([v.shape[0] == 1 for v in lower_bounds.values()])
 
    # initial learned conflict clauses
    clauses = [_history_to_clause(c, self.var_mapping) for c in preconditions]
    # print(clauses)
    
    # masks: 1 for active, -1 for inactive, 0 for unstable
    masks = {
        k: ((lower_bounds[k] > 0).flatten().int() - (upper_bounds[k] < 0).flatten().int()).detach().cpu() 
            for k in lower_bounds if k != self.net.final_name
    }
    
    # assign
    literals_to_assign = []
    for lname, lmask in masks.items():
        for nid, nstatus in enumerate(lmask):
            if nstatus != 0:
                literal = int(nstatus * self.var_mapping[lname, nid])
                literals_to_assign.append(literal)
            
    # create sat solver
    new_sat_solver = SATSolver(clauses)
    if not new_sat_solver.multiple_assign(literals_to_assign):
        return False
    
    # propagation
    bcp_stat, bcp_vars = new_sat_solver.bcp()
    if not bcp_stat: # conflict
        return False
          
    # TODO: fix batch > 1
    # save
    self.all_sat_solvers = [copy.deepcopy(new_sat_solver)]
    
    # update bcp variables
    for literal in bcp_vars:
        if not self.update_hidden_bounds_histories(lower_bounds, upper_bounds, histories, literal, batch_idx=0):
            return False
        
    logger.debug('Create SAT solver:', time.time() - tic)
    return True
    
    
def boolean_propagation(self, domain_params, decisions, batch_idx):
    assert len(decisions) * 2 == len(domain_params.input_lowers) 
    assert len(decisions) * 2 == len(domain_params.sat_solvers) 
    
    # new solver
    new_sat_solver = copy.deepcopy(domain_params.sat_solvers[batch_idx])
    
    # TODO: Fixme: generalize this
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


def save_conflict_clauses(self, domain_params, remaining_index):
    for idx_ in range(len(domain_params.histories)):
        if idx_ in remaining_index:
            continue
        self.all_conflict_clauses.append(domain_params.histories[idx_])