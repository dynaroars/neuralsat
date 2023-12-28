import torch
import time
import copy

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


def _histories_to_clauses(histories, var_mapping):
    # TODO:
    raise
    clauses = []
    for history in histories:
        literals = []
        for lid, lds in enumerate(history):
            var_names, signs = lds
            var_names = [var_mapping[lid, v] for v in var_names]
            signs = [-int(s) for s in signs]
            literals += [v * s for v, s in zip(var_names, signs)]
        clauses.append(literals)
    # clauses += [[i+1, -(i+1)] for i in range(len(var_mapping))] # init clauses
    return clauses
    
    
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
    lid, nid = self.reversed_var_mapping[abs(literal)]
    # update histories
    histories[lid][0].append(nid)
    histories[lid][1].append(1.0 if literal > 0 else -1.0)
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
    # TODO:
    raise
    tic = time.time()
    assert lower_bounds[0].shape[0] == 1
    layer_sizes = [_.flatten(start_dim=1).shape[-1] for _ in lower_bounds[:-1]]
    var_mapping = {}
    for lid, layer_size in enumerate(layer_sizes):
        for nid in range(layer_size):
            var_mapping[lid, nid] = 1 + nid + sum(layer_sizes[:lid])
    
    # initial learned conflict clauses
    clauses = _histories_to_clauses(preconditions, var_mapping)
    
    # masks: 1 for active, -1 for inactive, 0 for unstable
    masks = [((lower_bounds[j] > 0).flatten().to(torch.get_default_dtype()) - (upper_bounds[j] < 0).flatten().to(torch.get_default_dtype())).detach().cpu() 
                for j in range(len(lower_bounds) - 1)]
    literals_to_assign = []
    for lid, lmask in enumerate(masks):
        for nid, nstatus in enumerate(lmask):
            if nstatus != 0:
                literal = int(nstatus * var_mapping[lid, nid])
                literals_to_assign.append(literal)
            
    # create sat solver
    new_sat_solver = SATSolver(clauses)
    if not new_sat_solver.multiple_assign(literals_to_assign):
        return False
    
    # propagation
    bcp_stat, bcp_vars = new_sat_solver.bcp()
    if not bcp_stat: # conflict
        return False
                
    # save
    self.var_mapping = var_mapping
    self.reversed_var_mapping = {v: k for k, v in var_mapping.items()}
    assert len(self.var_mapping) == len(self.reversed_var_mapping)
    self.all_sat_solvers = [copy.deepcopy(new_sat_solver)]
    
    # update bcp variables
    for literal in bcp_vars:
        if not self.update_hidden_bounds_histories(lower_bounds, upper_bounds, histories, literal, batch_idx=0):
            return False
        
    logger.debug('Create SAT solver:', time.time() - tic)
    return True
    
    
def boolean_propagation(self, domain_params, batch_idx):
    # TODO
    raise
    batch = len(domain_params.input_lowers)
    idx = batch_idx % batch
    
    # new solver
    new_sat_solver = copy.deepcopy(domain_params.sat_solvers[idx])
    
    # new decision
    variable = self.var_mapping[decisions[idx][0], decisions[idx][1]]
    literal = variable if batch_idx < batch else -variable
    
    # assign
    if not new_sat_solver.assign(literal):
        logger.debug('[!] Assign conflicted')
        return None
    
    # propagation
    bcp_stat, bcp_vars = new_sat_solver.bcp()
    if not bcp_stat: # conflict
        return None
    
    if len(bcp_vars):
        # print('BCP', bcp_vars)
        update_stats = [self.update_hidden_bounds_histories(
                            lower_bounds=domain_params.lower_bounds, 
                            upper_bounds=domain_params.upper_bounds, 
                            histories=new_history, 
                            literal=lit, 
                            batch_idx=batch_idx) for lit in bcp_vars]
        if not all(update_stats):
            logger.debug('[!] BCP assign conflicted')
            return None
        
    return new_sat_solver


def save_conflict_clauses(self, domain_params, remaining_index):
    for idx_ in range(len(domain_params.histories)):
        if idx_ in remaining_index:
            continue
        self.all_conflict_clauses.append(domain_params.histories[idx_])