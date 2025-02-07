import gurobipy as grb
import multiprocessing
import numpy as np
import time
import sys
import os

MULTIPROCESS_MODEL = None
REMOVE_UNUSED = True
DEBUG = False

def _gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise NotImplementedError()
  
  
def _get_prefix_constr_name(name):
    if name.startswith('lay'):
        return ''.join(name.split('_')[:-2])
    return ''.join(name.split('_')[:-3])

def _get_prefix_var_name(name):
    return ''.join(name.split('_')[:-1])
    

def _mip_solver_worker(candidate):
    """ Multiprocess worker for solving MIP models in build_the_model_mip_refine """
    
    def remove_unused_vars_and_constrs(grb_model, var_name, pre_activation_names, activation_names, final_name):
        # print('removing some variables and constraints', var_name, pre_activation_names, activation_names)
        
        current_layer_name = ''.join(var_name.split('_')[:-1])[3:] # remove "lay" and "_{nid}"
        assert current_layer_name in pre_activation_names
        current_layer_id = pre_activation_names[current_layer_name]
        # print('current_layer_name:', current_layer_name)
        # print('current_layer_id:', current_layer_id)
        
        remove_pre_activation_patterns = [f'lay{k}' for k, v in pre_activation_names.items() if v >= current_layer_id]
        remove_pre_activation_patterns += [f'lay{final_name}']
        remove_activation_patterns = [f'ReLU{v}' for k, v in activation_names.items() if k >= current_layer_id]
        remove_activation_patterns += [f'aReLU{v}' for k, v in activation_names.items() if k >= current_layer_id]
        # print('remove_pre_activation_patterns:', remove_pre_activation_patterns)
        # print('remove_activation_patterns:', remove_activation_patterns)
        all_remove_patterns = remove_pre_activation_patterns + remove_activation_patterns
        remove_vars = []
        remove_constrs = []
        
        # remove constraints
        for c_ in grb_model.getConstrs():
            if c_.ConstrName == f'{var_name}_eq':
                # print('skip', c_.ConstrName)
                continue
            if _get_prefix_constr_name(c_.ConstrName) in all_remove_patterns:
                remove_constrs.append(c_)
                # remove_constrs.append(c_.ConstrName)
            # print(c_.ConstrName, _get_prefix_constr_name(c_.ConstrName), )
            
        # remove variables
        for v_ in grb_model.getVars():
            if v_.VarName == var_name:
                # print('skip', var_name)
                continue
            if _get_prefix_var_name(v_.VarName) in all_remove_patterns:
                remove_vars.append(v_)
                # remove_vars.append(v_.VarName)
            # print(v_.VarName)
        
        grb_model.remove(remove_constrs)
        grb_model.remove(remove_vars)
        grb_model.update()
        # grb_model.write('example/test_gurobi_removed.lp')
        

    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9: # Timed out. Get current bound.
            bound = bound_type(grb_model.objbound, reference)
            refined = abs(bound - reference) >= eps
        elif grb_model.status == 2: # Optimally solved.
            bound = grb_model.objbound
            refined = abs(bound - reference) >= eps
        elif grb_model.status == 15: # Found an lower bound >= 0 or upper bound <= 0, so this neuron becomes stable.
            bound = bound_type(1., -1.) * eps
            refined = True
        else:
            bound = reference
        return bound, refined, grb_model.status

    def solve_ub(model, v, out_ub, eps=1e-5):
        status_ub_r = -1  # Gurobi solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        # model.write(f'example/test_gurobi_ub.lp')
        
        try:
            model.optimize()
        except grb.GurobiError as e:
            _gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min, eps=eps)
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5):
        status_lb_r = -1  # Gurobi solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        # model.write(f'example/test_gurobi_lb.lp')
        try:
            model.optimize()
        except grb.GurobiError as e:
            _gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max, eps=eps)
        return vlb, refined, status_lb, status_lb_r

    refine_time = time.time()
    model = MULTIPROCESS_MODEL.copy()
    l_id, n_id, var_name, pre_relu_names, relu_names, final_name = candidate
    v = model.getVarByName(var_name)
    out_lb, out_ub = v.LB, v.UB
    neuron_refined = False
    eps = 1e-5
    v.LB, v.UB = -np.inf, np.inf
    model.update()

    if REMOVE_UNUSED:
        remove_unused_vars_and_constrs(model, var_name, {v_: k_ for k_, v_ in pre_relu_names.items()}, relu_names, final_name)
    
    if abs(out_lb) < abs(out_ub): # lb is tighter, solve lb first.
        vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
        neuron_refined = neuron_refined or refined
        if vlb <= 0: # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # lb >= 0, neuron is stable, we skip solving ub.
            vlb, vub, status_ub, status_ub_r = 0.0, out_ub, -1, -1
            
    else: # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
        neuron_refined = neuron_refined or refined
        if vub >= 0: # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # ub <= 0, neuron is stable, we skip solving ub.
            vlb, vub, status_lb, status_lb_r = out_lb, 0.0, -1, -1

    if DEBUG:
        # print(model)
        if neuron_refined:
            print(f"Solving MIP for {v.VarName:<10}: [{out_lb:.6f}, {out_ub:.6f}]=>[{vlb:.6f}, {vub:.6f}] ({status_lb}, {status_ub}), time: {time.time()-refine_time:.4f}s, #vars: {model.NumVars}, #constrs: {model.NumConstrs}")
            sys.stdout.flush()

    return l_id, n_id, var_name, vlb, vub, neuron_refined, status_lb, status_ub



def stabilize(self, mip_model, candidates, unified_lower_bounds, unified_upper_bounds, timeout):
    global MULTIPROCESS_MODEL
    MULTIPROCESS_MODEL = mip_model
    MULTIPROCESS_MODEL.setParam('TimeLimit', timeout)
    
    # step 1: tightening
    solver_result = []
    if len(candidates):
        with multiprocessing.Pool(min(len(candidates), os.cpu_count())) as pool:
            solver_result = pool.map(_mip_solver_worker, candidates, chunksize=1)
    MULTIPROCESS_MODEL = None
    
    # step 2: update refined bounds
    unified_lower_bounds_refined = {k: v.clone() for k, v in unified_lower_bounds.items()}
    unified_upper_bounds_refined = {k: v.clone() for k, v in unified_upper_bounds.items()}
    unstable_to_stable_neurons = []
    num_neuron_refined = 0
    for l_id, n_id, var_name, vlb, vub, neuron_refined, s_lb, s_ub in solver_result:
        if neuron_refined:
            num_neuron_refined += 1 
            unified_lower_bounds_refined[l_id][n_id] = max(unified_lower_bounds[l_id][n_id], vlb)
            unified_upper_bounds_refined[l_id][n_id] = min(unified_upper_bounds[l_id][n_id], vub)
            if vlb >= 0:
                unstable_to_stable_neurons.append((l_id, n_id, 1.0))
            elif vub <= 0:
                unstable_to_stable_neurons.append((l_id, n_id, -1.0))
            # print(f'neuron[{l_id}][{n_id}]: [{unified_lower_bounds[l_id][n_id]:.06f}, {unified_upper_bounds[l_id][n_id]:.06f}] => [{unified_lower_bounds_refined[l_id][n_id]:.06f}, {unified_upper_bounds_refined[l_id][n_id]:.06f}]')
            
        # TODO: add neurons to blacklist
        # self.black_list.append(var_name)
        
    return unified_lower_bounds_refined, unified_upper_bounds_refined, num_neuron_refined, unstable_to_stable_neurons
