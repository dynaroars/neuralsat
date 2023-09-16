import gurobipy as grb
import multiprocessing
import numpy as np
import random
import torch
import copy
import time
import sys
import os

from auto_LiRPA.bound_ops import BoundRelu
from util.misc.logger import logger
from setting import Settings

MULTIPROCESS_MODEL = None
DEBUG = True

REMOVE_UNUSED = True

def _bound_improvement(orig, refined, bound_type):
    assert len(orig) == len(refined)
    if bound_type == 'lower':
        assert all([(i <= ii).all() for (i, ii) in zip(orig, refined)])
        return [(ii - i).sum() for (i, ii) in zip(orig, refined)]
    assert all([(i >= ii).all() for (i, ii) in zip(orig, refined)])
    return [(i - ii).sum() for (i, ii) in zip(orig, refined)]
        
def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise
  
def _get_prefix_constr_name(name):
    if name.startswith('lay'):
        return ''.join(name.split('_')[:-2])
    return ''.join(name.split('_')[:-3])

def _get_prefix_var_name(name):
    return ''.join(name.split('_')[:-1])
    
    
def mip_solver_worker(candidate):
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
        status_ub_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        # model.write(f'example/test_gurobi_ub.lp')
        
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min, eps=eps)
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5):
        status_lb_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        # model.write(f'example/test_gurobi_lb.lp')
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max, eps=eps)
        return vlb, refined, status_lb, status_lb_r

    refine_time = time.time()
    model = MULTIPROCESS_MODEL.copy()
    l_id, n_id, var_name, solve_both, pre_relu_names, relu_names, final_name = candidate
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
        if vlb <= 0 and solve_both: # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # lb > 0, neuron is stable, we skip solving ub.
            vub, status_ub, status_ub_r = out_ub, -1, -1
    else: # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps)
        neuron_refined = neuron_refined or refined
        if vub >= 0 and solve_both: # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps)
            neuron_refined = neuron_refined or refined
        else: # ub < 0, neuron is stable, we skip solving ub.
            vlb, status_lb, status_lb_r = out_lb, -1, -1

    if DEBUG:
        # print(model)
        if neuron_refined:
            print(f"Solving MIP for {v.VarName:<10}: [{out_lb:.6f}, {out_ub:.6f}]=>[{vlb:.6f}, {vub:.6f}] ({status_lb}, {status_ub}), time: {time.time()-refine_time:.4f}s, #vars: {model.NumVars}, #constrs: {model.NumConstrs}")
        else:
            pass
            # print(f"Solving MIP for {v.VarName:<10}: [{out_lb:.6f}, {out_ub:.6f}] ({status_lb}, {status_ub}), time: {time.time()-refine_time:.4f}s")
        sys.stdout.flush()

    return l_id, n_id, var_name, vlb, vub, neuron_refined, status_lb, status_ub



def print_tightened_bounds(name, olds, news):
    old_lowers, old_uppers = olds
    new_lowers, new_uppers = news
    
    old_lowers = old_lowers.flatten().detach().cpu()#.numpy()
    old_uppers = old_uppers.flatten().detach().cpu()#.numpy()
    
    new_lowers = new_lowers.flatten().detach().cpu()#.numpy()
    new_uppers = new_uppers.flatten().detach().cpu()#.numpy()
    
    print(f'[+] Layer: {name}')
    for i in range(len(old_lowers)):
        if old_lowers[i] * old_uppers[i] < 0:
            if (new_lowers[i] - old_lowers[i]).abs() > 1e-4 or (new_uppers[i] - old_uppers[i]).abs() > 1e-4:
                print(f'\t- neuron {i}: [{old_lowers[i]:.04f}, {old_uppers[i]:.04f}] => [{new_lowers[i]:.04f}, {new_uppers[i]:.04f}]')

class Tightener:
    
    def __init__(self, abstractor, objectives):
        self.abstractor = abstractor
        # self.objectives = copy.deepcopy(objectives)
        self.input_lowers = objectives.lower_bounds[0].clone().view(self.abstractor.input_shape).to(self.abstractor.device)
        self.input_uppers = objectives.upper_bounds[0].clone().view(self.abstractor.input_shape).to(self.abstractor.device)
        self.c_to_use = objectives.cs.clone().transpose(0, 1).to(self.abstractor.device)
        
        # assert abstractor.net.model.ModelName == 'mip', print(f'Model error: "{abstractor.net.model.ModelName}" != "mip"')
        # self.orig_mip_model = abstractor.net.model.copy()
        
        # self.pre_relu_indices = [i for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations) if isinstance(layer, BoundRelu)]
        assert len(abstractor.net.relus) == len(abstractor.net.perturbed_optimizable_activations), print('[!] Error: Support ReLU only')
        self.pre_relu_names = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.relu_names = {i: layer.name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.black_list = []
        self.tightened_layers = []
        
        
    def reset(self):
        self.black_list = []
        self.tightened_layers = []
        
        
    def select_layer(self):
        for i in range(1, len(self.relu_names)):
            if i not in self.tightened_layers:
                return i
        return random.choice(self.tightened_layers)
        
        
    # FIXME: only work with ReLU
    def __call__(self, domain_list, topk=64, largest=False, timeout=2.0, solve_both=False):
        # step 1: select domains
        worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')
        batch = len(worst_domains.lower_bounds[0])
        logger.debug(f'Tightening: {batch}')
        if batch == 0:
            return
        
        # worst bounds
        unified_lower_bounds = [_.min(dim=0).values.flatten() for _ in worst_domains.lower_bounds[:-1]]
        unified_upper_bounds = [_.max(dim=0).values.flatten() for _ in worst_domains.upper_bounds[:-1]]
        assert all([(u_lb <= o_lb.flatten(1)).all() for u_lb, o_lb in zip(unified_lower_bounds, worst_domains.lower_bounds[:-1])])
        assert all([(u_ub >= o_ub.flatten(1)).all() for u_ub, o_ub in zip(unified_upper_bounds, worst_domains.upper_bounds[:-1])])
        
        # assert all([(u_lb <= o_lb.data.flatten(1)).all() for u_lb, o_lb in zip(unified_lower_bounds, domain_list.all_lower_bounds[:-1])])
        # assert all([(u_ub >= o_ub.data.flatten(1)).all() for u_ub, o_ub in zip(unified_upper_bounds, domain_list.all_upper_bounds[:-1])])
        
        # unified_lower_bounds_cl = [_.clone() for _ in unified_lower_bounds]
        # unified_upper_bounds_cl = [_.clone() for _ in unified_upper_bounds]
        # current_model = self.rebuild_mip_model(unified_lower_bounds, unified_upper_bounds)
        # current_model.setParam('TimeLimit', timeout)
        
        # update bounds
        assert len(self.pre_relu_names) == len(unified_lower_bounds)
        # tic = time.time()
        if 0:
            for layer_idx, pre_relu_name in self.pre_relu_names.items():
                for neuron_idx in range(unified_lower_bounds[layer_idx].numel()):
                    var = current_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
                    assert var is not None
                    # print(layer_idx, neuron_idx, var)
                    # if var.LB > unified_lower_bounds[layer_idx][neuron_idx] + 1e-3: print('lb', var.LB, unified_lower_bounds[layer_idx][neuron_idx].item())
                    # if var.UB < unified_upper_bounds[layer_idx][neuron_idx] - 1e-3: print('ub', var.UB, unified_upper_bounds[layer_idx][neuron_idx].item())
                    
                    var.LB = max(var.LB, unified_lower_bounds[layer_idx][neuron_idx].item())
                    var.UB = min(var.UB, unified_upper_bounds[layer_idx][neuron_idx].item())
                    
                    a_var = current_model.getVarByName(f"aReLU{self.relu_names[layer_idx]}_{neuron_idx}")
                    if a_var is not None:
                        if unified_lower_bounds[layer_idx][neuron_idx] >= 0:
                            a_var.LB = 1
                            a_var.UB = 1
                        elif unified_upper_bounds[layer_idx][neuron_idx] <= 0:
                            a_var.LB = 0
                            a_var.UB = 0
                            # print(unified_lower_bounds[layer_idx][neuron_idx], unified_upper_bounds[layer_idx][neuron_idx])
                    # print(layer_idx, neuron_idx, var, var.lb, var.ub, a_var)
                    
                    # if var.LB * unified_lower_bounds[layer_idx][neuron_idx] < 0:
                    #     print('\t- lower:', layer_idx, neuron_idx, var, var.LB, unified_lower_bounds[layer_idx][neuron_idx])
                    # if var.UB * unified_upper_bounds[layer_idx][neuron_idx] < 0:
                    #     print('\t- upper:', layer_idx, neuron_idx, var, var.UB, unified_upper_bounds[layer_idx][neuron_idx])
        
        
        # step 2: select candidates
        unified_masks = [torch.where(lb_ * ub_ < 0)[0].numpy() for (lb_, ub_) in zip(unified_lower_bounds, unified_upper_bounds)]
        unified_indices = [(l_id, n_id) for l_id in range(1, len(unified_masks)) for n_id in unified_masks[l_id]] # skip first layer
        
        unified_scores = torch.concat([
            torch.min(unified_upper_bounds[l_id][unified_masks[l_id]].abs(), unified_lower_bounds[l_id][unified_masks[l_id]].abs()).flatten() 
                for l_id in range(1, len(unified_masks)) # skip first layer
        ])
        assert unified_scores.numel() == len(unified_indices)
        
        if not len(unified_indices):
            return
        
        n_candidates = len(unified_indices)
        candidates = []
        selected_indices = unified_scores.topk(n_candidates, largest=largest).indices
        selected_layer = self.select_layer()
        
        for s_idx in selected_indices:
            l_id, n_id = unified_indices[s_idx]
            if l_id != selected_layer:
                continue
            
            var_name = f"lay{self.pre_relu_names[l_id]}_{n_id}"
            if var_name in self.black_list:
                continue
            
            candidates.append((
                l_id, 
                n_id, 
                var_name, 
                solve_both, 
                self.pre_relu_names, 
                self.relu_names, 
                self.abstractor.net.final_name,
            ))
            # print('added:', l_id, n_id, var_name)
            
            if selected_layer not in self.tightened_layers: # 1st time, tighten all neurons
                continue
            
            if len(candidates) == topk: # 2nd time, tighten topk neurons
                break
        
        if selected_layer not in self.tightened_layers:
            self.tightened_layers.append(selected_layer)


        # step 3: rebuild mip model
        # assert all([(i==ii).all() for (i, ii) in zip(unified_upper_bounds, unified_upper_bounds_cl)])
        # assert all([(i==ii).all() for (i, ii) in zip(unified_lower_bounds, unified_lower_bounds_cl)])
        
        current_model = self.rebuild_mip_model(unified_lower_bounds, unified_upper_bounds)
        current_model.setParam('TimeLimit', timeout)
        global MULTIPROCESS_MODEL
        MULTIPROCESS_MODEL = current_model.copy()


        # step 4: tightening
        solver_result = []
        if len(candidates):
            with multiprocessing.Pool(min(len(candidates), os.cpu_count())) as pool:
                solver_result = pool.map(mip_solver_worker, candidates, chunksize=1)
        MULTIPROCESS_MODEL = None


        # step 5: update refined bounds
        unified_lower_bounds_refined = [lb.clone() for lb in unified_lower_bounds]
        unified_upper_bounds_refined = [ub.clone() for ub in unified_upper_bounds]
        unstable_to_stable_neurons = []
        num_neuron_refined = 0
        for l_id, n_id, var_name, vlb, vub, neuron_refined, s_lb, s_ub in solver_result:
            # print(l_id, n_id, var_name, vlb, vub, neuron_refined, s_lb, s_ub)
            if neuron_refined:
                num_neuron_refined += 1 
                unified_lower_bounds_refined[l_id][n_id] = max(unified_lower_bounds[l_id][n_id], vlb)
                unified_upper_bounds_refined[l_id][n_id] = min(unified_upper_bounds[l_id][n_id], vub)
                if vlb > 0:
                    unstable_to_stable_neurons.append((l_id, n_id, 1.0))
                elif vub < 0:
                    unstable_to_stable_neurons.append((l_id, n_id, -1.0))

                # print(f'neuron[{l_id}][{n_id}]: [{unified_lower_bounds[l_id][n_id]:.06f}, {unified_upper_bounds[l_id][n_id]:.06f}] => [{unified_lower_bounds_refined[l_id][n_id]:.06f}, {unified_upper_bounds_refined[l_id][n_id]:.06f}]')
                
            # if (s_lb, s_ub) == (2, 2):
            #     self.black_list.append(var_name)
                    
        logger.debug(f'Selected {len(candidates)}/{len(unified_indices)}, tightened {num_neuron_refined} neurons, stabilized {len(unstable_to_stable_neurons)} neurons, blacklisted {len(self.black_list)} neurons')
        
        
        # step 6: update domains bounds
        class TMP:
            pass
        
        refined_domain = TMP()
        refined_domain.lower_bounds = unified_lower_bounds_refined
        refined_domain.upper_bounds = unified_upper_bounds_refined
        
        domain_list.update_refined_bounds(refined_domain)
        
        return
        
        
    def rebuild_mip_model(self, refined_lower_bounds, refined_upper_bounds):
        intermediate_layer_bounds = {}
        for l_id, l_name in self.pre_relu_names.items():
            intermediate_layer_bounds[l_name] = [
                refined_lower_bounds[l_id].to(self.abstractor.device), 
                refined_upper_bounds[l_id].to(self.abstractor.device)
            ]
        
        # for name, (lbs, ubs) in intermediate_layer_bounds.items():
        #     print(name, lbs.shape)
            
        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=self.input_lowers,
            input_upper=self.input_uppers,
            c=self.c_to_use,
            refine=False,
            intermediate_layer_bounds=intermediate_layer_bounds,
        )
        current_model = self.abstractor.net.model.copy()
        current_model.setParam('Threads', 1)
        current_model.setParam('MIPGap', 0.01)
        current_model.setParam('MIPGapAbs', 0.01)
        current_model.update()
        
        return current_model