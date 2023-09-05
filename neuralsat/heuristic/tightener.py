import gurobipy as grb
import multiprocessing
import torch
import copy
import time
import sys
import os

from auto_LiRPA.bound_ops import BoundRelu
from setting import Settings

MULTIPROCESS_MODEL = None
DEBUG = True

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
        for c_ in model.getConstrs():
            if c_.ConstrName == f'{var_name}_eq':
                # print('skip', c_.ConstrName)
                continue
            if _get_prefix_constr_name(c_.ConstrName) in all_remove_patterns:
                remove_constrs.append(c_)
                # remove_constrs.append(c_.ConstrName)
            # print(c_.ConstrName, _get_prefix_constr_name(c_.ConstrName), )
            
        # remove variables
        for v_ in model.getVars():
            if v_.VarName == var_name:
                # print('skip', var_name)
                continue
            if _get_prefix_var_name(v_.VarName) in all_remove_patterns:
                remove_vars.append(v_)
                # remove_vars.append(v_.VarName)
            # print(v_.VarName)
        
        # print(remove_constrs)
        # print(remove_vars)
        # print()
        # print()
        # print()
        model.remove(remove_constrs)
        model.remove(remove_vars)
        model.update()
        # model.write('example/test_gurobi_removed.lp')
        
        # exit()

    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9: # Timed out. Get current bound.
            bound = bound_type(grb_model.objbound, reference)
            refined = abs(bound - reference) >= 1e-4
        elif grb_model.status == 2: # Optimally solved.
            bound = grb_model.objbound
            refined = abs(bound - reference) >= 1e-4
            # refined = True
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
        vub, refined, status_ub = get_grb_solution(model, out_ub, min)
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
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max)
        return vlb, refined, status_lb, status_lb_r

    model = MULTIPROCESS_MODEL.copy()
    var_name, pre_relu_names, relu_names, final_name = candidate
    v = model.getVarByName(var_name)
    out_lb, out_ub = v.lb, v.ub
    refine_time = time.time()
    neuron_refined = False
    eps = 1e-5

    solve_both = False
    
    remove_unused_vars_and_constrs(model, var_name, {v: k for k, v in pre_relu_names.items()}, relu_names, final_name)
    
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

    return vlb, vub, neuron_refined



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
    
    def __init__(self, abstractor):
        self.abstractor = abstractor
        assert abstractor.net.model.ModelName == 'mip', print(f'Model error: "{abstractor.net.model.ModelName}" != "mip"')
        self.mip_model = abstractor.net.model.copy()
        
        # self.pre_relu_indices = [i for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations) if isinstance(layer, BoundRelu)]
        assert len(abstractor.net.relus) == len(abstractor.net.perturbed_optimizable_activations), print('[!] Error: Support ReLU only')
        self.pre_relu_names = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.relu_names = {i: layer.name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
               
        
    # def test(self, domain_params):
    #     if self.abstractor.input_split:
    #         return domain_params
        
    #     if not Settings.use_mip_refine_domain_bounds:
    #         return domain_params
        
    #     assert len(self.abstractor.pre_relu_indices) == len(domain_params.lower_bounds) - 1, print('Support ReLU only')
        
    #     remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]
    #     for idx in remaining_index:
    #         cur_intermediate_layer_bounds = {
    #             self.abstractor.name_dict[d]: [
    #                 domain_params.lower_bounds[d][idx][None].clone(),
    #                 domain_params.upper_bounds[d][idx][None].clone(),
    #             ] for d in self.abstractor.pre_relu_indices # exclude output layer
    #         } 
    #         cur_input_lowers = domain_params.input_lowers[idx][None]
    #         cur_input_uppers = domain_params.input_uppers[idx][None]
    #         # print('refined bounds before:', sum([(v[1] - v[0]).sum().item() for _, v in cur_intermediate_layer_bounds.items()]))
                
    #         # tic = time.time()
    #         self.abstractor.build_lp_solver(
    #             model_type='mip', 
    #             input_lower=cur_input_lowers, 
    #             input_upper=cur_input_uppers, 
    #             c=None, 
    #             intermediate_layer_bounds=copy.deepcopy(cur_intermediate_layer_bounds),
    #             # intermediate_layer_bounds=cur_intermediate_layer_bounds,
    #             timeout_per_neuron=1.0,
    #             refine=True,
    #         )
    #         # print(idx, 'refine in:', time.time() - tic)
    #         # print('refined bounds after:', sum([(v[1] - v[0]).sum().item() for _, v in cur_intermediate_layer_bounds.items()]))
                
    #         new_intermediate_layer_bounds = self.abstractor.net.get_refined_intermediate_bounds()
            
    #         # for k in new_intermediate_layer_bounds:
    #         #     old_bounds = cur_intermediate_layer_bounds[k]
    #         #     new_bounds = new_intermediate_layer_bounds[k]
    #         #     print_tightened_bounds(name=k, olds=old_bounds, news=new_bounds)
            
    #         for i_ in self.abstractor.pre_relu_indices:
    #             new_l, new_u = new_intermediate_layer_bounds[self.abstractor.name_dict[i_]]
    #             domain_params.lower_bounds[i_][idx] = new_l.clone()
    #             domain_params.upper_bounds[i_][idx] = new_u.clone()
            
    #     return domain_params
    
    
    def __call__(self, domain_list):
        # return
        worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')
        batch = len(worst_domains.lower_bounds[0])
        print('[+] Tightening:', batch)
        # self.mip_model.write(f'example/test_gurobi_all.lp')
        # print(self.pre_relu_indices)
        print(self.pre_relu_names)
        print(self.relu_names)
        # print(worst_domains.lower_bounds[-1].flatten())
        print(self.mip_model)
        # exit()
        # FIXME: only work with ReLU
        # domain_activations = [((worst_domains.lower_bounds[j] == 0).int() - (worst_domains.upper_bounds[j] == 0).int()).flatten(1).cpu() for j in range(len(worst_domains.lower_bounds) - 1)]
        # # print([_.shape for _ in domain_activations])
        # for i in range(batch):
        #     print('domain_activations:', i, [da[i].abs().sum() for da in domain_activations])
        
        # repeat_domain_activations = [_[0:1].repeat(batch, 1) for _ in domain_activations]
        print([_.shape for _ in worst_domains.lower_bounds[:-1]])
        unified_lower_bounds = [_.min(dim=0).values for _ in worst_domains.lower_bounds[:-1]]
        unified_upper_bounds = [_.max(dim=0).values for _ in worst_domains.upper_bounds[:-1]]
        print([_.shape for _ in unified_upper_bounds])
        
        assert all([(u_lb <= o_lb).all() for u_lb, o_lb in zip(unified_lower_bounds, worst_domains.lower_bounds[:-1])])
        assert all([(u_ub >= o_ub).all() for u_ub, o_ub in zip(unified_upper_bounds, worst_domains.upper_bounds[:-1])])
        print()
        
        
        # step 1: update bounds
        # FIXME: support other activation
        assert len(self.pre_relu_names) == len(unified_lower_bounds)
        current_model = self.mip_model.copy()
        tic = time.time()
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
        current_model.update()
        print('update:', time.time() - tic)
        
        # step 2: select candidates
        tic = time.time()
        unified_masks = [torch.where(lb_ * ub_ < 0)[0].numpy() for (lb_, ub_) in zip(unified_lower_bounds, unified_upper_bounds)]
        unified_indices = [(l_id, n_id) for l_id in range(len(unified_masks)) for n_id in unified_masks[l_id]]
        
        unified_scores = torch.concat([torch.min(unified_upper_bounds[l_id][unified_masks[l_id]].abs(), unified_lower_bounds[l_id][unified_masks[l_id]].abs()).flatten() for l_id in range(len(unified_masks))])
        # unified_scores = [unified_upper_bounds[l_id] for l_id in range(len(unified_masks))]
        
        # print(unified_masks)
        # print(unified_indices)
        # print(unified_scores, len(unified_indices))
        # print(unified_scores.topk(5), len(unified_indices))
        all_candidates = sum(_.numel() for _ in unified_scores)
        n_candidates = min(96, all_candidates)
        
        for idx, (l_id, n_id) in enumerate(unified_indices):
            assert torch.min(unified_upper_bounds[l_id][n_id], unified_lower_bounds[l_id][n_id].abs()).item() == unified_scores[idx]
        
        candidates = []
        select_indices = unified_scores.topk(n_candidates, largest=False).indices
        print(f'select {n_candidates} indices:', select_indices)
        candidates += [(f"lay{self.pre_relu_names[unified_indices[select_idx][0]]}_{unified_indices[select_idx][1]}", self.pre_relu_names, self.relu_names, self.abstractor.net.final_name) for select_idx in select_indices]
        # candidates += [f"lay{self.pre_relu_names[unified_indices[select_idx][0]]}_{unified_indices[select_idx][1]}" for select_idx in unified_scores.topk(n_candidates, largest=False).indices]
        # candidates = [candidates[3]]
        print('select:', time.time() - tic)
        
        global MULTIPROCESS_MODEL
        MULTIPROCESS_MODEL = current_model.copy()
        MULTIPROCESS_MODEL.setParam('TimeLimit', 20.0)
        MULTIPROCESS_MODEL.setParam('Threads', 1)
        MULTIPROCESS_MODEL.setParam('MIPGap', 0.01)
        MULTIPROCESS_MODEL.setParam('MIPGapAbs', 0.01)
        # MULTIPROCESS_MODEL.setParam('Threads', 128 // len(candidates))
        print(MULTIPROCESS_MODEL)
        # exit()
        
        # for can in candidates:
        #     mip_solver_worker(can)
        tic = time.time()
        # print(candidates)
        with multiprocessing.Pool(min(len(candidates), os.cpu_count())) as pool:
            solver_result = pool.map(mip_solver_worker, candidates, chunksize=1)
        MULTIPROCESS_MODEL = None
        print('refine:', sum([_[-1] for _ in solver_result]), len(candidates), time.time() - tic)
        
        # if len(domain_list) > 3:
        exit()
        # for i in range(batch):
        #     print('repeat_domain_activations:', i, [rda[i].abs().sum() for rda in repeat_domain_activations])
        # # print('repeat_domain_activations:', [_.abs().sum() for _ in repeat_domain_activations])
        # print()
        # print()
        
        # print([(i==ii).shape for i, ii in zip(domain_activations, repeat_domain_activations)])
        # print([(i==ii).all() for i, ii in zip(domain_activations, repeat_domain_activations)])
        # print([torch.where(i == ii, i, 0) * (i != 0) for i, ii in zip(domain_activations, repeat_domain_activations)])
        
        # print('repeat_domain_activations:', repeat_domain_activations)
        # print([i == ii for (i, ii) in zip(domain_activations, repeat_domain_activations)])
        # print([_.shape for _ in repeat_domain_activations])
        
        # 1. get worst domains
        # 2. unify bounds
        # 3. update mip model
        # 4. select top-k candidates
        # 5. refine
        # 6. update bounds
        
    def _find_common_activation(self, domain_params):
        pass