import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import torch
import time
import sys

multiprocess_mip_model = None
N_PROC = 32

def handle_gurobi_error(message):
    print(f'Gurobi error: {message}')
    raise


def copy_model(model):
    model_split = model.copy()
    model_split.update()
    return model_split
    

def mip_solver_worker(candidate, init=None):
    """ Multiprocess worker for solving MIP models in build_the_model_mip_refine """

    # print('solving:', candidate)

    def get_grb_solution(grb_model, reference, bound_type, eps=1e-5):
        refined = False
        if grb_model.status == 9: # Timed out. Get current bound.
            bound = bound_type(grb_model.objbound, reference)
            refined = bound != reference
        elif grb_model.status == 2: # Optimally solved.
            bound = grb_model.objbound
            refined = True
        elif grb_model.status == 15: # Found an lower bound >= 0 or upper bound <= 0, so this neuron becomes stable.
            bound = bound_type(1., -1.) * eps
            refined = True
        else:
            bound = reference
        return bound, refined, grb_model.status

    def solve_ub(model, v, out_ub, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
        status_ub_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MAXIMIZE)
        model.reset()
        model.setParam('BestBdStop', -eps)  # Terminiate as long as we find a negative upper bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vub, refined, status_ub = get_grb_solution(model, out_ub, min)
        return vub, refined, status_ub, status_ub_r

    def solve_lb(model, v, out_lb, eps=1e-5, init=None):
        if init is not None:
            init_x_start(model, init)
        status_lb_r = -1  # Gurbo solver status.
        model.setObjective(v, grb.GRB.MINIMIZE)
        model.reset()
        model.setParam('BestBdStop', eps)  # Terminiate as long as we find a positive lower bound.
        try:
            model.optimize()
        except grb.GurobiError as e:
            handle_gurobi_error(e.message)
        vlb, refined, status_lb = get_grb_solution(model, out_lb, max)
        return vlb, refined, status_lb, status_lb_r

    init_lb, init_ub = None, None
    if init is not None:
        init_lb, init_ub = init

    model = multiprocess_mip_model.copy()
    v = model.getVarByName(candidate)
    out_lb, out_ub = v.lb, v.ub
    refine_time = time.time()
    neuron_refined = False

    # TODO: check timeout
    # if time.time() - mip_refine_time_start >= mip_refine_timeout:
    #     return out_lb, out_ub, False

    eps = 1e-5

    if abs(out_lb) < abs(out_ub): # lb is tighter, solve lb first.
        vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps, init=init_lb)
        neuron_refined = neuron_refined or refined
        if vlb < 0: # Still unstable. Solve ub.
            vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps, init=init_ub)
            neuron_refined = neuron_refined or refined
        else: # lb > 0, neuron is stable, we skip solving ub.
            vub, status_ub, status_ub_r = out_ub, -1, -1
    else: # ub is tighter, solve ub first.
        vub, refined, status_ub, status_ub_r = solve_ub(model, v, out_ub, eps=eps, init=init_lb)
        neuron_refined = neuron_refined or refined
        if vub > 0: # Still unstable. Solve lb.
            vlb, refined, status_lb, status_lb_r = solve_lb(model, v, out_lb, eps=eps, init=init_ub)
            neuron_refined = neuron_refined or refined
        else: # ub < 0, neuron is stable, we skip solving ub.
            vlb, status_lb, status_lb_r = out_lb, -1, -1

    # print("Solving MIP for {:<10}: [{},{}]=>[{:.4f},{:.4f}], time: {:.4f}s, #vars: {}, #constrs: {}, improved: {}".format(v.VarName, out_lb, out_ub, vlb, vub, time.time()-refine_time, model.NumVars, model.NumConstrs, neuron_refined))
    # sys.stdout.flush()

    return vlb, vub, neuron_refined


def build_solver_mip(self, input_domain, lower_bounds, upper_bounds, timeout, adv_warmup=False):

    global multiprocess_mip_model

    self.mip_model = grb.Model()
    self.mip_model.setParam('OutputFlag', False)
    # self.mip_model.setParam('Threads', 3)
    self.mip_model.setParam("FeasibilityTol", 2e-5)
    self.mip_model.setParam('MIPGap', 1e-2)  # Relative gap between primal and dual.
    self.mip_model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between primal and dual.
    self.mip_model.setParam('TimeLimit', 10) # per neuron timeout

    # print(self.mip_model)

    inp_gurobi_vars = []
    gurobi_vars = []

    zero_var = self.mip_model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    if input_domain.dim() == 2:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(input_domain):
            v = self.mip_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        raise NotImplementedError
        assert input_domain.dim() == 4
        dim = 0
        for chan in range(input_domain.size(0)):
            chan_vars = []
            for row in range(input_domain.size(1)):
                row_vars = []
                for col in range(input_domain.size(2)):
                    lb = input_domain[chan, row, col, 0]
                    ub = input_domain[chan, row, col, 1]
                    v = self.mip_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{dim}')
                    row_vars.append(v)
                    dim += 1
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)
    self.mip_model.update()

    gurobi_vars.append(inp_gurobi_vars)

    new_relu_mask = []
    relu_constrs = []
    layer_idx = 1
    relu_idx = 0
    maximum_refined_relu_layers = 0
    need_refine = True
    unstable_to_stable = [[] for _ in self.net.relus]
    # print('unstable_to_stable', unstable_to_stable)

    for layer in self.layers:
        this_layer_refined = False
        new_layer_gurobi_vars = []
        if type(layer) is nn.Linear:
            out_lbs = lower_bounds[relu_idx].squeeze(0)
            out_ubs = upper_bounds[relu_idx].squeeze(0)
            weight = layer.weight.clone()
            bias = layer.bias.clone()
            if layer == self.layers[-1] and self.c is not None:
                weight = self.c.squeeze(0).mm(weight)
                bias = self.c.squeeze(0).mm(bias.unsqueeze(-1)).view(-1)
            # print('relu_idx:', relu_idx, '\tlayer_idx:', layer_idx, out_lbs.shape)

            candidates = []
            candidate_neuron_ids = []
            for neuron_idx in range(weight.size(0)):
                coeffs = weight[neuron_idx, :]
                # print(len(coeffs), len(gurobi_vars[-1]))
                lin_expr = grb.LinExpr(coeffs, gurobi_vars[-1]) + bias[neuron_idx].item()

                out_lb = out_lbs[neuron_idx].item()
                out_ub = out_ubs[neuron_idx].item()

                v = self.mip_model.addVar(lb=out_lb, ub=out_ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'lay{layer_idx}_{neuron_idx}')
                self.mip_model.addConstr(lin_expr == v)
                self.mip_model.update()

                # TODO: check timeout
                if (relu_idx >= 1 and relu_idx < len(self.net.relus)) and (out_lb * out_ub < 0): 
                    candidates.append(v.VarName)
                    candidate_neuron_ids.append(neuron_idx)

                new_layer_gurobi_vars.append(v)
        
            # update inf to all current layer bounds!!! somehow it makes solver run faster
            for vi in new_layer_gurobi_vars:
                vi.lb = -np.inf
                vi.ub = np.inf
            self.mip_model.update()

            if need_refine and (relu_idx >= 1 and relu_idx < len(self.net.relus)) and len(candidates):
                multiprocess_mip_model = self.mip_model.copy()

                if True:
                    if adv_warmup: # create pgd adv list as mip refinement warmup
                        raise NotImplementedError
                    else:
                        # TODO: #threads
                        # the second relu layer where mip refine starts
                        if True:
                            with multiprocessing.Pool(N_PROC) as pool:
                                solver_result = pool.map(mip_solver_worker, candidates, chunksize=1)
                        else:
                            solver_result = []
                            for can in candidates:
                                solver_result.append(mip_solver_worker(can))
                
                    lb_refined_sum, ub_refined_sum, refined_num = 0., 0., 0
                    for (vlb, vub, refined), neuron_idx in zip(solver_result, candidate_neuron_ids):
                        if refined:
                            vlb = max(vlb, lower_bounds[relu_idx][0, neuron_idx]) #
                            vub = min(vub, upper_bounds[relu_idx][0, neuron_idx]) #
                            refined_num += 1
                            lb_refined_sum += vlb - lower_bounds[relu_idx][0, neuron_idx]
                            ub_refined_sum += upper_bounds[relu_idx][0, neuron_idx] - vub
                            lower_bounds[relu_idx][0, neuron_idx] = vlb
                            upper_bounds[relu_idx][0, neuron_idx] = vub
                            if vlb >= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, 1))
                            if vub <= 0:
                                unstable_to_stable[relu_idx].append((neuron_idx, -1))

                        v = new_layer_gurobi_vars[neuron_idx]
                        v.lb, v.ub = lower_bounds[relu_idx][0, neuron_idx], upper_bounds[relu_idx][0, neuron_idx]
                    self.mip_model.update()

                    print(f"MIP improved {refined_num} nodes out of {len(candidates)} unstable nodes, lb improved {lb_refined_sum}, ub improved {ub_refined_sum}")
                    if refined_num > 0:
                        maximum_refined_relu_layers = relu_idx
                        this_layer_refined = True
                        last_relu_layer_refined = True
                    else:
                        need_refine = False
                        last_relu_layer_refined = False

        elif type(layer) is nn.ReLU:
            new_relu_layer_constr = []
            this_relu = self.net.relus[relu_idx]
            if isinstance(gurobi_vars[-1][0], list): # This is convolutional relu
                raise NotImplementedError
            else: # This is linear relu
                pre_lbs = lower_bounds[relu_idx].squeeze(0)
                pre_ubs = upper_bounds[relu_idx].squeeze(0)
                # print(pre_lbs.shape)
                new_layer_mask = []
                assert isinstance(gurobi_vars[-1][0], grb.Var)
                # print(len(gurobi_vars[-1]))
                for neuron_idx, pre_var in enumerate(gurobi_vars[-1]):
                    pre_ub = pre_ubs[neuron_idx].item()
                    pre_lb = pre_lbs[neuron_idx].item()
                    if pre_lb >= 0: # The ReLU is always passing
                        v = pre_var
                        new_layer_mask.append(1)
                    elif pre_ub <= 0:
                        v = zero_var
                        new_layer_mask.append(0)
                    else:
                        lb = 0
                        ub = pre_ub
                        # post-relu var
                        v = self.mip_model.addVar(ub=ub, lb=0, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'ReLU{relu_idx}_{neuron_idx}')
                        # binary indicator
                        a = self.mip_model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{relu_idx}_{neuron_idx}')

                        new_relu_layer_constr.append(self.mip_model.addConstr(pre_var - pre_lb * (1 - a) >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_0'))
                        new_relu_layer_constr.append(self.mip_model.addConstr(v >= pre_var, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_1'))
                        new_relu_layer_constr.append(self.mip_model.addConstr(pre_ub * a >= v, name=f'ReLU{relu_idx - 1}_{neuron_idx}_a_2'))

                        new_layer_mask.append(-1)

                    new_layer_gurobi_vars.append(v)

            new_relu_mask.append(torch.tensor(new_layer_mask).to(lower_bounds[0].device))
            relu_constrs.append(new_relu_layer_constr)
            relu_idx += 1

        elif type(layer) == nn.Flatten or "flatten" in str(type(layer)).lower():
            if isinstance(gurobi_vars[-1][0], list):
                for chan_idx in range(len(gurobi_vars[-1])):
                    for row_idx in range(len(gurobi_vars[-1][chan_idx])):
                        new_layer_gurobi_vars.extend(gurobi_vars[-1][chan_idx][row_idx])
            else:
                new_layer_gurobi_vars.extend(gurobi_vars[-1])
       
        else:
            print(layer)
            raise NotImplementedError

        gurobi_vars.append(new_layer_gurobi_vars)

        layer_idx += 1

        # TODO: check timeout

    multiprocess_mip_model = None
    self.mip_model.update()

    refined_bounds = {}
    for i, layer in enumerate(self.net.relus):
        # only refined with the relu layers that are refined by mip before
        nd = self.net.relus[i].inputs[0].name
        refined_bounds[nd] = [lower_bounds[i], upper_bounds[i]]

    return lower_bounds, upper_bounds


def lp_solve_all_node_split(self, lower_bounds, upper_bounds, assignment, rhs):
    all_node_model = copy_model(self.net.model)
    pre_relu_layer_names = [relu_layer.inputs[0].name for relu_layer in self.net.relus]
    relu_layer_names = [relu_layer.name for relu_layer in self.net.relus]

    assert (lower_bounds is not None) or (assignment is not None)
    if lower_bounds is not None:
        for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
            lbs, ubs = lower_bounds[relu_idx].reshape(-1), upper_bounds[relu_idx].reshape(-1)
            for neuron_idx in range(lbs.shape[0]):
                pre_var = all_node_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
                pre_var.lb = pre_lb = lbs[neuron_idx]
                pre_var.ub = pre_ub = ubs[neuron_idx]
                var = all_node_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
                # var is None if originally stable
                if var is not None:
                    if pre_lb >= 0 and pre_ub >= 0:
                        # ReLU is always passing
                        var.lb = pre_lb
                        var.ub = pre_ub
                        all_node_model.addConstr(pre_var == var)
                    elif pre_lb <= 0 and pre_ub <= 0:
                        var.lb = 0
                        var.ub = 0
                    else:
                        raise ValueError(f'Exists unstable neuron at index [{relu_idx}][{neuron_idx}]: lb={pre_lb} ub={pre_ub}')

    else:
        for relu_idx, (pre_relu_name, relu_name) in enumerate(zip(pre_relu_layer_names, relu_layer_names)):
            for neuron_idx in range(len(self.layers_mapping[relu_idx])):
                pre_var = all_node_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
                var = all_node_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
                # var is None if originally stable
                if var is not None:
                    if assignment[self.layers_mapping[relu_idx][neuron_idx]]['value']:
                        # ReLU is always passing
                        all_node_model.addConstr(pre_var == var)
                    else:
                        var.lb = 0
                        var.ub = 0

    all_node_model.update()
    
    feasible = True
    adv = None
    
    orig_out_vars = self.net.final_node().solver_vars
    assert len(orig_out_vars) == len(rhs), f"out shape not matching! {len(orig_out_vars)} {len(rhs)}"
    for out_idx in range(len(orig_out_vars)):
        objVar = all_node_model.getVarByName(orig_out_vars[out_idx].VarName)
        decision_threshold = rhs[out_idx]
        all_node_model.setObjective(objVar, grb.GRB.MINIMIZE)
        all_node_model.update()
        all_node_model.optimize()

        if all_node_model.status == 2:
            print("Gurobi all node split: feasible!")
            glb = objVar.X
        elif all_node_model.status == 3:
            print("Gurobi all node split: infeasible!")
            glb = float('inf')
        else:
            print(f"Warning: model status {m.all_node_model.status}!")
            glb = float('inf')

        if glb > decision_threshold:
            feasible = False
            break

        input_vars = [all_node_model.getVarByName(var.VarName) for var in self.net.input_vars]
        adv = torch.tensor([var.X for var in input_vars], device=self.device).view(self.input_shape)

    del all_node_model
    # print(lp_status, glb)
    return feasible, adv