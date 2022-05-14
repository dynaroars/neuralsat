from joblib import Parallel, delayed
from pprint import pprint
import gurobipy as grb
import torch.nn as nn
import numpy as np
import torch
import time
import copy
import re
import os

from heuristic.randomized_falsification import randomized_falsification
from abstract.eran import deepz, assigned_deeppoly
from utils.misc import MP
import settings


class DNNTheoremProver:

    epsilon = 1e-6
    skip = 1e-3

    def __init__(self, dnn, layers_mapping, spec):
        self.dnn = dnn
        self.layers_mapping = layers_mapping
        self.spec = spec

        # tic = time.time()
        # self.model = grb.Model()
        # self.model.setParam('OutputFlag', False)
        # self.model.setParam('Threads', 16)

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.Tensor(bounds_init['lbs'])
        self.ubs_init = torch.Tensor(bounds_init['ubs'])

        # self.gurobi_vars = [
        #     self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
        #     for i in range(self.n_inputs)
        # ]

        # # self.model.setObjective(0, grb.GRB.MAXIMIZE)
        # self.model.update()
        # print('gurobi init:', time.time() - tic)
        
        self.count = 0 # debug

        self.solution = None
        self.constraints = []

        if settings.HEURISTIC_DEEPPOLY:
            # self.deeppoly = eran.ERAN(self.dnn.path[:-4] + 'onnx', 'deeppoly')
            self.deeppoly = assigned_deeppoly.AssignedDeepPoly(dnn, back_sub_steps=100)

        # clean trash
        os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
            self.rf = randomized_falsification.RandomizedFalsification(dnn, spec)
            self.count_rf = 0
            self.tic_rf = time.time()

        # self.model_scc = grb.Model()
        # self.model_scc.setParam('OutputFlag', False)
        # self.model_scc.setParam('Threads', 16)
        # self.gurobi_vars_scc = [
        #     self.model_scc.addVar(name=f'x{i}_scc', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
        #     for i in range(self.n_inputs)
        # ]
        # self.model_scc.update()

    @property
    def n_outputs(self):
        return self.dnn.output_shape[1]

    @property
    def n_inputs(self):
        return self.dnn.input_shape[1]


    def update_input_bounds(self, variables, lbs, ubs):
        if lbs is None or ubs is None:
            return True
        lbs = np.array(lbs)
        ubs = np.array(ubs)
        flag = lbs > ubs

        if np.any(flag):
            if not np.all(lbs[flag] - ubs[flag] < DNNTheoremProver.epsilon):
                # self.model.write(f'gurobi/debug_{self.count}.lp')
                return False

        for i, var in enumerate(variables):
            if abs(lbs[i] - ubs[i]) < DNNTheoremProver.epsilon: # concretize
                # var.lb = lbs[i]
                # var.ub = lbs[i]
                continue
            if (abs(var.lb - lbs[i]) > DNNTheoremProver.skip):
                var.lb = lbs[i]
            if (abs(var.ub - ubs[i]) > DNNTheoremProver.skip):
                var.ub = ubs[i]

        return True

    def _find_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes

    def _get_equation(self, coeffs, variables):
        return grb.LinExpr(coeffs[:-1], variables) + coeffs[-1]

    def __call__(self, assignment):
        tic = time.time()
        with grb.Env() as env, grb.Model(env=env) as model:

            # debug
            self.count += 1

            if self.solution is not None:
                return True, {}, None

            model.setParam('OutputFlag', False)
            model.setParam('LogToConsole', False)
            # self.model.setParam('Threads', 16)


            gurobi_vars = [
                model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) for i in range(self.n_inputs)
            ]

            # self.model.setObjective(0, grb.GRB.MAXIMIZE)
            # self.model.update()
            print('gurobi init:', time.time() - tic)

            # reset constraints
            # self.model.remove(self.constraints)
            # self.restore_input_bounds()
            # self.constraints = []

            imply_nodes = self._find_nodes(assignment)
            is_full_assignment = True if imply_nodes is None else False

            substitute_dict_torch = {}

            inputs = torch.hstack([torch.eye(self.n_inputs), torch.zeros(self.n_inputs, 1)])
            layer_id = 0
            variables = self.layers_mapping.get(layer_id, None)
            flag_break = False

            tic = time.time()

            substitute_mat = []
            substitute_var = []
            substitute_b = []

            for layer in self.dnn.layers:
                if variables is None: # output layer
                    output = layer.weight.mm(inputs)
                    output[:, -1] += layer.bias
                    output_constraint = self.spec.get_output_property(
                        [self._get_equation(output[i], gurobi_vars) for i in range(self.n_outputs)]
                    )
                else:
                    if isinstance(layer, nn.Linear):
                        output = layer.weight.mm(inputs)
                        output[:, -1] += layer.bias

                    elif isinstance(layer, nn.ReLU):
                        inputs = torch.zeros(output.shape)
                        for i, v in enumerate(variables):
                            status = assignment.get(v, None)
                            if status is None: # unassigned node
                                flag_break = True
                                # continue
                            
                            if status:
                                inputs[i] = output[i]
                            else:
                                # inputs[i] = zero_torch
                                pass
                            substitute_dict_torch[v] = self._get_equation(output[i], gurobi_vars)


                        layer_id += 1
                        variables = self.layers_mapping.get(layer_id, None)
                    else:
                        raise NotImplementedError

                    if flag_break and (not is_full_assignment):
                        break

            print('\t- forward:', time.time() - tic)
            # print(substitute_dict_torch)
            # constraints_mapping = {}

            # tic = time.time()
            # print(substitute_var)

            # substitute_dict_matmul = dict(zip(substitute_var, np.array(substitute_mat) @ self.gurobi_vars + substitute_b))
            # print('\t- substitute_dict_matmul:', time.time() - tic)

            # for i in substitute_var:
            #     print('torch:', substitute_dict_torch[i])
            #     print('numpy:', substitute_dict_matmul[i])
            #     print()
            # def _worker1(idx):
            #     return idx, self._get_equation(substitute_dict_torch[idx])

            # results = Parallel(n_jobs=16, backend='multiprocessing')(delayed(_worker1)(_) for _ in substitute_dict_torch)


            # tic = time.time()
            # substitute_dict_torch = {k: self._get_equation(v) for k,v in substitute_dict_torch.items()}
            # print('\t- substitute_dict_torch:', time.time() - tic)

            # exit()



            tic = time.time()

            for node in substitute_dict_torch:
                status = assignment.get(node, None)
                if status is None:
                    continue

                # eqx = self._get_equation(substitute_dict_torch[node])
                eqx = substitute_dict_torch[node]
                if status:
                    model.addLConstr(eqx >= DNNTheoremProver.epsilon)
                else:
                    model.addLConstr(eqx <= 0)
                # self.constraints.append(ci)
                # constraints_mapping[ci] = node
            print('\t- addLConstrs:', time.time() - tic)

            # debug
            if settings.DEBUG:
                model.write(f'gurobi/{self.count}.lp')


            self._optimize(model)
            if model.status == grb.GRB.INFEASIBLE:
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

            if settings.DEBUG:
                print('[+] Check assignment: `SAT`')

            # output
            if is_full_assignment:
                flag_sat = False
                for cnf in output_constraint:
                    ci = [model.addLConstr(_) for _ in cnf]
                    self._optimize(model)
                    model.remove(ci)
                    if model.status == grb.GRB.OPTIMAL:
                        if self.spec.check_solution(self.dnn(self.get_solution(model, gurobi_vars))):
                            flag_sat = True
                            break

                if flag_sat:
                    self.solution = self.get_solution(model, gurobi_vars)
                    return True, {}, is_full_assignment
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None


            if settings.TIGHTEN_BOUND: # compute new input lower/upper bounds

                if settings.DEBUG:
                    print('[+] TIGHTEN_BOUND ')

                # upper
                model.setObjective(grb.quicksum(gurobi_vars), grb.GRB.MAXIMIZE)
                self._optimize(model)
                if model.status == grb.GRB.OPTIMAL:
                    ubs = [var.X for var in gurobi_vars]
                else:
                    ubs = None

                # lower
                model.setObjective(grb.quicksum(gurobi_vars), grb.GRB.MINIMIZE)
                self._optimize(model)
                if model.status == grb.GRB.OPTIMAL:
                    lbs = [var.X for var in gurobi_vars]
                else:
                    lbs = None
                    
                print('before:', [var.lb for var in gurobi_vars])
                print('before:', [var.ub for var in gurobi_vars])

                if not self.update_input_bounds(gurobi_vars, lbs, ubs): # conflict
                    ccs = self.shorten_conflict_clause(assignment)
                    return False, ccs, None

                print('after:', [var.lb for var in gurobi_vars])
                print('after:', [var.ub for var in gurobi_vars])
                print()

                # reset objective
                # self.model.setObjective(0, grb.GRB.MAXIMIZE)

                lbs = torch.Tensor([var.lb for var in gurobi_vars])
                ubs = torch.Tensor([var.ub for var in gurobi_vars])

                if settings.DEBUG:
                    print('[+] HEURISTIC input')
                    print('\t- lower:', lbs.data)
                    print('\t- upper:', ubs.data)

                if settings.HEURISTIC_DEEPZONO: # eran deepzono
                    (lower, upper), hidden_bounds = deepz.forward(self.dnn, lbs, ubs)

                    if settings.DEBUG:
                        print('[+] HEURISTIC DEEPZONO output')
                        print('\t- lower:', lower)
                        print('\t- upper:', upper)

                else:
                    if settings.HEURISTIC_DEEPPOLY_W_ASSIGNMENT:
                        (lower, upper), hidden_bounds = self.deeppoly(lbs, ubs, assignment=assignment)

                        if settings.DEBUG:
                            print('[+] HEURISTIC DEEPPOLY output (w assignment)')
                            print('\t- lower:', lower)
                            print('\t- upper:', upper)
                    else:
                        (lower, upper), hidden_bounds = self.deeppoly(lbs, ubs, assignment=None)

                        if settings.DEBUG:
                            print('[+] HEURISTIC DEEPPOLY output (w/o assignment)')
                            print('\t- lower:', lower)
                            print('\t- upper:', upper)


                if not self.spec.check_output_reachability(lower, upper): # conflict
                    ccs = self.shorten_conflict_clause(assignment)
                    return False, ccs, None

                tic = time.time()
                if settings.HEURISTIC_DEEPPOLY:
                    Ml, Mu, bl, bu  = self.deeppoly.get_params()
                    lbs_expr = [grb.LinExpr(wl.numpy(), gurobi_vars) + cl for (wl, cl) in zip(Ml, bl)]
                    ubs_expr = [grb.LinExpr(wu.numpy(), gurobi_vars) + cu for (wu, cu) in zip(Mu, bu)]
                    dnf_contrs = self.spec.get_output_reachability_constraints(lbs_expr, ubs_expr)
                    flag_sat = False
                    for cnf, adv_obj in dnf_contrs:
                        ci = [model.addLConstr(_) for _ in cnf]
                        model.setObjective(adv_obj, grb.GRB.MINIMIZE)
                        self._optimize(model)
                        # self.model.setObjective(0, grb.GRB.MAXIMIZE)
                        model.remove(ci)
                        if model.status == grb.GRB.OPTIMAL:
                            tmp_input = torch.Tensor([var.X for var in gurobi_vars])
                            # print(self.model.objval, tmp_input)
                            if self.spec.check_solution(self.dnn(tmp_input)):
                                self.solution = tmp_input
                                print('ngon')
                                return True, {}, None

                            flag_sat = True
                            break

                    if not flag_sat:
                        ccs = self.shorten_conflict_clause(assignment)
                        return False, ccs, None

                print('\t- deeppoly:', time.time() - tic)

                # self.model.setObjective(0, grb.GRB.MAXIMIZE)

            if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
                tic = time.time()
                stat, adv = self.rf.eval_constraints(None)
                print('\t- falsify:', time.time() - tic)
                # stat, adv = self.rf.eval(new_ranges, timeout=2)
                if stat == 'violated':
                    # self.count_rf += 1
                    # print(self.count_rf)
                    # print(self.count_rf, adv, time.time()-self.tic_rf)
                    self.solution = adv[0]
                    # new_assignment = self.dnn.get_assignment(adv[0], self.layers_mapping)
                    # print(new_assignment)
                    return True, {}, is_full_assignment

                new_ranges = torch.stack([lbs, ubs], dim=1)
                # print(new_ranges)
                stat, adv = self.rf.eval_constraints(new_ranges)
                if stat == 'violated':
                    # self.count_rf += 1
                    # print(self.count_rf)
                    # print(self.count_rf, adv, time.time()-self.tic_rf)
                    self.solution = adv[0]
                    # new_assignment = self.dnn.get_assignment(adv[0], self.layers_mapping)
                    # print(new_assignment)
                    return True, {}, is_full_assignment

            self.restore_input_bounds(gurobi_vars)
            # imply next hidden nodes
            tic = time.time()


            implications = {}

            if settings.TIGHTEN_BOUND:
                signs = {}
                for idx, (lb, ub) in enumerate(hidden_bounds):
                    sign = 2 * torch.ones(len(lb), dtype=int) 
                    sign[lb >= 0] = 1
                    sign[ub <= 0] = -1
                    signs.update(dict(zip(self.layers_mapping[idx], sign.numpy())))

                for node, status in assignment.items():
                    if signs[node] == 2:
                        continue
                    abt_status = signs[node] == 1
                    if abt_status != status:
                        ccs = self.shorten_conflict_clause(assignment)
                        return False, ccs, None

                for node, value in signs.items():
                    if node in assignment:
                        continue
                    if node != 2:
                        implications[node] = {'pos': value==1, 'neg': value==-1}


            print('\t- implications:', time.time() - tic)


            return True, implications, is_full_assignment

    def _optimize(self, model):
        model.update()
        model.reset()
        model.optimize()

    def _optimize_scc(self):
        self.model_scc.update()
        self.model_scc.reset()
        self.model_scc.optimize()


    def get_solution(self, model, variables):
        if model.status == grb.GRB.LOADED:
            self._optimize(model)
        if model.status == grb.GRB.OPTIMAL:
            return torch.Tensor([var.X for var in variables])
        return None


    def restore_input_bounds(self, variables):
        for i, var in enumerate(variables):
            var.lb = self.lbs_init[i]
            var.ub = self.ubs_init[i]



    def shorten_conflict_clause(self, assignment):
        return None
        # print('cac', self.layers_mapping)
        _, _, params = self.deeppoly(self.lbs_init, self.ubs_init, assignment=None, return_params=True)

        conflict_clauses = []
        # print(assignment)
        exprs = {}
        for idx, p in enumerate(params[:-1]):
            Ml, Mu, bl, bu = p
            lbs_expr = [sum(wl.numpy() * self.gurobi_vars_scc) + cl for (wl, cl) in zip(Ml, bl)]
            ubs_expr = [sum(wu.numpy() * self.gurobi_vars_scc) + cu for (wu, cu) in zip(Mu, bu)]
            lu_expr = [(l, u) for l, u in zip(lbs_expr, ubs_expr)]
            exprs.update(dict(zip(self.layers_mapping[idx], lu_expr)))

        constraints_scc = []
        constraints_scc_mapping = {}
        for node, status in assignment.items():
            if status is None:
                continue
            if status:
                ci = self.model_scc.addLConstr(exprs[node][0] >= 1e-6)
            else:
                ci = self.model_scc.addLConstr(exprs[node][1] <= 0)
            constraints_scc.append(ci)
            constraints_scc_mapping[ci] = (node, status)


        Ml, Mu, bl, bu  = self.deeppoly.get_params()
        lbs_expr = [sum(wl.numpy() * self.gurobi_vars_scc) + cl for (wl, cl) in zip(Ml, bl)]
        ubs_expr = [sum(wu.numpy() * self.gurobi_vars_scc) + cu for (wu, cu) in zip(Mu, bu)]
        dnf_contrs = self.spec.get_output_reachability_constraints(lbs_expr, ubs_expr)

        self.model_scc.setObjective(0, grb.GRB.MAXIMIZE)
        for cnf, _ in dnf_contrs:
            conflict_clause = set()
            ci = [self.model_scc.addLConstr(_) for _ in cnf]
            self._optimize_scc()
            print('cac', self.model_scc.status == grb.GRB.OPTIMAL)
            if self.model_scc.status == grb.GRB.INFEASIBLE:
                # print('cac')
                # print(len(self.model_scc.getConstrs()), len(assignment))
                self.model_scc.computeIIS()
                # print(len([c for c in self.model_scc.getConstrs() if c.IISConstr]))
                print('input:', [(constraints_scc_mapping[c], exprs[constraints_scc_mapping[c][0]][int(constraints_scc_mapping[c][1])]) for c in constraints_scc if c.IISConstr])
                for variable, status in [constraints_scc_mapping[c] for c in constraints_scc if c.IISConstr]:
                    conflict_clause.add(-variable if status else variable)
                if len(conflict_clause) > 0 and conflict_clause not in conflict_clauses:
                    conflict_clauses.append(conflict_clause)
            self.model_scc.remove(ci)

        self.model_scc.remove(constraints_scc)

        # print('cac', conflict_clauses)
        return None
        return conflict_clauses
