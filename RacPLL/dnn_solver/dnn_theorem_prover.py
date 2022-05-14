from pprint import pprint
import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import contextlib
import torch
import time
import copy
import re
import os

from heuristic.randomized_falsification import randomized_falsification
from abstract.eran import deepz, assigned_deeppoly
from utils.terminatable_thread import *
from utils.misc import MP
import settings


def _solve_worker(assignment, mat_dict, nodes, shared_queue, kwargs):
    if len(nodes) == 0:
        return None

    n_vars, lbs, ubs = kwargs
    with contextlib.redirect_stdout(open(os.devnull, 'w')):

        with grb.Env() as env, grb.Model(env=env) as model:
            model.setParam('OutputFlag', False)

            variables = [
                model.addVar(name=f'x{i}', lb=lbs[i], ub=ubs[i]) for i in range(n_vars)
            ]
            model.update()

            for node in mat_dict:
                status = assignment.get(node, None)
                # print(node, status)
                if status is None:
                    continue
                mat = mat_dict[node]
                eqx = grb.LinExpr(mat[:-1], variables) + mat[-1]
                # print(eqx)
                if status:
                    model.addLConstr(eqx >= 1e-6)
                else:
                    model.addLConstr(eqx <= 0)
                #     ci = self.model.addLConstr(mat_dict[node] <= 0)

            results = []
            for node in nodes:
                res = {}
                mat = mat_dict[node]
                # print(node, mat, variables)
                obj = grb.LinExpr(mat[:-1], variables) + mat[-1]

                model.setObjective(obj, grb.GRB.MINIMIZE)
                model.update()
                model.reset()
                model.optimize()

                if model.status == grb.GRB.OPTIMAL:
                    res['pos'] = True if model.objval > 0 else False

                model.setObjective(obj, grb.GRB.MAXIMIZE)
                model.update()
                model.reset()
                model.optimize()

                if model.status == grb.GRB.OPTIMAL:
                    res['neg'] = True if model.objval <= 0 else False

                results.append((node, res))

            shared_queue.put(results)


    # except ThreadTerminatedError:
    #     # time.sleep(0.01)
    #     print(f'{name} terminated')
    #     return None


class DNNTheoremProver:

    epsilon = 1e-5
    skip = 1e-3

    def __init__(self, dnn, layers_mapping, spec):
        self.dnn = dnn
        self.layers_mapping = layers_mapping
        self.spec = spec


        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)
            self.model.setParam('Threads', 16)

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
            for i in range(self.n_inputs)
        ]

        self.count = 0 # debug

        self.solution = None
        self.constraints = []

        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = assigned_deeppoly.AssignedDeepPoly(dnn, back_sub_steps=100)

        # clean trash
        os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
            self.rf = randomized_falsification.RandomizedFalsification(dnn, spec)


    @property
    def n_outputs(self):
        return self.dnn.output_shape[1]

    @property
    def n_inputs(self):
        return self.dnn.input_shape[1]


    def update_input_bounds(self, lbs, ubs):
        if lbs is None or ubs is None:
            return True
        lbs = np.array(lbs)
        ubs = np.array(ubs)
        flag = lbs > ubs

        if np.any(flag):
            if not np.all(lbs[flag] - ubs[flag] < DNNTheoremProver.epsilon):
                return False

        for i, var in enumerate(self.gurobi_vars):
            if abs(lbs[i] - ubs[i]) < DNNTheoremProver.epsilon: # concretize
                # var.lb = lbs[i]
                # var.ub = lbs[i]
                continue
            if (abs(var.lb - lbs[i]) > DNNTheoremProver.skip):
                var.lb = lbs[i]
            if (abs(var.ub - ubs[i]) > DNNTheoremProver.skip):
                var.ub = ubs[i]

        self.model.update()
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

    def _get_equation(self, coeffs):
        return grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]

    def __call__(self, assignment):

        # debug
        self.count += 1

        if self.solution is not None:
            return True, {}, None

        # reset constraints
        self.model.remove(self.constraints)
        self.restore_input_bounds()
        self.constraints = []

        imply_nodes = self._find_nodes(assignment)
        is_full_assignment = True if imply_nodes is None else False


        inputs = torch.hstack([torch.eye(self.n_inputs), torch.zeros(self.n_inputs, 1)]).to(settings.DTYPE)

        layer_id = 0
        variables = self.layers_mapping.get(layer_id, None)
        flag_break = False
        backsub_dict = {}
        for layer in self.dnn.layers:
            if variables is None: # output layer
                output = layer.weight.mm(inputs)
                output[:, -1] += layer.bias
                output_constraint = self.spec.get_output_property(
                    [self._get_equation(output[i]) for i in range(self.n_outputs)]
                )
            else:
                if isinstance(layer, nn.Linear):
                    output = layer.weight.mm(inputs)
                    output[:, -1] += layer.bias

                elif isinstance(layer, nn.ReLU):
                    inputs = torch.zeros(output.shape, dtype=settings.DTYPE)
                    for i, v in enumerate(variables):
                        status = assignment.get(v, None)
                        if status is None: # unassigned node
                            flag_break = True
                        elif status:
                            inputs[i] = output[i]
                        else:
                            # inputs[i] = zero_torch
                            pass
                        backsub_dict[v] = output[i]

                    layer_id += 1
                    variables = self.layers_mapping.get(layer_id, None)
                else:
                    raise NotImplementedError

                if flag_break and (not is_full_assignment):
                    break


        if not is_full_assignment and settings.HEURISTIC_GUROBI_IMPLICATION and settings.PARALLEL_IMPLICATION:
            backsub_dict_np = {k: v.detach().numpy() for k, v in backsub_dict.items()}
            kwargs = (self.n_inputs, self.lbs_init, self.ubs_init)
            wloads = MP.get_workloads(imply_nodes, n_cpus=settings.N_THREADS)

            Q = multiprocessing.Queue()

            self.workers = [
                multiprocessing.Process(target=_solve_worker, 
                                        args=(assignment, backsub_dict_np, wl, Q, kwargs),
                                        name=f'Thread {i}',
                                        daemon=True) 
                for i, wl in enumerate(wloads)
            ]

            for w in self.workers:
                w.start()


        backsub_dict = {k: self._get_equation(v) for k, v in backsub_dict.items()}

        for node in backsub_dict:
            status = assignment.get(node, None)
            if status is None:
                continue
            if status:
                ci = self.model.addLConstr(backsub_dict[node] >= DNNTheoremProver.epsilon)
            else:
                ci = self.model.addLConstr(backsub_dict[node] <= 0)
            self.constraints.append(ci)

        # debug
        if settings.DEBUG:
            self.model.write(f'gurobi/{self.count}.lp')

        if not is_full_assignment:
            self._optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

        else: # output
            flag_sat = False
            for cnf in output_constraint:
                ci = [self.model.addLConstr(_) for _ in cnf]
                self._optimize()
                self.model.remove(ci)
                if self.model.status == grb.GRB.OPTIMAL:
                    if self.check_solution(self.get_solution()):
                        flag_sat = True
                        break

            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            ccs = self.shorten_conflict_clause(assignment)
            return False, ccs, None


        if settings.DEBUG:
            print('[+] Check assignment: `SAT`')


        if settings.TIGHTEN_BOUND: # compute new input lower/upper bounds

            if settings.DEBUG:
                print('[+] TIGHTEN_BOUND ')

            # upper
            self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MAXIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                ubs = [var.X for var in self.gurobi_vars]
            else:
                ubs = None

            # lower
            self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MINIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                lbs = [var.X for var in self.gurobi_vars]
            else:
                lbs = None
                
            if not self.update_input_bounds(lbs, ubs): # conflict
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

            lbs = torch.tensor([var.lb for var in self.gurobi_vars], dtype=settings.DTYPE)
            ubs = torch.tensor([var.ub for var in self.gurobi_vars], dtype=settings.DTYPE)

            if settings.DEBUG:
                print('[+] HEURISTIC input')
                print('\t- lower:', lbs.data)
                print('\t- upper:', ubs.data)

            if settings.HEURISTIC_DEEPZONO: # eran deepzono
                (lower, upper), _ = deepz.forward(self.dnn, lbs, ubs)

                if settings.DEBUG:
                    print('[+] HEURISTIC DEEPZONO output')
                    print('\t- lower:', lower)
                    print('\t- upper:', upper)

            else:
                if settings.HEURISTIC_DEEPPOLY_W_ASSIGNMENT:
                    (lower, upper), _ = self.deeppoly(lbs, ubs, assignment=assignment)

                    if settings.DEBUG:
                        print('[+] HEURISTIC DEEPPOLY output (w assignment)')
                        print('\t- lower:', lower)
                        print('\t- upper:', upper)
                else:
                    (lower, upper), _ = self.deeppoly(lbs, ubs, assignment=None)

                    if settings.DEBUG:
                        print('[+] HEURISTIC DEEPPOLY output (w/o assignment)')
                        print('\t- lower:', lower)
                        print('\t- upper:', upper)


            if not self.spec.check_output_reachability(lower, upper): # conflict
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

            if settings.HEURISTIC_DEEPPOLY:
                Ml, Mu, bl, bu  = self.deeppoly.get_params()
                lbs_expr = [grb.LinExpr(wl.numpy(), self.gurobi_vars) + cl for (wl, cl) in zip(Ml, bl)]
                ubs_expr = [grb.LinExpr(wu.numpy(), self.gurobi_vars) + cu for (wu, cu) in zip(Mu, bu)]
                dnf_contrs = self.spec.get_output_reachability_constraints(lbs_expr, ubs_expr)
                flag_sat = False
                for cnf, adv_obj in dnf_contrs:
                    ci = [self.model.addLConstr(_) for _ in cnf]
                    self.model.setObjective(adv_obj, grb.GRB.MINIMIZE)
                    self._optimize()
                    self.model.remove(ci)
                    if self.model.status == grb.GRB.OPTIMAL:
                        tmp_input = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE)
                        # print(self.model.objval, tmp_input)
                        if self.check_solution(tmp_input):
                            self.solution = tmp_input
                            print('ngon')
                            return True, {}, None

                        flag_sat = True
                        break

                if not flag_sat:
                    ccs = self.shorten_conflict_clause(assignment)
                    return False, ccs, None

        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
            stat, adv = self.rf.eval_constraints(None)
            if stat == 'violated':
                self.solution = adv[0]
                return True, {}, is_full_assignment

            new_ranges = torch.stack([lbs, ubs], dim=1).to(settings.DTYPE)
            stat, adv = self.rf.eval_constraints(new_ranges)
            if stat == 'violated':
                self.solution = adv[0]
                return True, {}, is_full_assignment

        self.restore_input_bounds()

        # imply next hidden nodes
        implications = {}

        if settings.HEURISTIC_DEEPPOLY_IMPLICATION:
            _, hidden_bounds = self.deeppoly(self.lbs_init, self.ubs_init, assignment=assignment)

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
                implications[node] = {'pos': value==1, 'neg': value==-1}


        if settings.HEURISTIC_GUROBI_IMPLICATION and settings.PARALLEL_IMPLICATION:
            for w in self.workers:
                w.join()

            for w in self.workers:
                res = Q.get()
                implications.update(res)
        else:
            for node in imply_nodes:
                if node in implications and (implications[node]['pos'] or implications[node]['neg']):
                    continue

                implications[node] = {'pos': False, 'neg': False}
                # neg
                ci = self.model.addLConstr(backsub_dict[node] >= DNNTheoremProver.epsilon)
                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['neg'] = True
                    self.model.remove(ci)
                    continue
                self.model.remove(ci)

                # pos
                ci = self.model.addLConstr(backsub_dict[node] <= 0)
                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['pos'] = True
                    self.model.remove(ci)
                else:
                    self.model.remove(ci)

        return True, implications, is_full_assignment

    def _optimize(self):
        self.model.update()
        self.model.reset()
        self.model.optimize()


    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self._optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE)
        return None


    def check_solution(self, solution):
        if any(solution < self.lbs_init) or any(solution > self.ubs_init):
            return False
        if self.spec.check_solution(self.dnn(solution)):
            return True
        return False


    def restore_input_bounds(self):
        for i, var in enumerate(self.gurobi_vars):
            var.lb = self.lbs_init[i]
            var.ub = self.ubs_init[i]
        self.model.update()



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