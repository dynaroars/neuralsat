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
from dnn_solver.symbolic_network import SymbolicNetwork
from dnn_solver.worker import implication_gurobi_worker
from abstract.eran import deepzono, deeppoly
from utils.read_nnet import NetworkNNET
from utils.misc import MP
import settings


class DNNTheoremProver:

    epsilon = 1e-5
    skip = 1e-3

    def __init__(self, net, spec):
        self.net = net
        self.layers_mapping = net.layers_mapping
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

        if settings.HEURISTIC_DEEPZONO:
            self.deepzono = deepzono.DeepZono(net)


        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=100)

        # clean trash
        os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
            self.rf = randomized_falsification.RandomizedFalsification(net, spec)

        self.transformer = SymbolicNetwork(net)

    @property
    def n_outputs(self):
        return self.net.n_output

    @property
    def n_inputs(self):
        return self.net.n_input


    def _update_input_bounds(self, lbs, ubs):
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

    def _find_unassigned_nodes(self, assignment):
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

    @torch.no_grad()
    def __call__(self, assignment):

        # debug
        self.count += 1

        if self.solution is not None:
            return True, {}, None

        # reset constraints
        self.model.remove(self.constraints)
        self._restore_input_bounds()
        self.constraints = []

        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False

        # forward
        output_mat, backsub_dict = self.transformer(assignment)

        # parallel implication
        if not is_full_assignment and settings.HEURISTIC_GUROBI_IMPLICATION and settings.PARALLEL_IMPLICATION:
            backsub_dict_np = {k: v.detach().numpy() for k, v in backsub_dict.items()}
            kwargs = (self.n_inputs, self.lbs_init, self.ubs_init)
            wloads = MP.get_workloads(unassigned_nodes, n_cpus=settings.N_THREADS)
            Q = multiprocessing.Queue()
            self.workers = [multiprocessing.Process(target=implication_gurobi_worker, 
                                                    args=(assignment, backsub_dict_np, wl, Q, kwargs),
                                                    name=f'Thread {i}',
                                                    daemon=True) for i, wl in enumerate(wloads)]
            for w in self.workers:
                w.start()

        # print(self.layers_mapping)
        # for k, v in backsub_dict.items():
        #     print(k, v.detach().numpy())
        # print()

        # convert to gurobi LinExpr
        backsub_dict = {k: self._get_equation(v) for k, v in backsub_dict.items()}

        # add constraints
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

        # check satisfiability
        if not is_full_assignment:
            self._optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

        else: # output
            flag_sat = False
            output_constraint = self.spec.get_output_property(
                [self._get_equation(output_mat[i]) for i in range(self.n_outputs)]
            )
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


        # reachable heuristic
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
                
            if not self._update_input_bounds(lbs, ubs): # conflict
                ccs = self.shorten_conflict_clause(assignment)
                return False, ccs, None

            lbs = torch.tensor([var.lb for var in self.gurobi_vars], dtype=settings.DTYPE)
            ubs = torch.tensor([var.ub for var in self.gurobi_vars], dtype=settings.DTYPE)

            if settings.DEBUG:
                print('[+] HEURISTIC input')
                print('\t- lower:', lbs.data)
                print('\t- upper:', ubs.data)

            if settings.HEURISTIC_DEEPZONO: # eran deepzono
                (lower, upper), _ = self.deepzono(lbs, ubs)

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
                        tmp_input = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE).view(self.net.input_shape)
                        if self.check_solution(tmp_input):
                            self.solution = tmp_input
                            # print('ngon')
                            return True, {}, None

                        flag_sat = True
                        break

                if not flag_sat:
                    ccs = self.shorten_conflict_clause(assignment)
                    return False, ccs, None


        # implication heuristic
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

        self._restore_input_bounds()

        implications = {}

        if not settings.HEURISTIC_DEEPZONO and settings.HEURISTIC_DEEPPOLY_IMPLICATION:
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
            for node in unassigned_nodes:
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
            return torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE).view(self.net.input_shape)
        return None


    def check_solution(self, solution):
        if torch.any(solution < self.lbs_init.view(self.net.input_shape)) or torch.any(solution > self.ubs_init.view(self.net.input_shape)):
            return False
        if self.spec.check_solution(self.net(solution)):
            return True
        return False


    def _restore_input_bounds(self):
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