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

from heuristic.falsification import randomized_falsification
from dnn_solver.symbolic_network import SymbolicNetwork
from dnn_solver.worker import implication_gurobi_worker
from abstract.eran import deepzono, deeppoly
from lp_solver.lp_solver import LPSolver

from utils.cache import BacksubCacher, AbstractionCacher
from utils.read_nnet import NetworkNNET
from utils.timer import Timers
from utils.misc import MP
import settings

def print_bounds(layers_mapping, hidden_bounds, assignment, implications):
    mapping_bounds = {}
    reversed_layers_mapping = {i: k for k, v in layers_mapping.items() for i in v}

    for idx, (lb, ub) in enumerate(hidden_bounds):
        b = [(l, u) for l, u in zip(lb, ub)]
        mapping_bounds.update(dict(zip(layers_mapping[idx], b)))
    for k, v in mapping_bounds.items():
        if k in assignment:
            continue

        if k in implications and (implications[k]['pos'] or implications[k]['neg']):
            continue

        lidx = reversed_layers_mapping[k]
        break

    for k, v in mapping_bounds.items():
        if k in layers_mapping[lidx]:
            print(k, v[0].numpy(), v[1].numpy())


    return mapping_bounds

class DNNTheoremProver:

    epsilon = 1e-6
    skip = 1e-3

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        self.decider = decider

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)

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

        # if settings.HEURISTIC_DEEPZONO:
        #     self.deepzono = deepzono.DeepZono(net)


        self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)

        self.transformer = SymbolicNetwork(net)

        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=1000)

            self.cs = LPSolver(net, spec, self.deeppoly)

            # (l, u), _ = self.deeppoly(self.lbs_init, self.ubs_init)
            # self.spec.register(l, u)

        self.concrete = self.net.get_concrete((self.lbs_init + self.ubs_init) / 2.0)

        self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)
        # self.abstraction_cacher = AbstractionCacher((self.lbs_init, self.ubs_init), max_caches=100)


        # clean trash
        # os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        # test
        self.decider.target_direction_list = [[self.rf.targets[0], self.rf.directions[0]]]

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

    def _update_new_input_bounds(self, l, u , i):
        if (l == self.lbs_init[i]) and (u < self.ubs_init[i]):
            # print('\tnew lower', i, self.lbs_init[i], '--->', u)
            self.lbs_init[i] = u
            self.gurobi_vars[i].lb = u
        elif (u == self.ubs_init[i]) and (l > self.lbs_init[i]):
            # print('\tnew upper', i, self.ubs_init[i], '--->', l)
            self.ubs_init[i] = l
            self.gurobi_vars[i].ub = l

        self.model.update()

    def _single_range_check(self, lbs, ubs, assignment):
        return
        for i in range(self.net.n_input):
            tmp_lbs = self.lbs_init.clone()
            tmp_ubs = self.ubs_init.clone()
            modified = False

            if (lbs[i] > self.lbs_init[i]):
                tmp_lbs[i] = lbs[i]
                modified = True
            elif (ubs[i] < self.ubs_init[i]):
                modified = True
                tmp_ubs[i] = ubs[i]

            if modified:
                # print('check ', i)
                # print('origin lower', self.lbs_init)
                # print('modify lower', tmp_lbs)

                # print('origin upper', self.ubs_init)
                # print('modify upper', tmp_ubs)

                (l, u), hidden_bounds = self._compute_output_abstraction(tmp_lbs, tmp_ubs, assignment)
                if not self.spec.check_output_reachability(l, u): # conflict
                    self._update_new_input_bounds(tmp_lbs[i], tmp_ubs[i], i)
                    # return



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

        # check overapprox sat
        # print('start')
        # cc = self._shorten_conflict_clause(assignment)
        # print('start:', cc)
        # if len(cc):
        #     return False, cc, None


        # reset constraints

        Timers.tic('Reset solver')

        self.model.remove(self.constraints)
        self._restore_input_bounds()
        self.constraints = []
        Timers.toc('Reset solver')

        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # forward
        Timers.tic('backsub_dict')
        output_mat, backsub_dict = self.transformer(assignment)
        Timers.toc('backsub_dict')

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

        # convert to gurobi LinExpr
        Timers.tic('Get Linear Equation')
        backsub_dict_expr = self.backsub_cacher.get_cache(assignment)
        if backsub_dict_expr is not None:
            backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
        else:
            backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

        self.backsub_cacher.put(assignment, backsub_dict_expr)
        Timers.toc('Get Linear Equation')

        # add constraints
        Timers.tic('Add constraints')
        for node in backsub_dict_expr:
            status = assignment.get(node, None)
            if status is None:
                continue
            if status:
                ci = self.model.addLConstr(backsub_dict_expr[node] >= DNNTheoremProver.epsilon)
            else:
                ci = self.model.addLConstr(backsub_dict_expr[node] <= 0)
            self.constraints.append(ci)

        Timers.toc('Add constraints')

        # check satisfiability
        if not is_full_assignment:
            self._optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('call from partial assignment')
                cc = self._shorten_conflict_clause(assignment, True)
                return False, cc, None

            if settings.DEBUG:
                print('[+] Check partial assignment: `SAT`')


        else: # output
            flag_sat = False
            Timers.tic('Check output property')
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

            Timers.toc('Check output property')
            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            # print('call from full assignment')
            cc = self._shorten_conflict_clause(assignment, True)
            return False, cc, None



        # reachable heuristic:
        if settings.TIGHTEN_BOUND:# and (self.count % settings.HEURISTIC_DEEPPOLY_INTERVAL == 0): 
            # compute new input lower/upper bounds
            Timers.tic('Tighten bounds')

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

            Timers.toc('Tighten bounds')

            Timers.tic('Update bounds')
            stat = self._update_input_bounds(lbs, ubs)
            Timers.toc('Update bounds')
                
            if not stat: # conflict
                # print('call from update bounds')
                cc = self._shorten_conflict_clause(assignment, False)
                return False, cc, None

            lbs = torch.tensor([var.lb for var in self.gurobi_vars], dtype=settings.DTYPE)
            ubs = torch.tensor([var.ub for var in self.gurobi_vars], dtype=settings.DTYPE)

            # Timers.tic('Cache abstraction')
            # score = self.abstraction_cacher.get_score((lbs, ubs))
            # Timers.toc('Cache abstraction')

            # should_run_abstraction = True
            # print('should_run_abstraction:', should_run_abstraction)

            # if should_run_abstraction:
            Timers.tic('Compute output abstraction')
            (lower, upper), hidden_bounds = self._compute_output_abstraction(lbs, ubs, assignment)
            Timers.toc('Compute output abstraction')


            Timers.tic('Heuristic Decision Update')
            if self.decider is not None and settings.DECISION != 'RANDOM':
                self.decider.update(output_bounds=(lower, upper), hidden_bounds=hidden_bounds)
            Timers.toc('Heuristic Decision Update')



            Timers.tic('Check output reachability')
            stat, should_run_again = self.spec.check_output_reachability(lower, upper)
            Timers.toc('Check output reachability')

            # self.abstraction_cacher.put((lbs, ubs), stat)
            # print(stat, score)

            if not stat: # conflict
                
                # Timers.tic('_single_range_check')
                # self._single_range_check(lbs, ubs, assignment)
                # Timers.toc('_single_range_check')

                # print('call from reachability heuristic')
                cc = self._shorten_conflict_clause(assignment, False)
                return False, cc, None

            if settings.HEURISTIC_DEEPPOLY and should_run_again:
                Timers.tic('Deeppoly optimization reachability')
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
                            Timers.toc('Deeppoly optimization reachability')
                            # print('ngon')
                            return True, {}, None
                        self.concrete = self.net.get_concrete(tmp_input)

                        flag_sat = True
                        break

                if not flag_sat:
                    # print('call from optimized reachability heuristic')
                    Timers.toc('Deeppoly optimization reachability')
                    cc = self._shorten_conflict_clause(assignment, False)
                    return False, cc, None
                Timers.toc('Deeppoly optimization reachability')


        # implication heuristic
        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION:
            # tic = time.time()
            # stat, adv = self.rf.eval_constraints(None)
            # if stat == 'violated':
            #     self.solution = adv[0]
            #     return True, {}, is_full_assignment
            Timers.tic('randomized_falsification')

            new_ranges = torch.stack([lbs, ubs], dim=1).to(settings.DTYPE)
            stat, adv = self.rf.eval_constraints(new_ranges)

            Timers.toc('randomized_falsification')

            if stat == 'violated':
                self.solution = adv[0]
                return True, {}, is_full_assignment

            # print(time.time() - tic)

        self._restore_input_bounds()

        implications = {}
        Timers.tic('Implications')

        # if settings.HEURISTIC_DEEPPOLY_IMPLICATION:
        #     _, hidden_bounds = self._compute_output_abstraction(self.lbs_init, self.ubs_init)
        #     signs = {}
        #     for idx, (lb, ub) in enumerate(hidden_bounds):
        #         sign = 2 * torch.ones(len(lb), dtype=int) 
        #         sign[lb >= 0] = 1
        #         sign[ub <= 0] = -1
        #         signs.update(dict(zip(self.layers_mapping[idx], sign.numpy())))


        #     # for node, status in assignment.items():
        #     #     if signs[node] == 2:
        #     #         continue
        #     #     abt_status = signs[node] == 1
        #     #     if abt_status != status:
        #     #         # print('doan ngu')
        #     #         cc = self._shorten_conflict_clause(assignment)
        #     #         # print('doan ngu:', cc)
        #     #         return False, cc, None
                    
        #     for node, value in signs.items():
        #         implications[node] = {'pos': value==1, 'neg': value==-1}


        if settings.HEURISTIC_GUROBI_IMPLICATION and settings.PARALLEL_IMPLICATION:
            for w in self.workers:
                w.join()

            for w in self.workers:
                res = Q.get()
                implications.update(res)

        elif settings.HEURISTIC_GUROBI_IMPLICATION:
            for node in unassigned_nodes:
                if node in implications and (implications[node]['pos'] or implications[node]['neg']):
                    continue

                implications[node] = {'pos': False, 'neg': False}
                # neg
                if self.concrete[node] <= 0:
                    ci = self.model.addLConstr(backsub_dict_expr[node] >= DNNTheoremProver.epsilon)
                    self._optimize()
                    if self.model.status == grb.GRB.INFEASIBLE:
                        implications[node]['neg'] = True
                else:
                # pos
                    ci = self.model.addLConstr(backsub_dict_expr[node] <= 0)
                    self._optimize()
                    if self.model.status == grb.GRB.INFEASIBLE:
                        implications[node]['pos'] = True
                
                self.model.remove(ci)

        Timers.toc('Implications')

        # if self.decider is not None:
            # flag_implied = False
            # for k, v in implications.items():
            #     if v['pos'] or v['neg']:
            #         print('-----------', k, v['pos'], v['neg'])
            #         flag_implied = True
            #         break

            # if not flag_implied:
            #     # print()
            #     # print('-------------------------------------------------------------')
            #     # print_bounds(self.layers_mapping, hidden_bounds, assignment, implications)
            #     # print('-------------------------------------------------------------')
            #     # print()
            # self.decider.update(unassigned_nodes, hidden_bounds)

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

    def _compute_output_abstraction(self, lbs, ubs, assignment=None):

        # if settings.HEURISTIC_DEEPZONO: # eran deepzono
        #     (lower, upper), _ = self.deepzono(lbs, ubs)
        # else:  # eran deeppoly
        # if settings.HEURISTIC_DEEPPOLY_W_ASSIGNMENT:
        #     (lower, upper), hidden_bounds = self.deeppoly(lbs, ubs, assignment=assignment)
        # else:
        #     (lower, upper), hidden_bounds = self.deeppoly(lbs, ubs, assignment=None)
        # return (lower, upper), hidden_bounds

        return self.deeppoly(lbs, ubs, assignment=assignment)


    def _shorten_conflict_clause(self, assignment, run_flag):
        return frozenset()
        Timers.tic('shorten_conflict_clause')
        if run_flag:
            cc = self.cs.shorten_conflict_clause(assignment)
            # print('unsat moi vao day', len(assignment), len(cc))
        else:
            cc = frozenset()
        Timers.toc('shorten_conflict_clause')
        # print('assignment =', assignment)
        # print()
        # exit()
        return cc
