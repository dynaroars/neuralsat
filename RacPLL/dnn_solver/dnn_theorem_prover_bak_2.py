from pprint import pprint
import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import contextlib
import random
import torch
import time
import copy
import re
import os

from heuristic.falsification import randomized_falsification
from heuristic.falsification import gradient_falsification
from dnn_solver.symbolic_network import SymbolicNetwork
from abstract.eran import deepzono, deeppoly
from lp_solver.lp_solver import LPSolver
from dnn_solver.worker import *

from utils.cache import BacksubCacher, AbstractionCacher
from utils.read_nnet import NetworkNNET
from utils.timer import Timers
from utils.misc import MP
import settings


class DNNTheoremProver:

    epsilon = 1e-6
    skip = 1e-3

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        self.hidden_nodes = sum([len(v) for k, v in self.layers_mapping.items()])

        self.decider = decider

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
            for i in range(self.net.n_input)
        ]

        self.count = 0 # debug

        self.solution = None

        # if settings.HEURISTIC_DEEPZONO:
        #     self.deepzono = deepzono.DeepZono(net)

        self.transformer = SymbolicNetwork(net)

        if settings.HEURISTIC_DEEPPOLY:
            self.flag_use_backsub = True
            for layer in net.layers:
                if isinstance(layer, nn.Conv2d):
                    self.flag_use_backsub = False
                    break
            if self.flag_use_backsub:
                self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=1000)
            else:
                self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=0)


        self.concrete = self.net.get_concrete((self.lbs_init + self.ubs_init) / 2.0)
        self.reversed_layers_mapping = {n: k for k, v in self.layers_mapping.items() for n in v}

        # clean trash
        # os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        # test
        # self.decider.target_direction_list = [[self.rf.targets[0], self.rf.directions[0]]]

        self.last_assignment = {}        

        # pgd attack 
        self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)
        self.mvars = grb.MVar(self.gurobi_vars)

        if net.n_input <= 10:
            self.implication_interval = 1
            self.flag_use_mvar = False
            Timers.tic('Randomized attack')
            self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)
            stat, adv = self.rf.eval(timeout=settings.FALSIFICATION_TIMEOUT)
            if settings.DEBUG:
                print('Randomized attack:', stat)
            if stat=='violated':
                self.solution = adv[0]
            Timers.toc('Randomized attack')


        else:
            self.implication_interval = 10
            self.flag_use_mvar = True

            Timers.tic('PGD attack')
            self.gf = gradient_falsification.GradientFalsification(net, spec)
            stat, adv = self.gf.evaluate()
            if settings.DEBUG:
                print('PGD attack:', stat)
            if stat:
                assert spec.check_solution(net(adv))
                assert (adv >= self.gf.lower).all()
                assert (adv <= self.gf.upper).all()
                self.solution = adv
            Timers.toc('PGD attack')

        self.update_input_bounds_last_iter = False

        ###########################################################
        if True:
            print('- Use MVar:', self.flag_use_mvar)
            print('- Implication interval:', self.implication_interval)
            print()
        ###########################################################

        self.Q1 = multiprocessing.Queue()
        self.Q2 = multiprocessing.Queue()

        # (lower, upper), hidden_bounds = self._compute_output_abstraction(self.lbs_init, self.ubs_init)
        # if self.decider is not None and settings.DECISION != 'RANDOM':
        #     self.decider.update(output_bounds=(lower, upper), hidden_bounds=hidden_bounds)



    def _update_input_bounds(self, lbs, ubs):
        for i, var in enumerate(self.gurobi_vars):
            # if abs(lbs[i] - ubs[i]) < DNNTheoremProver.epsilon: # concretize
            var.lb = lbs[i]
            var.ub = ubs[i]

            # if (abs(var.lb - lbs[i]) > DNNTheoremProver.skip):
            #     var.lb = lbs[i]
            # if (abs(var.ub - ubs[i]) > DNNTheoremProver.skip):
            #     var.ub = ubs[i]
        self.model.update()
        self.update_input_bounds_last_iter = True
        return True


    def _restore_input_bounds(self):
        if self.update_input_bounds_last_iter:
            for i, var in enumerate(self.gurobi_vars):
                var.lb = self.lbs_init[i]
                var.ub = self.ubs_init[i]
            self.model.update()
            self.update_input_bounds_last_iter = False



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
        # tic = time.time()
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        # print(len(coeffs), time.time() - tic)
        return expr

    @torch.no_grad()
    def __call__(self, assignment, assignment_full=None):

        # debug
        self.count += 1
        cc = frozenset()

        if self.solution is not None:
            return True, {}, None

        # reset constraints
        # Timers.tic('Reset solver')
        # self._restore_input_bounds()
        # Timers.toc('Reset solver')

        # Timers.tic('Nhu cac')
        # (lower, upper), _ = self.deeppoly(self.lbs_init, self.ubs_init, assignment)
        # stat, _ = self.spec.check_output_reachability(lower, upper)
        # Timers.toc('Nhu cac')
        # if not stat: # conflict
        #     print('hehe')
        #     return False, cc, None


        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # forward
        Timers.tic('backsub_dict')
        output_mat, backsub_dict = self.transformer(assignment)
        Timers.toc('backsub_dict')

        Timers.tic('Find caching assignment')
        cache_nodes = self.get_cache_assignment(assignment)
        remove_nodes = [n for n in self.last_assignment if n not in cache_nodes]
        new_nodes = [n for n in assignment if n not in cache_nodes and n in backsub_dict]
        Timers.toc('Find caching assignment')

        # print('backsub_dict   :', list(backsub_dict.keys()))
        # print('cache_nodes    :', cache_nodes)
        # print('remove_nodes   :', remove_nodes)
        # print('new_nodes      :', new_nodes)
        # print('assignment     :', list(assignment.keys()))
        # print('last_assignment:', list(self.last_assignment.keys()))


        # caching assignment
        self.last_assignment = assignment.copy()
        if len(new_nodes) == 0 and len(remove_nodes) == 0 and len(assignment) > 0:
            return True, {}, is_full_assignment

        # flag_parallel_implication = True
        # print('flag_parallel_implication:', flag_parallel_implication)

        if not self.flag_use_mvar:

            if len(remove_nodes):
                remove_constraints = []
                for node in remove_nodes:
                    cr = self.model.getConstrByName(f'cstr[{node}]')
                    if cr is not None:
                        remove_constraints.append(cr)
                self.model.remove(remove_constraints)


        if not self.flag_use_mvar:

            Timers.tic('get cache backsub_dict')
            backsub_dict_expr = self.backsub_cacher.get_cache(assignment)
            Timers.toc('get cache backsub_dict')
            # convert to gurobi LinExpr
            # Timers.tic('Get Linear Equation')
            # # print(len(new_nodes), len(assignment), len(self.last_assignment))
            # if backsub_dict_expr is not None:
            #     for node in new_nodes:
            #         if node not in backsub_dict_expr:
            #             backsub_dict_expr[node] = self._get_equation(backsub_dict[node])
            # else:
            #     backsub_dict_expr = {k: self._get_equation(backsub_dict[k]) for k in new_nodes}
            # self.backsub_cacher.put({k: assignment[k] for k in new_nodes}, backsub_dict_expr)

            # Timers.toc('Get Linear Equation')


            Timers.tic('Get Linear Equation')
            if backsub_dict_expr is not None:
                backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
            else:
                backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

            self.backsub_cacher.put(assignment, backsub_dict_expr)
            Timers.toc('Get Linear Equation')



            # add constraints
            Timers.tic('Add constraints')
            if len(new_nodes):
                for node in new_nodes:
                    status = assignment.get(node, None)
                    # assert status is not None
                    if status:
                        ci = self.model.addLConstr(backsub_dict_expr[node] >= 1e-6, name=f'cstr[{node}]')
                    else:
                        ci = self.model.addLConstr(backsub_dict_expr[node] <= 0, name=f'cstr[{node}]')
            Timers.toc('Add constraints')

        else:

            Timers.tic('Add constraints')
            if len(remove_nodes) == 0 and len(new_nodes) <= 2:
                # print(len(new_nodes), len(assignment), len(backsub_dict))
                # exit()
                for node in new_nodes:
                    status = assignment.get(node, None)
                    # assert status is not None
                    eqx = self._get_equation(backsub_dict[node])
                    if status:
                        self.model.addLConstr(eqx >= 1e-6)
                    else:
                        self.model.addLConstr(eqx <= 0)

            elif len(assignment) > 0:
                lhs = np.zeros([len(backsub_dict), len(self.gurobi_vars)])
                rhs = np.zeros(len(backsub_dict))
                # mask = np.zeros(len(mat_dict), dtype=np.int32)
                for i, node in enumerate(backsub_dict):
                    status = assignment.get(node, None)
                    if status is None:
                        continue
                    # mask[i] = 1
                    if status:
                        lhs[i] = -1 * backsub_dict[node][:-1]
                        rhs[i] = backsub_dict[node][-1] - 1e-6
                    else:
                        lhs[i] = backsub_dict[node][:-1]
                        rhs[i] = -1 * backsub_dict[node][-1]

                self.model.remove(self.model.getConstrs())
                self.model.addConstr(lhs @ self.mvars <= rhs) 
            Timers.toc('Add constraints')

        self.model.update()

        

        # upper objective
        # self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MAXIMIZE)

        # check satisfiability
        Timers.tic('Check output property')
        if not is_full_assignment:
            self._optimize()
            Timers.toc('Check output property')
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('call from partial assignment')
                # self._restore_input_bounds()
                return False, cc, None

            if settings.DEBUG:
                print('[+] Check partial assignment: `SAT`')


        else: # output
            flag_sat = False
            output_constraint = self.spec.get_output_property(
                [self._get_equation(output_mat[i]) for i in range(self.net.n_output)]
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
            # self._restore_input_bounds()
            return False, cc, None


        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################

        # for node, data in assignment_full.items():
        #     if data['is_implied']:
        #         # exit()
        #         if data['value']:
        #             self.model.setObjective(self._get_equation(backsub_dict[node]), grb.GRB.MINIMIZE)
        #             self.model.optimize()
        #             if self.model.objval < 0:
        #                 print(self.model.objval)
        #                 return False, cc, None
        #         else:
        #             self.model.setObjective(self._get_equation(backsub_dict[node]), grb.GRB.MAXIMIZE)
        #             self.model.optimize()
        #             if self.model.objval >= 0:
        #                 print(self.model.objval)
        #                 return False, cc, None



        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
    
        bounds = {}
        lidx = self.reversed_layers_mapping[list(unassigned_nodes)[0]]
        layer_nodes = list(self.layers_mapping[lidx])
        flag_run_parallel = False

        # print('run tighten bounds', lidx, len(layer_nodes))

        flag_run_tighten = len(assignment) < int(self.hidden_nodes * 0.97)
        implications = {}

        if flag_run_tighten:
            Timers.tic('Tighten bounds')

            if flag_run_parallel:
                backsub_dict_np = {k: v.detach().cpu().numpy() for k, v in backsub_dict.items()}
                wloads = MP.get_workloads(layer_nodes, n_cpus=os.cpu_count())
                workers = [multiprocessing.Process(target=implication_worker2, 
                                                   args=(self.model.copy(), backsub_dict_np, wl, self.Q2, f'Thread {i}'),
                                                   name=f'Thread {i}',
                                                   daemon=True).start() for i, wl in enumerate(wloads)]

                for w in workers:
                    res = self.Q2.get()
                    bounds.update(res)

            else:

                Timers.tic('Optimize layer bounds')
                for node in layer_nodes:

                    if not self.flag_use_mvar:
                        obj = backsub_dict_expr[node]
                    else:
                        obj = self._get_equation(backsub_dict[node])

                    # status = assignment.get(node, None)
                    # if status is None:
                    # lower bound
                    self.model.setObjective(obj, grb.GRB.MINIMIZE)
                    self.model.optimize()
                    lb = self.model.objval

                    # upper bound
                    self.model.setObjective(obj, grb.GRB.MAXIMIZE)
                    self.model.optimize()
                    ub = self.model.objval

                    # elif status: 
                    #     lb = 0 

                    #     # upper bound
                    #     self.model.setObjective(obj, grb.GRB.MAXIMIZE)
                    #     self.model.optimize()
                    #     ub = self.model.objval
                    # else:
                    #     # lower bound
                    #     self.model.setObjective(obj, grb.GRB.MINIMIZE)
                    #     self.model.optimize()
                    #     lb = self.model.objval
                    #     ub = 0 



                    bounds[node] = {'lb': lb, 'ub': ub}
                Timers.toc('Optimize layer bounds')

            Timers.toc('Tighten bounds')

            Timers.tic('Check output reachability')
            if len(assignment) != 0 or True:
                lbs = torch.tensor([bounds[node]['lb'] for node in layer_nodes], dtype=settings.DTYPE, device=self.net.device)
                ubs = torch.tensor([bounds[node]['ub'] for node in layer_nodes], dtype=settings.DTYPE, device=self.net.device)
                (lower, upper), hidden_bounds = self.deeppoly.forward_layer(lbs, ubs, lidx)
            else:
                lbs = self.lbs_init
                ubs = self.ubs_init
                lidx = -1
                (lower, upper), hidden_bounds = self.deeppoly(lbs, ubs, assignment)

            stat, _ = self.spec.check_output_reachability(lower, upper)
            Timers.toc('Check output reachability')


            # # ##################################################
            # # ##################################################
            # # ##################################################

            # print('assignment:', [assignment.keys()])
            # print('unassigned_nodes:', list(unassigned_nodes))

            # print('lbs optimized:', lbs)
            # print('ubs optimized:', ubs)


            # W_cac = torch.zeros(len(layer_nodes), self.net.n_input, dtype=settings.DTYPE, device=self.net.device)
            # b_cac = torch.zeros(len(layer_nodes), dtype=settings.DTYPE, device=self.net.device)
            # # print(W_cac.shape, b_cac.shape)
            # for idx, node in enumerate(layer_nodes):
            #     W_cac[idx] = backsub_dict[node][:-1]
            #     b_cac[idx] = backsub_dict[node][-1]
            # # print(W_cac)


            # W_cac_plus = torch.clamp(W_cac, min=0.)
            # W_cac_minus = torch.clamp(W_cac, max=0.)


            # lower_cac = W_cac_plus @ self.lbs_init + W_cac_minus @ self.ubs_init + b_cac
            # upper_cac = W_cac_plus @ self.ubs_init + W_cac_minus @ self.lbs_init + b_cac

            # print()
            # print('lbs cac:', lower_cac)
            # print('ubs cac:', upper_cac)


            # # exit()
            # print()


            # # ##################################################
            # # ##################################################
            # # ##################################################

            if not stat: # conflict
                return False, cc, None

            # print(len(hidden_bounds), lidx)

            # bounds_mapping = {}
            # for idx, (lb, ub) in enumerate(hidden_bounds):
            #     b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
            #     assert len(b) == len(self.layers_mapping[idx+lidx])
            #     bounds_mapping.update(dict(zip(self.layers_mapping[idx+lidx], b)))
            #     # print(idx+lidx)

            # for node in bounds_mapping:
            #     l, u = bounds_mapping[node]

            #     new_status = None
            #     if l > 0:
            #         new_status = True
            #     elif u <= 0:
            #         new_status = False

            #     if new_status is not None:
            #         if node in assignment:
            #             if assignment[node] != new_status:
            #                 print(assignment[node], new_status, node, l, u)
            #                 raise
            #         else:
            #             implications[node] = {'pos': new_status, 'neg': not new_status}



            # print(implications)

            # Timers.tic('Random example')
            # tmp_input = (lbs + ubs) / 2
            # tmp_output = self.net.forward_layer(tmp_input, lidx)
            # Timers.toc('Random example')
            # if self.spec.check_solution(tmp_output):
            #     # TODO: correct this
            #     self.solution = tmp_input
            #     return True, {}, None


            Timers.tic('Heuristic Decision Update')
            if self.decider is not None and settings.DECISION != 'RANDOM':
                self.decider.update(layer_bounds=bounds)
            Timers.toc('Heuristic Decision Update')

            Timers.tic('Implications')
            for node in unassigned_nodes:
                if node in implications:
                    continue

                new_status = None
                if bounds[node]['lb'] > -1e-6:
                    implications[node] = {'pos': True, 'neg': False}
                    new_status = True
                elif bounds[node]['ub'] <= 1e-6:
                    implications[node] = {'pos': False, 'neg': True}
                    new_status = False
                
                if node in assignment and new_status is not None:
                    if assignment[node] != new_status:
                        raise

            # print(implications)

            # print('implications   :', list(implications.keys()))
            Timers.toc('Implications')

        # # check again
        # lbs, ubs = [], []
        # for node in self.gurobi_vars:
        #     # lower bound
        #     self.model.setObjective(node, grb.GRB.MINIMIZE)
        #     self.model.optimize()
        #     lbs.append(self.model.objval)

        #     # upper bound
        #     self.model.setObjective(node, grb.GRB.MAXIMIZE)
        #     self.model.optimize()
        #     ubs.append(self.model.objval)

        # lbs = torch.tensor(lbs, dtype=settings.DTYPE, device=self.net.device)
        # ubs = torch.tensor(ubs, dtype=settings.DTYPE, device=self.net.device)
        # (lower, upper), _ = self.deeppoly(lbs, ubs, assignment)

        # same = torch.equal(lbs, self.lbs_init) or torch.equal(ubs, self.ubs_init)

        # # print('same', same)
        # # print('lbs:', lbs)
        # # print('ubs:', ubs)

        # stat, _ = self.spec.check_output_reachability(lower, upper)

        # if not stat: # conflict
        #     print('hehe')
        #     # exit()
        #     return False, cc, None



        # # check again again

        # lidx = 0
        # layer_nodes = list(self.layers_mapping[lidx])
        # lbs, ubs = [], []
        # for node in layer_nodes:
        #     obj = self._get_equation(backsub_dict[node])

        #     # lower bound
        #     self.model.setObjective(obj, grb.GRB.MINIMIZE)
        #     self.model.optimize()
        #     lbs.append(self.model.objval)

        #     # upper bound
        #     self.model.setObjective(obj, grb.GRB.MAXIMIZE)
        #     self.model.optimize()
        #     ubs.append(self.model.objval)

        # lbs = torch.tensor(lbs, dtype=settings.DTYPE, device=self.net.device)
        # ubs = torch.tensor(ubs, dtype=settings.DTYPE, device=self.net.device)
        # # print('lbs:', lbs)
        # # print('ubs:', ubs)
        # lower, upper = self.deeppoly.forward_layer(lbs, ubs, lidx)
        # stat, _ = self.spec.check_output_reachability(lower, upper)
        # if not stat: # conflict
        #     print('hehe 2')
        #     # exit()
        #     return False, cc, None


        return True, implications, is_full_assignment

    def _optimize(self):
        self.model.update()
        self.model.reset()
        self.model.optimize()


    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self._optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)
        return None


    def check_solution(self, solution):
        if torch.any(solution < self.lbs_init.view(self.net.input_shape)) or torch.any(solution > self.ubs_init.view(self.net.input_shape)):
            return False
        if self.spec.check_solution(self.net(solution)):
            return True
        return False

    def _compute_output_abstraction(self, lbs, ubs, assignment=None):
        if settings.HEURISTIC_DEEPZONO: # eran deepzono
            return self.deepzono(lbs, ubs)
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

    def get_cache_assignment(self, assignment):
        if len(self.last_assignment) == 0 or len(assignment) == 0:
            return []

        cache_nodes = []
        for idx, variables in self.layers_mapping.items():
            a1 = {n: self.last_assignment.get(n, None) for n in variables}
            a2 = {n: assignment.get(n, None) for n in variables}
            if a1 == a2:
                tmp = [n for n in a1 if a1[n] is not None]
                cache_nodes += tmp
                if len(tmp) < len(a1):
                    break
            else:
                for n in variables:
                    if n in assignment and n in self.last_assignment and assignment[n]==self.last_assignment[n]:
                        cache_nodes.append(n)
                break
        return cache_nodes