from multiprocessing import Pool
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
from abstract.crown import CrownWrapper
from dnn_solver.worker import *

from utils.cache import BacksubCacher, AbstractionCacher
from utils.read_nnet import NetworkNNET
from utils.timer import Timers
from utils.misc import MP
import settings

from lp_solver.glpk_solver import GLPKSolver



        
def init_worker(wfunc, solver):
    wfunc.solver = solver
    # print(wfunc)


def worker_func(param):
    # print(param)

    solver = worker_func.solver

    idx = param

    lb = solver.minimize_output(idx)
    ub = solver.minimize_output(idx, maximize=True)

    return idx, (lb, ub)



class DNNTheoremProverGurobi:

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

        self.transformer = SymbolicNetwork(net)

        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=1000)


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

        Timers.tic('Randomized attack')
        self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)
        stat, adv = self.rf.eval(timeout=settings.FALSIFICATION_TIMEOUT)
        if settings.DEBUG:
            print('Randomized attack:', stat)
        if stat == 'violated':
            self.solution = adv[0]
        Timers.toc('Randomized attack')

        self.crown = CrownWrapper(net)
        self.deepzono = deepzono.DeepZono(net)

        # self.glpk_solver = GLPKSolver(net.n_input)
        self.verified = False

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
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        return expr

    @torch.no_grad()
    def __call__(self, assignment, info=None):

        # debug
        # print(assignment)
        self.count += 1
        cc = frozenset()

        if self.solution is not None:
            return True, {}, None


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


        # caching assignment
        self.last_assignment = assignment.copy()

        if len(new_nodes) == 0 and len(remove_nodes) == 0 and len(assignment) > 0:
            return True, {}, is_full_assignment

        Timers.tic('remove constraints')
        if len(remove_nodes):
            remove_constraints = []
            for node in remove_nodes:
                cr = self.model.getConstrByName(f'cstr[{node}]')
                if cr is not None:
                    remove_constraints.append(cr)
            self.model.remove(remove_constraints)
        Timers.toc('remove constraints')



        Timers.tic('get cache backsub_dict')
        backsub_dict_expr = self.backsub_cacher.get_cache(assignment)
        Timers.toc('get cache backsub_dict')

        Timers.tic('Get Linear Equation')
        if backsub_dict_expr is not None:
            backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
        else:
            backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

        self.backsub_cacher.put(assignment, backsub_dict_expr)
        Timers.toc('Get Linear Equation')


        # add constraints
        Timers.tic('Add constraints')
        # Timers.tic('Gurobi functions')
        if len(new_nodes):
            for node in new_nodes:
                status = assignment.get(node, None)
                # assert status is not None
                if status:
                    ci = self.model.addLConstr(backsub_dict_expr[node] >= 1e-6, name=f'cstr[{node}]')
                else:
                    ci = self.model.addLConstr(backsub_dict_expr[node] <= 0, name=f'cstr[{node}]')
        # Timers.toc('Gurobi functions')
        Timers.toc('Add constraints')
        self.model.update()

        


        # check satisfiability
        if not is_full_assignment:
            Timers.tic('Check output property')
            self._optimize()
            Timers.toc('Check output property')
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('call from partial assignment')
                # self._restore_input_bounds()
                return False, cc, None

            if settings.DEBUG:
                print('[+] Check partial assignment: `SAT`')


        else: # output
            Timers.tic('Check output property')
            flag_sat = False
            output_constraint = self.spec.get_output_property(
                [self._get_equation(output_mat[i]) for i in range(self.net.n_output)]
            )
            self.optimize_input_bounds()
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


        
        bounds = {}
        lidx = self.reversed_layers_mapping[list(unassigned_nodes)[0]]
        layer_nodes = list(self.layers_mapping[lidx])

        implications = {}
        hidden_bounds = {}

        # Timers.tic('Tighten bounds')
        self.optimize_input_bounds()

        # li = [x.lb for x in self.gurobi_vars]
        # ui = [x.ub for x in self.gurobi_vars]

        # Timers.tic('GLPK functions')
        # self.glpk_solver.build(backsub_dict, assignment, li, ui, normalize=False)


        # init_arg = (worker_func, self.glpk_solver)
        # params = layer_nodes

        # with Pool(initializer=init_worker, initargs=init_arg, processes=50) as pool:

        #     for i, res in enumerate(pool.imap_unordered(worker_func, params)):
        #         pass

        # pool.close() # do we need this? we're in a manager

        # Timers.toc('GLPK functions')


        Timers.tic('Gurobi functions')
        for node in layer_nodes:
            obj = backsub_dict_expr[node]
            # lower bound
            self.model.setObjective(obj, grb.GRB.MINIMIZE)
            self.model.optimize()
            lb = self.model.objval
            # upper bound
            self.model.setObjective(obj, grb.GRB.MAXIMIZE)
            self.model.optimize()
            ub = self.model.objval
            # else:
            #     status = assignment[node]
            #     # print(node, status)
            #     mat = backsub_dict[node]
            #     w_pos = torch.clamp(mat[:-1], min=0)
            #     w_neg = torch.clamp(mat[:-1], max=0)

            #     lb = w_pos @ self.lbs_init + w_neg @ self.ubs_init + mat[-1]
            #     ub = w_pos @ self.ubs_init + w_neg @ self.lbs_init + mat[-1]

            #     if status:
            #         lb = max(lb, 0)
            #     else:
            #         ub = min(ub, 0)

                # print(lb, ub)
                # print()
            bounds[node] = {'lb': lb, 'ub': ub}

        # Timers.toc('Tighten bounds')
        Timers.toc('Gurobi functions')

        lbs = torch.tensor([bounds[node]['lb'] for node in layer_nodes], dtype=settings.DTYPE, device=self.net.device)
        ubs = torch.tensor([bounds[node]['ub'] for node in layer_nodes], dtype=settings.DTYPE, device=self.net.device)
        # Timers.tic('DeepPoly')
        # (lower, upper), hidden_bounds = self.deeppoly.forward_layer(lbs, ubs, lidx)
        # Timers.toc('DeepPoly')

        # print('---------------------------')
        # print(lower)
        # print(upper)
        # print()

        Timers.tic('Crown functions')
        (lower, upper), unstable_neurons = self.crown.forward_layer(lbs, ubs, lidx)
        Timers.toc('Crown functions')


        # Timers.tic('DeepZono functions')
        # lower, upper = self.deepzono.forward_layer(lbs, ubs, lidx)
        # Timers.toc('DeepZono functions')


        # print(lower2)
        # print(upper2)
        # print('---------------------------')

        stat, _ = self.spec.check_output_reachability(lower, upper)


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


        Timers.tic('Heuristic Decision Update')
        if self.decider is not None and settings.DECISION != 'RANDOM':
            self.decider.update(layer_bounds=bounds, hidden_bounds=hidden_bounds)
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
            
            # if node in assignment and new_status is not None:
            #     if assignment[node] != new_status:
            #         raise

        # print('implications   :', list(implications.keys()))
        Timers.toc('Implications')

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


    def restore_input_bounds(self):
        for i in range(len(self.gurobi_vars)):
            self.gurobi_vars[i].lb = self.lbs_init[i]
            self.gurobi_vars[i].ub = self.ubs_init[i]
        self.model.update()


    def optimize_input_bounds(self):
        Timers.tic('Gurobi functions')
        # Timers.tic('Tighten input bounds')
        self.model.update()
        for i, v in enumerate(self.gurobi_vars):

            # lower bound
            self.model.setObjective(v, grb.GRB.MINIMIZE)
            self.model.optimize()
            v.lb = self.model.objval
            # upper bound
            self.model.setObjective(v, grb.GRB.MAXIMIZE)
            self.model.optimize()
            v.ub = self.model.objval
        self.model.update()
        Timers.toc('Gurobi functions')
        # Timers.toc('Tighten input bounds')