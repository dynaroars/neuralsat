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
from dnn_solver.worker import implication_gurobi_worker
from abstract.eran import deepzono, deeppoly
from lp_solver.lp_solver import LPSolver

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


        self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)

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


            # self.cs = LPSolver(net, spec, self.deeppoly)

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

        self.last_assignment = {}        

        # pgd attack 
        if 'mnist' in net.dataset or 'cifar' in net.dataset:
            self.gf = gradient_falsification.GradientFalsification(net, spec)
            stat, adv = self.gf.evaluate()
            if stat:
                assert spec.check_solution(net(adv))
                assert (adv >= gf.lower).all()
                assert (adv <= gf.upper).all()
                # print(adv.shape)
                self.solution = adv



    def _update_input_bounds(self, lbs, ubs):
        if torch.any(lbs > ubs):
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
        cc = frozenset()

        if self.solution is not None:
            return True, {}, None
        # reset constraints
        Timers.tic('Reset solver')
        self._restore_input_bounds()
        Timers.toc('Reset solver')

        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # forward
        Timers.tic('backsub_dict')
        output_mat, backsub_dict = self.transformer(assignment)
        Timers.toc('backsub_dict')

        flag_parallel_implication = False if unassigned_nodes is None else len(unassigned_nodes) > 50
        # parallel implication
        if not is_full_assignment and settings.HEURISTIC_GUROBI_IMPLICATION and flag_parallel_implication:
            backsub_dict_np = {k: v.detach().cpu().numpy() for k, v in backsub_dict.items()}
            kwargs = (self.net.n_input, self.lbs_init.detach().cpu().numpy(), self.ubs_init.detach().cpu().numpy())
            wloads = MP.get_workloads(unassigned_nodes, n_cpus=16)
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

        Timers.tic('Find caching assignment')

        cache_nodes = self.get_cache_assignment(assignment)
        remove_nodes = [n for n in self.last_assignment if n not in cache_nodes]
        new_nodes = [n for n in assignment if n not in cache_nodes]

        assert len(cache_nodes) + len(remove_nodes) == len(self.last_assignment)
        
        if len(remove_nodes):
            self.model.remove([self.model.getConstrByName(f'cstr[{node}]') for node in remove_nodes])

        Timers.toc('Find caching assignment')

        # add constraints
        Timers.tic('Add constraints')
        if len(new_nodes):
            for node in new_nodes:
                status = assignment.get(node, None)
                assert status is not None
                if status:
                    ci = self.model.addLConstr(backsub_dict_expr[node] >= DNNTheoremProver.epsilon, name=f'cstr[{node}]')
                else:
                    ci = self.model.addLConstr(backsub_dict_expr[node] <= 0, name=f'cstr[{node}]')
        Timers.toc('Add constraints')

        # caching assignment
        self.last_assignment = assignment

        # upper objective
        self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MAXIMIZE)

        # check satisfiability
        if not is_full_assignment:
            self._optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('call from partial assignment')
                return False, cc, None

            if settings.DEBUG:
                print('[+] Check partial assignment: `SAT`')


        else: # output
            flag_sat = False
            Timers.tic('Check output property')
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
            return False, cc, None



        # reachable heuristic:
        if settings.TIGHTEN_BOUND:# and (self.count % settings.HEURISTIC_DEEPPOLY_INTERVAL == 0): 
            # compute new input lower/upper bounds
            Timers.tic('Tighten bounds')
            # upper
            if self.model.status == grb.GRB.OPTIMAL:
                ubs = [var.X for var in self.gurobi_vars]
            else:
                ubs = [var.ub for var in self.gurobi_vars]

            # lower
            self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MINIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                lbs = [var.X for var in self.gurobi_vars]
            else:
                lbs = [var.lb for var in self.gurobi_vars]

            Timers.toc('Tighten bounds')

            lbs = torch.tensor(lbs, dtype=settings.DTYPE, device=self.net.device)
            ubs = torch.tensor(ubs, dtype=settings.DTYPE, device=self.net.device)

            Timers.tic('Update bounds')
            stat = self._update_input_bounds(lbs, ubs)
            Timers.toc('Update bounds')
                
            if not stat: # conflict
                # print('call from update bounds')
                return False, cc, None


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
                return False, cc, None

            if settings.HEURISTIC_DEEPPOLY and should_run_again and self.flag_use_backsub:
                Timers.tic('Deeppoly optimization reachability')
                Ml, Mu, bl, bu  = self.deeppoly.get_params()
                lbs_expr = [grb.LinExpr(wl, self.gurobi_vars) + cl for (wl, cl) in zip(Ml.detach().cpu().numpy(), bl.detach().cpu().numpy())]
                ubs_expr = [grb.LinExpr(wu, self.gurobi_vars) + cu for (wu, cu) in zip(Mu.detach().cpu().numpy(), bu.detach().cpu().numpy())]
                dnf_contrs = self.spec.get_output_reachability_constraints(lbs_expr, ubs_expr)
                flag_sat = False
                for cnf, adv_obj in dnf_contrs:
                    ci = [self.model.addLConstr(_) for _ in cnf]
                    self.model.setObjective(adv_obj, grb.GRB.MINIMIZE)
                    self._optimize()
                    self.model.remove(ci)
                    if self.model.status == grb.GRB.OPTIMAL:
                        tmp_input = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)
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
                    return False, cc, None
                Timers.toc('Deeppoly optimization reachability')

            if not self.flag_use_backsub:
                tmp_input = torch.tensor(
                    [random.uniform(lbs[i], ubs[i]) for i in range(self.net.n_input)], 
                    dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)

                if self.check_solution(tmp_input):
                    self.solution = tmp_input
                    return True, {}, None
                self.concrete = self.net.get_concrete(tmp_input)



        # implication heuristic
        if settings.HEURISTIC_RANDOMIZED_FALSIFICATION and 'acasxu' in self.net.dataset:
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

        Timers.tic('Implications')
        implications = {}

        if settings.HEURISTIC_GUROBI_IMPLICATION:
            if flag_parallel_implication:
                for w in self.workers:
                    w.join()

                for w in self.workers:
                    res = Q.get()
                    implications.update(res)

            else:
                for node in unassigned_nodes:
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

    def get_cache_assignment(self, assignment):
        cache_nodes = []
        if len(self.last_assignment) == 0 or len(assignment) == 0:
            return cache_nodes

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