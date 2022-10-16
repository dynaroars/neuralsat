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
import pickle
from contextlib import contextmanager

from batch_processing import deeppoly, domain, gradient_abstractor
from dnn_solver.symbolic_network import SymbolicNetwork
from dnn_solver.worker import *

from utils.cache import BacksubCacher
from utils.timer import Timers
from abstract.crown import arguments
from utils.misc import MP
import settings


# @contextmanager
# def poolcontext(*args, **kwargs):
#     pool = multiprocessing.Pool(*args, **kwargs)
#     yield pool
#     pool.terminate()

# def optimize_bounds(param):

#     jj, (assignment, backsub_dict, (lid, lnodes), (input_lower, input_upper)) = param
#     # print('processing:', jj)
#     # output_mat,  = self.transformer(self.assignment)
#     # lid, lnodes = self.get_layer_nodes()
#     # unassigned_nodes = [n for n in lnodes if self.bounds_mapping[n][0] < 0 < self.bounds_mapping[n][1]]
#     # print('unassigned_nodes:', unassigned_nodes)
#     unsat = False
#     bounds = {}
#     with grb.Env() as env, grb.Model(env=env) as model:
#         model.setParam('OutputFlag', False)

#         variables = [model.addVar(name=f'x{i}', lb=input_lower[i], ub=input_upper[i]) for i in range(len(input_lower))]
#         mvars = grb.MVar(variables)

#         lhs = np.zeros([len(assignment), len(variables)])
#         rhs = np.zeros(len(assignment))
#         for i, (node, status) in enumerate(assignment.items()):
#             if status:
#                 lhs[i] = -1 * backsub_dict[node][:-1]
#                 rhs[i] = backsub_dict[node][-1] - 1e-6
#             else:
#                 lhs[i] = backsub_dict[node][:-1]
#                 rhs[i] = -1 * backsub_dict[node][-1]

#         model.addConstr(lhs @ mvars <= rhs) 
#         model.update()
#         # model.write(f'gurobi/{hash(frozenset(self.assignment.items()))}.lp')

#         for node in lnodes:
#             coeffs = backsub_dict[node]
#             obj = grb.LinExpr(coeffs[:-1], variables) + coeffs[-1]
            
#             model.setObjective(obj, grb.GRB.MINIMIZE)
#             model.update()
#             # model.write(f'gurobi/{hash(frozenset(self.assignment.items()))}_{node}.lp')
#             model.optimize()
#             if model.status == grb.GRB.INFEASIBLE:
#                 unsat = True
#                 # continue
#                 break
#             lb = model.objval

#             model.setObjective(obj, grb.GRB.MAXIMIZE)
#             model.update()
#             model.optimize()
#             ub = model.objval

#             # old_lb, old_ub = bounds_mapping[node]
#             # if lb > old_lb or ub < old_ub:
#             #     print(f'[{node}] from ({old_lb:.02f}, {old_ub:.02f}) to ({lb:.02f}, {ub:.02f})')
#             bounds[node] = (lb, ub)

#     # lid, lnodes = self.get_layer_nodes()
#     # unassigned_nodes = [n for n in lnodes if self.bounds_mapping[n][0] < 0 < self.bounds_mapping[n][1]]
#     # print('unassigned_nodes:', unassigned_nodes)
#     return jj, (lid, assignment, unsat, bounds)



class DNNTheoremProverGurobi:

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        # self.hidden_nodes = sum([len(v) for k, v in self.layers_mapping.items()])

        self.decider = decider

        # with contextlib.redirect_stdout(open(os.devnull, 'w')):
        if 1:
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)
            self.model.setParam('FeasibilityTol', 1e-8)
            

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        # print(self.lbs_init.shape)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) for i in range(self.net.n_input)
        ]
        self.mvars = grb.MVar(self.gurobi_vars)

        self.count = 0 # debug

        self.solution = None

        self.transformer = SymbolicNetwork(net)

        self.deeppoly = deeppoly.BatchDeepPoly(net, back_sub_steps=1000)
        self.ga = gradient_abstractor.GradientAbstractor(net, spec)

        # self.concrete = self.net.get_concrete((self.lbs_init + self.ubs_init) / 2.0)
        # self.reversed_layers_mapping = {n: k for k, v in self.layers_mapping.items() for n in v}

        # self.last_assignment = {}        

        # # pgd attack 
        # self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)

        # Timers.tic('Randomized attack')
        # self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)
        # stat, adv = self.rf.eval(timeout=settings.FALSIFICATION_TIMEOUT)
        # if settings.DEBUG:
        #     print('Randomized attack:', stat)
        # if stat == 'violated':
        #     self.solution = adv[0]
        # Timers.toc('Randomized attack')

        # self.crown = CrownWrapper(net)
        # self.deepzono = deepzono.DeepZono(net)

        # # self.glpk_solver = GLPKSolver(net.n_input)
        self.verified = False
        # self.optimized_layer_bounds = {}
        self.next_iter_implication = False

        self.batch = arguments.Config["general"]["batch"]
        self.domains = {}

        os.makedirs('gurobi', exist_ok=True)
        self.cac_queue = multiprocessing.Queue()

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

    def add_domains(self, domains):
        for d in domains:
            if d.unsat:
                continue
                
            var = d.get_next_variable()
            if var in d.assignment:
                print('duplicated:', var)
                raise
            if var is None:
                continue
            # f_assignment = copy.deepcopy(d.assignment)
            # f_assignment[var] = False
            new_d1 = d.clone(var, False)
            new_d2 = d.clone(var, True)

            # print(d.assignment)

            self.domains[hash(frozenset(new_d1.assignment.items()))] = new_d1
            self.domains[hash(frozenset(new_d2.assignment.items()))] = new_d2


    def get_domains(self, cur_domain, batch=1):
        # print(f'get {batch} domains')
        ds = [cur_domain]
        if batch == 1:
            return ds
        idx = 1
        for k, v in self.domains.items():
            if v.valid:
                ds.append(v)
                idx += 1
            if idx == batch:
                break
        return ds

    # @torch.no_grad()
    def __call__(self, assignment, info=None, full_assignment=None, use_implication=True):

        # debug
        self.count += 1
        cc = frozenset()
        implications = {}

        if self.solution is not None:
            return True, {}, None


        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # print('\t', self.count, 'full_assignment:', full_assignment)
        # print('\t', self.count, 'unassigned_nodes', unassigned_nodes)
        # print(f'\t[{self.count}] full_assignment:', full_assignment)


        if len(assignment) == 0:
            # print('first time')

            (lbs, ubs), hidden_bounds = self.deeppoly(self.lbs_init.unsqueeze(0), self.ubs_init.unsqueeze(0), return_hidden_bounds=True)
            # print(lbs.shape)
            # print(lbs.shape)
            # print(len(hidden_bounds))

            stat, _ = self.spec.check_output_reachability(lbs[0], ubs[0])

            # print('reachable:', stat)
            if not stat: # conflict
                return False, cc, None

            
            # for bound in hidden_bounds:
            #     print(bound.shape)

            bounds_mapping = {}
            for idx, (lb, ub) in enumerate([b[0] for b in hidden_bounds]):
                b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
                assert len(b) == len(self.layers_mapping[idx])
                bounds_mapping.update(dict(zip(self.layers_mapping[idx], b)))
                
            self.decider.update(bounds_mapping=bounds_mapping)

            d = domain.ReLUDomain(self.net, self.lbs_init, self.ubs_init, assignment, bounds_mapping)
            d.init_optimizer()
            d.valid = False

            self.add_domains([d])

            # implication
            for node, (l, u) in bounds_mapping.items():
                if u <= 1e-6:
                    implications[node] = {'pos': False, 'neg': True}
                elif l >= -1e-6:
                    implications[node] = {'pos': True, 'neg': False}

            if len(implications):
                self.next_iter_implication = True

            return True, implications, is_full_assignment
        
        if is_full_assignment:
            output_mat, backsub_dict = self.transformer(assignment)
            # print('full assignment')
            # raise

            lhs = np.zeros([len(assignment), len(self.gurobi_vars)])
            rhs = np.zeros(len(assignment))
            for i, (node, status) in enumerate(assignment.items()):
                if status:
                    lhs[i] = -1 * backsub_dict[node][:-1]
                    rhs[i] = backsub_dict[node][-1] - 1e-8
                else:
                    lhs[i] = backsub_dict[node][:-1]
                    rhs[i] = -1 * backsub_dict[node][-1]

            self.model.remove(self.model.getConstrs())
            self.model.addConstr(lhs @ self.mvars <= rhs) 
            self.model.update()


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

            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            return False, cc, None



        if self.next_iter_implication:
            self.next_iter_implication = False
            # print('implication')
            # todo: haven't known yet
            return True, implications, is_full_assignment






        # print(f'\t[{self.count}] full_assignment:', full_assignment)

        cur_domain = self.domains[hash(frozenset(full_assignment.items()))] 
        if cur_domain.valid:
            cur_domain.valid = False

            if cur_domain.unsat:
                # print('unsat')
                return False, cc, None
            # print('cur_domain', cur_domain.assignment, cur_domain.output_lower)

            # if cur_domain.output_lower is not None:

            #     stat, _ = self.spec.check_output_reachability(cur_domain.output_lower, cur_domain.output_upper)
            #     print('reachable:', stat)
            #     if not stat: # conflict
            #         return False, cc, None


            #     # implication:
            #     return True, implications, is_full_assignment
            


            ds = self.get_domains(cur_domain, batch=self.batch)
            # print(len(ds))
            for d in ds:
                d.valid = False
                # print(d)
            

            # init_arg = (worker_func, 'a')
            batch_layer_id = torch.zeros(len(ds))

            # test_lb1, test_ub1 = cur_domain.get_layer_bounds(0)
            # print(len(ds))
            # print(cur_domain.assignment == full_assignment)
            # print(cur_domain.deserialize()[2])
            # print([cur_domain.deserialize()[1].keys()])
            # if 1:
            # with contextlib.redirect_stdout(open(os.devnull, 'w')):
            #     with Pool(processes=8) as pool:
            #         for res in pool.imap_unordered(optimize_bounds, [(jj, d.deserialize()) for jj, d in enumerate(ds)]):
            #             idx, (lid, _, unsat, bounds) = res
            #             # print(idx, assignment==ds[idx].assignment, assignment)
            #             ds[idx].bounds_mapping.update(bounds)
            #             ds[idx].unsat = unsat
            #             batch_layer_id[idx] = lid
            #     pool.close()

            Timers.tic('Optimize bounds')
            tic = time.time()
            for idx, d in enumerate(ds):
                lid = d.optimize_bounds()
                if d.unsat:
                    batch_layer_id[idx] = -1
                else:
                    batch_layer_id[idx] = lid

            # print(f'[{self.count}] Optimized {len(ds)} domains in', time.time() - tic)
            Timers.toc('Optimize bounds')

            # exit()

            # print('batch_layer_id', batch_layer_id)
            # test_lb2, test_ub2 = cur_domain.get_layer_bounds(0)

            # print((test_lb2 > test_lb1).sum(), (test_ub2 < test_ub1).sum())
            batch_bounds_mapping = {b: {} for b in range(len(ds))}

            Timers.tic('Abstraction')
            for lid in self.layers_mapping:
                indices = torch.where(batch_layer_id==lid)[0]
                if not len(indices): 
                    continue
                # print(lid, indices)
                batch_layer_bound = torch.stack([ds[i].get_layer_bounds(lid) for i in indices])
                batch_lower = batch_layer_bound[:, 0]
                batch_upper = batch_layer_bound[:, 1]
                # print(batch_lower.shape)
                assert (batch_lower <= batch_upper).all()
                # print(f'\t[{self.count}] lid:', lid, self.layers_mapping[lid])

                # print(f'\t[{self.count}] batch_lower:', batch_lower.numpy().tolist())
                # print(f'\t[{self.count}] batch_upper:', batch_upper.numpy().tolist())
                # with torch.no_grad():
                #     (lbs, ubs), hidden_bounds = self.deeppoly.forward_layer(batch_lower, batch_upper, lid, return_hidden_bounds=True, reset_param=True)

                # print(f'\t[{self.count}] lbs 1:', lbs.numpy().tolist())
                # print(f'\t[{self.count}] ubs 1:', ubs.numpy().tolist())


                (lbs, ubs), hidden_bounds = self.ga.get_optimized_bounds(batch_lower, batch_upper, lid)

                # print(f'\t[{self.count}] lbs 2:', lbs.detach().numpy().tolist())
                # print(f'\t[{self.count}] ubs 2:', ubs.detach().numpy().tolist())

                for bidx in range(len(indices)):
                    stat, _ = self.spec.check_output_reachability(lbs[bidx], ubs[bidx])
                    # if not stat: 
                        # print(bidx, 'unsat roi hehe')
                    ds[indices[bidx]].unsat = not stat

                # print(f'\t[{self.count}] stat:', stat)
                # print(lbs.shape)
                # print(ubs.shape)
                # print(len(hidden_bounds))
                # for bound in hidden_bounds:
                #     print(bound.shape)

                # test_bounds = hidden_bounds[0]
                # print(torch.equal(test_bounds[:, 0], batch_lower))
                # print(torch.equal(test_bounds[:, 1], batch_upper))
                # print('batch hidden bounds')
                for idx, bs in enumerate(hidden_bounds):
                    # print(idx, bs.shape)
                    for bidx in range(len(indices)):
                        # print(bidx)
                        b = [(l, u) for l, u in zip(bs[bidx][0], bs[bidx][1])]
                        assert len(b) == len(self.layers_mapping[idx+lid])
                        batch_bounds_mapping[int(indices[bidx])].update(dict(zip(self.layers_mapping[idx+lid], b)))
            Timers.toc('Abstraction')
                
            # print(batch_bounds_mapping)

            # for bm in batch_bounds_mapping.values():
            #     for node in bm:
            #         assert bm[node][0] <= bm[node][1]
            for bidx, d in enumerate(ds):
                d.bounds_mapping.update(batch_bounds_mapping[bidx])
                
            # for d in ds:
            #     lid = d.optimize_bounds()
            #     print(lid, d.get_layer_bounds(lid))
            # print(id(cur_domain) == id(ds[0]))
            Timers.tic('Add domains')
            self.add_domains(ds)
            Timers.toc('Add domains')

        if cur_domain.unsat:
            # print('unsat')
            return False, cc, None


        self.decider.update(bounds_mapping=cur_domain.bounds_mapping)

        # implication
        # print(' ------- assignment:', assignment)
        # nodes = []
        for node in unassigned_nodes:
            l, u = cur_domain.bounds_mapping[node]
            if u <= 1e-6:
                # nodes.append(node)
                implications[node] = {'pos': False, 'neg': True}
            elif l >= -1e-6:
                # nodes.append(node)
                implications[node] = {'pos': True, 'neg': False}

        if len(implications):
            # print('implication:', nodes)
            self.next_iter_implication = True
            
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
        # if self.spec.check_solution(self.net(solution)):
        #     return True
        return True

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
        return
        # Timers.tic('Gurobi functions')
        Timers.tic('Tighten input bounds')
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
        # Timers.toc('Gurobi functions')
        Timers.toc('Tighten input bounds')