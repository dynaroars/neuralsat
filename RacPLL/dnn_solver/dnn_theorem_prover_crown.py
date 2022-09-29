from pprint import pprint
import gurobipy as grb
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

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.utils import *
from abstract.crown import *


class DNNTheoremProverCrown:

    def __init__(self, net, spec, decider=None):

        torch.manual_seed(arguments.Config["general"]["seed"])
        random.seed(arguments.Config["general"]["seed"])
        np.random.seed(arguments.Config["general"]["seed"])

        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec
        self.verified = False
        self.decider = decider


        ##########################################################################################

        # with contextlib.redirect_stdout(open(os.devnull, 'w')):
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
        self.transformer = SymbolicNetwork(net)
        self.last_assignment = {}        
        self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)
        
        ##########################################################################################
        prop_mat, prop_rhs = spec.mat[0]
        if len(prop_rhs) > 1:
            raise
        else:
            assert len(prop_mat) == 1
            y = np.where(prop_mat[0] == 1)[0]
            if len(y) != 0:
                y = int(y)
            else:
                y = None
            target = np.where(prop_mat[0] == -1)[0]  # target label
            target = int(target) if len(target) != 0 else None  # Fix constant specification with no target label.
            if y is not None and target is None:
                y, target = target, y  # Fix vnnlib with >= const property.
            decision_thresh = prop_rhs[0]
            arguments.Config["bab"]["decision_thresh"] = decision_thresh

        if y is not None:
            if net.n_output > 1:
                c = torch.zeros((1, 1, net.n_output), dtype=settings.DTYPE, device=net.device)  # we only support c with shape of (1, 1, n)
                c[0, 0, y] = 1
                c[0, 0, target] = -1
            else:
                # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
                c = (float(y) - 0.5) * 2 * torch.ones(size=(1, 1, 1), dtype=settings.DTYPE, device=net.device)
        else:
            # if there is no ture label, we only verify the target output
            c = torch.zeros((1, 1, net.n_output), dtype=settings.DTYPE, device=net.device)  # we only support c with shape of (1, 1, n)
            c[0, 0, target] = -1

        print(f'##### True label: {y}, Tested against: {target} ######')

        if net.dataset == 'mnist':
            input_shape = (1, 1, 28, 28)
        elif net.dataset == 'cifar':
            input_shape = (1, 3, 32, 32)
        elif net.dataset == 'test':
            input_shape = (1, 1)
        else:
            raise

        x_range = torch.tensor(spec.bounds, dtype=settings.DTYPE, device=net.device)
        data_min = x_range[:, 0].reshape(input_shape)
        data_max = x_range[:, 1].reshape(input_shape)
        data = x_range.mean(1).reshape(input_shape)
        # print(x_range.shape, data_max.shape)

        # print((data_max - data_min).sum(), net.n_output)
        # print(c)
        # print(net.layers)

        self.lirpa = LiRPAConvNet(net.layers, y, target, device=net.device, in_size=input_shape, deterministic=False, conv_mode='patches', c=c)

        print('Model prediction is:', self.lirpa.net(data))

        self.decision_thresh = decision_thresh

        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=data_min, x_U=data_max)
        self.x = BoundedTensor(data, ptb).to(net.device)



        # exit()
        self.count = 0

        self.domains = []

        self.solution = None

        self.last_dl = None
        self.last_domain = None



    def _get_equation(self, coeffs):
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        return expr

    def _find_unassigned_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes


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


    def __call__(self, assignment, info=None):
        print('\n\n-------------------------------------------------\n')
        
        self.count += 1
        cc = frozenset()
        implications = {}
        cur_var, dl, branching_decision = info
        print(cur_var, dl, branching_decision, assignment.get(cur_var, None), assignment)
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False

        # initialize
        if len(assignment) == 0:
            global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = self.lirpa.build_the_model(
                None, self.x, stop_criterion_func=stop_criterion_sum(self.decision_thresh))

            self.pre_relu_indices = pre_relu_indices

            if global_lb >= self.decision_thresh:
                return False, cc, None

            if True:
                # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
                # We only keep the alpha for the last layer.
                new_slope = defaultdict(dict)
                output_layer_name = self.lirpa.net.final_name
                for relu_layer, alphas in slope.items():
                    new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
                slope = new_slope

            candidate_domain = ReLUDomain(lA, global_lb, global_ub, lower_bounds, upper_bounds, slope, history=history, depth=dl, primals=primals).to_device(self.net.device, partial=True)

            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(candidate_domain)
            history = [sd.history for sd in selected_domains]


            # print(unassigned_nodes)
            # print(orig_lbs)
            # print(lower_bounds)


            count = 1
            for lbs, ubs in zip(lower_bounds[:-1], upper_bounds[:-1]):
                # print(lbs)
                if (lbs - ubs).max() > 1e-6:
                    return False, cc, None

                for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):

                    if u <= 0:
                        implications[count + jj] = {'pos': False, 'neg': True}
                    elif l > 0:
                        implications[count + jj] = {'pos': True, 'neg': False}
                count += lbs.numel()




                # print(lbs)
                # print(ubs)
                # print()
            # print(len(implications))

            crown_params = orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, slopes, betas, history
            if self.decider is not None and settings.DECISION != 'RANDOM':
                self.decider.update(crown_params=crown_params)

            self.last_dl = dl

            # self.domains.append(candidate_domain)
            self.last_domain = candidate_domain

            return True, implications, is_full_assignment

        if is_full_assignment:
            output_mat, backsub_dict = self.transformer(assignment)
            backsub_dict_expr = self.backsub_cacher.get_cache(assignment)

            if backsub_dict_expr is not None:
                backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
            else:
                backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

            self.backsub_cacher.put(assignment, backsub_dict_expr)
            constrs = []
            for node, status in assignment.items():
                if status:
                    constrs.append(self.model.addLConstr(backsub_dict_expr[node] >= 1e-6))
                else:
                    constrs.append(self.model.addLConstr(backsub_dict_expr[node] <= 0))
            
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

            self.model.remove(constrs)
            
            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            return False, cc, None


        if (self.last_dl is not None) and (self.last_dl == dl) and (self.last_domain is not None):
            print('vao day')
            # implication iteration
            # TODO: haven't known yet
            return True, implications, is_full_assignment

        print('branching_decision:', cur_var, dl, branching_decision)
        # last_domain = self.domains[-1]
        if self.last_domain is not None:
            # assert dl == self.last_domain.depth + 1, f"{dl}, {self.last_domain.depth}"

            mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(self.last_domain)
            history = [sd.history for sd in selected_domains]
            split_history = [sd.split_history for sd in selected_domains]
            # print(last_domain.depth, selected_domains[0].depth)
            split = {}
            split["decision"] = [[bd] for bd in branching_decision]
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]
            split["diving"] = 0
            print(split)


            ret = self.lirpa.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history,
                                    split_history=split_history, layer_set_bound=True, betas=betas,
                                    single_node_split=True, intermediate_betas=intermediate_betas)

            dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = ret

            batch = 1
            domain_list = add_domain_parallel(lA=lAs[:2*batch], lb=dom_lb[:2*batch], ub=dom_ub[:2*batch], lb_all=dom_lb_all[:2*batch], up_all=dom_ub_all[:2*batch],
                                             domains=None, selected_domains=selected_domains[:batch], slope=slopes[:2*batch], beta=betas[:2*batch],
                                             growth_rate=0, branching_decision=branching_decision, decision_thresh=self.decision_thresh,
                                             split_history=split_history[:2*batch], intermediate_betas=intermediate_betas[:2*batch],
                                             check_infeasibility=False, primals=primals[:2*batch] if primals is not None else None)
            # for d in domain_list:
            #     print(d.lower_bound)
            # print(cur_var, assignment[cur_var])

            if assignment[cur_var]:
                self.last_domain, save_domain = domain_list
            else:
                save_domain, self.last_domain = domain_list

            print(self.last_domain.lower_bound)

            self.domains.append(save_domain)

            # if self.last_domain.lower_bound >= self.decision_thresh:
            #     self.last_domain = None
            #     self.last_dl = None
            #     print('unsat')
            #     return False, cc, None


        else:
            print('Back to', dl, cur_var, assignment[cur_var])
            self.last_domain = self.domains.pop()



        if self.last_domain.lower_bound >= self.decision_thresh:
            self.last_domain = None
            self.last_dl = None
            print('unsat')
            return False, cc, None


        mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(self.last_domain)
        history = [sd.history for sd in selected_domains]


        crown_params = orig_lbs, orig_ubs, mask, self.lirpa, self.pre_relu_indices, lAs, slopes, betas, history
        if self.decider is not None and settings.DECISION != 'RANDOM':
            self.decider.update(crown_params=crown_params)

        self.last_dl = dl


        count = 1
        for lbs, ubs in zip(orig_lbs[:-1], orig_ubs[:-1]):
            if (lbs - ubs).max() > 1e-6:
                return False, cc, None

            for jj, (l, u) in enumerate(zip(lbs[0].flatten(), ubs[0].flatten())):
                node = count + jj
                if node in assignment:
                    continue
                if u <= 0:
                    implications[node] = {'pos': False, 'neg': True}
                elif l > 0:
                    implications[node] = {'pos': True, 'neg': False}
            count += lbs.numel()

        print(implications)

        return True, implications, is_full_assignment


    def restore_input_bounds(self):
        pass


    def get_domain_params(self, selected_candidate_domain):
        lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []
        device = self.net.device
        batch = 1

        lAs.append(selected_candidate_domain.lA)
        lower_all.append(selected_candidate_domain.lower_all)
        upper_all.append(selected_candidate_domain.upper_all)
        slopes_all.append(selected_candidate_domain.slope)
        betas_all.append(selected_candidate_domain.beta)
        intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
        selected_candidate_domains.append(selected_candidate_domain)
        lower_bounds = []

        for j in range(len(lower_all[0])):
            lower_bounds.append(torch.cat([lower_all[i][j]for i in range(batch)]))
        lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

        upper_bounds = []
        for j in range(len(upper_all[0])):
            upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
        upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

        # Reshape to batch first in each list.
        new_lAs = []
        for j in range(len(lAs[0])):
            new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
        # Transfer to GPU.
        new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

        slopes = []
        if slopes_all[0] is not None:
            if isinstance(slopes_all[0], dict):
                # Per-neuron slope, each slope is a dictionary.
                slopes = slopes_all
            else:
                for j in range(len(slopes_all[0])):
                    slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

        # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
        lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
        upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]
        
        # Recompute the mask on GPU.
        new_masks = []
        for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
            new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())
        return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains
