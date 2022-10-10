import torch
import copy
import gurobipy as grb
import numpy as np
import contextlib
import multiprocessing
import torch.nn as nn
import contextlib
import random
import time
import re
import os

from dnn_solver.symbolic_network import SymbolicNetwork
import settings

class ReLUDomain:

    def __init__(self, net, input_lower, input_upper, assignment=None, bounds_mapping=None):

        self.assignment = assignment
        self.valid = True 
        self.unsat = False 
        self.input_lower = input_lower
        self.input_upper = input_upper

        self.net = net
        self.layers_mapping = net.layers_mapping
        self.transformer = SymbolicNetwork(net)
        self.reversed_layers_mapping = {n: k for k, v in self.layers_mapping.items() for n in v}

        self.bounds_mapping = bounds_mapping
        
        self.output_lower = None
        self.output_upper = None
        # self.unassigned_nodes = self._find_unassigned_nodes(assignment)
        # self.full_assignment = True if self.unassigned_nodes is None else False
        # if self.full_assignment:
        #     self.valid = False

        # self.layer_idx = self.reversed_layers_mapping[list(self.unassigned_nodes)[0]]
        # self.layer_nodes = list(self.layers_mapping[self.layer_idx])

        # print(self.assignment, self.layer_idx, self.unassigned_nodes)
        self.backsub_dict = None

    def init_optimizer(self):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)
    
        [self.model.addVar(name=f'x{i}', lb=self.input_lower[i], ub=self.input_upper[i]) for i in range(self.net.n_input)]
        self.model.update()

    def clone(self, node, status):
        new_assignment = copy.deepcopy(self.assignment)
        new_assignment[node] = status

        # print('\t + cloned:', new_assignment)

        full_assignment = {}
        full_assignment.update(new_assignment)
        for n in self.bounds_mapping:
            if n in full_assignment:
                continue
            l, u = self.bounds_mapping[n]
            if l >= -1e-6:
                full_assignment[n] = True
            elif u <= 1e-6:
                full_assignment[n] = False
        output_mat, backsub_dict = self.transformer(full_assignment)

        # print(node)
        # print(full_assignment)
        # print((backsub_dict.keys()))
        new_bounds_mapping = {}
        new_bounds_mapping.update(self.bounds_mapping)
        l, u = self.bounds_mapping[node]
        # print(node, l, u, l.clamp(min=0), u.clamp(min=0))
        if status:
            new_bounds_mapping[node] = (l.clamp(min=0), u)
        else:
            new_bounds_mapping[node] = (l, u.clamp(max=0))


        new_domain = ReLUDomain(self.net, self.input_lower, self.input_upper, new_assignment, new_bounds_mapping)

        # output_mat, backsub_dict = self.transformer(new_assignment)
        new_domain.model = self.model.copy()
        # assert id(new_domain.model) == id(self.model)
        new_vars = new_domain.model.getVars()
        coeffs = backsub_dict[node]
        cstr = grb.LinExpr(coeffs[:-1], new_vars) + coeffs[-1]
        new_domain.backsub_dict = backsub_dict
        if status:
            new_domain.model.addLConstr(cstr >= 1e-6)
        else:
            new_domain.model.addLConstr(cstr <= 0)
        new_domain.model.update()
        return new_domain

    def get_layer_bounds(self, lid):
        lb, ub = [], []
        for node in self.layers_mapping[lid]:
            l, u = self.bounds_mapping[node]
            lb.append(l)
            ub.append(u)
        return torch.stack([torch.tensor(lb, dtype=settings.DTYPE, device=self.net.device), torch.tensor(ub, dtype=settings.DTYPE, device=self.net.device)])


    def optimize_bounds(self):

        # print(len(self.assignment), len(self.model.getConstrs()))
        # self.model.write(f'gurobi/{hash(frozenset(self.assignment.items()))}.lp')

        # output_mat, backsub_dict = self.transformer(self.assignment)
        lid, lnodes = self.get_layer_nodes()
        unassigned_nodes = [n for n in lnodes if self.bounds_mapping[n][0] < 0 < self.bounds_mapping[n][1]]
        # print('unassigned_nodes:', unassigned_nodes)
        variables = self.model.getVars()

        for node in lnodes:
            coeffs = self.backsub_dict[node]
            obj = grb.LinExpr(coeffs[:-1], variables) + coeffs[-1]
            
            self.model.setObjective(obj, grb.GRB.MINIMIZE)
            self.model.update()
            # model.write(f'gurobi/{hash(frozenset(self.assignment.items()))}_{node}.lp')
            self.model.reset()
            self.model.optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                self.unsat = True
                break
            lb = self.model.objval

            self.model.setObjective(obj, grb.GRB.MAXIMIZE)
            self.model.update()
            self.model.reset()
            self.model.optimize()
            ub = self.model.objval

            # old_lb, old_ub = self.bounds_mapping[node]
            # if lb > old_lb or ub < old_ub:
            #     print(f'[{node}] from ({old_lb:.02f}, {old_ub:.02f}) to ({lb:.02f}, {ub:.02f})')
            self.bounds_mapping[node] = (lb, ub)

        # lid, lnodes = self.get_layer_nodes()
        # unassigned_nodes = [n for n in lnodes if self.bounds_mapping[n][0] < 0 < self.bounds_mapping[n][1]]
        # print('unassigned_nodes:', unassigned_nodes)
        return lid

    def deserialize(self):
        full_assignment = {}
        full_assignment.update(self.assignment)
        for node in self.bounds_mapping:
            if node in full_assignment:
                continue
            l, u = self.bounds_mapping[node]
            if l >= -1e-6:
                full_assignment[node] = True
            elif u <= 1e-6:
                full_assignment[node] = False
        output_mat, backsub_dict = self.transformer(full_assignment)
        lid, lnodes = self.get_layer_nodes()
        return self.assignment, backsub_dict, (lid, lnodes), (self.input_lower, self.input_upper)



    def get_next_variable(self):
        # if self.full_assignment:
        #     return None
        lid, lnodes = self.get_layer_nodes()
        if lnodes is None: 
            return None
        unassigned_nodes = [n for n in lnodes if self.bounds_mapping[n][0] < 0 < self.bounds_mapping[n][1]]
        # print(self.assignment)
        # print(unassigned_nodes)
        if not len(unassigned_nodes): 
            return None
        scores = [(n, self.get_score(n)) for n in unassigned_nodes]
        scores = sorted(scores, key=lambda tup: tup[1], reverse=False)
        return scores[0][0]
    


    def get_score(self, node):

        l, u = self.bounds_mapping[node]
        # score = (u - l)
        # score = torch.min(u, -l)
        score = (u + l) / (u - l)
        # print(score, u, l)
        # exit()
        return score.abs()


    
    def get_layer_nodes(self):
        for lidx, nodes in self.layers_mapping.items():
            for node in nodes:
                if node in self.assignment:
                    continue
                lb, ub = self.bounds_mapping[node]
                if lb < 0 < ub:
                    # print(node, lb, ub)
                    return lidx, nodes
        return None, None


    