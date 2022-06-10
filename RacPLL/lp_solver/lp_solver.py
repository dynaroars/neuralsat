import gurobipy as grb
import torch.nn as nn
import numpy as np
import contextlib
import torch
import time
import os

import settings


GRB_STATUS_CODE = {
    1 : 'LOADED',
    2 : 'OPTIMAL',
    3 : 'INFEASIBLE',
    9 : 'TIME_LIMIT'
}

class LPSolver:

    def __init__(self, net, spec, abstractor=None):

        self.net = net
        self.spec = spec

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE)
        self.ubs = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE)

        # gurobi optimizer
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model('clause_shortener')
            self.model.setParam('OutputFlag', False)
            # self.model.setParam('Threads', 1)
            self.model.setParam('FeasibilityTol', 1e-8)

        self.input_vars = [self.model.addVar(name=f'x{i}', lb=self.lbs[i], ub=self.ubs[i]) for i in range(self.net.n_input)]
        self.mapping_assignment = {}

        self.mapping_bounds = {}

        _, hidden_bounds = abstractor(self.lbs, self.ubs)
        for idx, (lb, ub) in enumerate(hidden_bounds):
            b = [(l, u) for l, u in zip(lb, ub)]
            self.mapping_bounds.update(dict(zip(self.net.layers_mapping[idx], b)))

        # for k, v in self.mapping_bounds.items():
            # print(k, v)


        self._initialize()


    def _initialize(self):
        idx = 0
        tmp_vars = self.input_vars
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                variables = self.net.layers_mapping.get(idx, None)
                if variables is not None:
                    b_vars = [self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f'a_{i}') for i in variables]
                else:
                    b_vars = [self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f'y_{i}') for i in range(self.net.n_output)]

                exprs = [grb.LinExpr(layer.weight[i], tmp_vars) + layer.bias[i] for i in range(len(b_vars))]
                if variables is not None:
                    self.mapping_assignment.update(dict(zip(variables, exprs)))

                for b_var, expr in zip(b_vars, exprs):
                    self.model.addLConstr(b_var == expr)

                self.model.update()

            elif isinstance(layer, nn.ReLU):
                f_vars = [self.model.addVar(lb=0, ub=grb.GRB.INFINITY, name=f'n_{i}') for i in variables]
                self.model.update()
                for b_var, f_var in zip(b_vars, f_vars):
                #     # self.model.addGenConstrPWL(b_var, f_var, [-1, 0, 1], [0, 0, 1])
                    lb, ub = self.mapping_bounds[int(b_var.VarName[2:])]
                    b_var.ub = ub
                    b_var.lb = lb
                    if lb > 0:
                        self.model.addLConstr(f_var == b_var)
                    elif ub < 0:
                        self.model.addLConstr(f_var == 0)
                    else:
                        self.model.addLConstr(f_var >= b_var)
                        self.model.addLConstr(f_var <= (ub / (ub - lb)) * b_var - lb * ub / (ub - lb))                    
                    # M = max(-lb, ub)
                    # b = self.model.addVar(vtype=grb.GRB.BINARY)

                    # self.model.addConstr(f_var >= b_var)
                    # self.model.addConstr(b_var - b*M <= 0)
                    # self.model.addConstr(b_var + (1-b)*M >= 0)
                    # self.model.addConstr(f_var <= b_var + (1-b)*M)
                    # self.model.addConstr(f_var <= b*M)

                idx += 1
                tmp_vars = f_vars

            # elif isinstance(layer, nn.Conv2d):
            #     pass
            # elif isinstance(layer, nn.Flatten):
            #     pass
            else:
                raise NotImplementedError

        out_vars = [self.model.getVarByName(f'y_{i}') for i in range(self.net.n_output)]
        self.output_constraints = self.spec.get_output_property(out_vars)


    def optimize(self):
        self.model.update()
        # self.model.write(f'gurobi/clause_shortener.lp')
        self.model.reset()
        self.model.optimize()


    def shorten_conflict_clause(self, assignment):
        # print('shorten_conflict_clause:', len(self.model.getConstrs()), 'constraints')
        # print(len(assignment))

        cc = []
        for node, status in assignment.items():
            # print(node, self.mapping_assignment[node])
            fv = self.model.getVarByName(f'n_{node}')
            if status:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] >= 1e-6, name=f'c_b_{node}'))
                cc.append(self.model.addLConstr(fv == self.mapping_assignment[node], name=f'c_f_{node}'))
            else:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] <= 0, name=f'c_b_{node}'))
                cc.append(self.model.addLConstr(fv == 0, name=f'c_f_{node}'))
        self.model.update()

        conflict_clause = set()
        for cnf in self.output_constraints:
            ci = [self.model.addLConstr(_) for _ in cnf]
            self.optimize()
            # print('Solving:', len(self.model.getConstrs()), 'constraints', GRB_STATUS_CODE[self.model.status])
            if self.model.status == grb.GRB.OPTIMAL:
                self.model.remove(ci)
                self.model.remove(cc)
                return frozenset()
            
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('Start computeIIS', GRB_STATUS_CODE[self.model.status])
                self.model.computeIIS()
                for node, value in assignment.items():
                    c = self.model.getConstrByName(f'c_f_{node}')
                    if c.IISConstr:
                        conflict_clause.add(-node if value else node)
                        continue
                    c = self.model.getConstrByName(f'c_b_{node}')
                    if c.IISConstr:
                        conflict_clause.add(-node if value else node)

            self.model.remove(ci)
        self.model.remove(cc)

        return frozenset(conflict_clause)
