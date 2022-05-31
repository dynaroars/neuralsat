import gurobipy as grb
import torch.nn as nn
import numpy as np
import contextlib
import torch
import time
import os


class LPSolver:

    def __init__(self, net, spec):

        self.net = net
        self.spec = spec

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs = bounds_init['lbs']
        self.ubs = bounds_init['ubs']

        # gurobi optimizer
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model('clause_shortener')
            # self.model.setParam('OutputFlag', False)
            # self.model.setParam('Threads', 16)
            self.model.setParam('IISMethod', 0)

        self.input_vars = [self.model.addVar(name=f'x{i}', lb=self.lbs[i], ub=self.ubs[i]) for i in range(self.net.n_input)]
        self.mapping_assignment = {}

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

            elif isinstance(layer, nn.ReLU):
                f_vars = [self.model.addVar(lb=0, ub=grb.GRB.INFINITY, name=f'n_{i}') for i in variables]
                for b_var, f_var in zip(b_vars, f_vars):
                    self.model.addGenConstrPWL(b_var, f_var, [-1, 0, 1], [0, 0, 1])

                idx += 1
                tmp_vars = f_vars

            # elif isinstance(layer, nn.Conv2d):
            #     pass
            # elif isinstance(layer, nn.Flatten):
            #     pass
            else:
                raise NotImplementedError

        self.model.update()

        out_vars = [self.model.getVarByName(f'y_{i}') for i in range(self.net.n_output)]
        self.output_constraints = self.spec.get_output_property(out_vars)


    def optimize(self):
        self.model.update()
        self.model.write(f'gurobi/clause_shortener.lp')
        self.model.reset()
        self.model.optimize()


    def shorten_conflict_clause(self, assignment):
        print('shorten_conflict_clause:', len(self.model.getConstrs()), 'constraints')
        print(len(assignment))
        conflict_clause = set()

        cc = []
        for node, status in assignment.items():
            if status:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] >= 1e-6, name=f'c_{node}'))
            else:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] <= 0, name=f'c_{node}'))
        self.model.update()

        flag_break = False
        for cnf in self.output_constraints:
            ci = [self.model.addLConstr(_) for _ in cnf]
            print('Solving:', len(self.model.getConstrs()), 'constraints')
            self.optimize()
            print('Done', self.model.status == grb.GRB.INFEASIBLE)

            if self.model.status == grb.GRB.INFEASIBLE:
                print('Start computeIIS', self.model.Params.IISMethod)
                self.model.computeIIS()
                print('Done computeIIS')
                # for c in self.model.getConstrs():
                for node, value in assignment.items():
                    c = self.model.getConstrByName(f'c_{node}')
                    if c.IISConstr:
                        conflict_clause.add(-node if value else node)

                flag_break = True
                
            self.model.remove(ci)
            if flag_break:
                break

        self.model.remove(cc)
        print(conflict_clause)
        exit()

        return frozenset(conflict_clause)
