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
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)
            self.model.setParam('FeasibilityTol', 1e-8)

        self.mapping_assignment = {}

        self.mapping_bounds = {}

        tic = time.time()
        (self.lower_out_init, self.upper_out_init), hidden_bounds = abstractor(self.lbs, self.ubs)
        print(time.time() - tic)
        for idx, (lb, ub) in enumerate(hidden_bounds):
            assert torch.all(lb <= ub)
            b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
            assert len(b) == len(self.net.layers_mapping[idx])
            self.mapping_bounds.update(dict(zip(self.net.layers_mapping[idx], b)))


        self._initialize()

        # for k, v in self.mapping_assignment.items():
        #     print(k, v)


    def _initialize(self, model_type='lp'):
        assert model_type in ['lp', 'mip', 'lp_integer']

        zero_var = self.model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
        gurobi_vars = []

        if len(self.net.input_shape) == 4:
            if self.net.input_shape[-1] == 3:
                _, s2, s3, s1 = self.net.input_shape
            elif self.net.input_shape[1] == 3:
                _, s1, s2, s3 = self.net.input_shape
            else:
                raise NotImplementedError

            lbs_init = self.lbs.view(1, s1, s2, s3)
            ubs_init = self.ubs.view(1, s1, s2, s3)
            input_vars = []
            for i in range(s1):
                s1_vars = []
                for j in range(s2):
                    s2_vars = []
                    for k in range(s3):
                        v = self.model.addVar(lb=lbs_init[0, i, j, k], 
                                              ub=ubs_init[0, i, j, k], 
                                              obj=0, 
                                              vtype=grb.GRB.CONTINUOUS, 
                                              name=f'x[{i*s2*s3 + j*s3 + k}]')
                        s2_vars.append(v)
                    s1_vars.append(s2_vars)
                input_vars.append(s1_vars)

        else:
            input_vars = [self.model.addVar(lb=self.lbs[i], 
                                            ub=self.ubs[i], 
                                            obj=0, 
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'x[{i}]')
                for i in range(self.net.n_input)]

        self.model.update()
        gurobi_vars.append(input_vars)

        relu_idx = 0
        for layer in self.net.layers:
            new_layer_gurobi_vars = []
            if isinstance(layer, nn.Linear):
                if layer == self.net.layers[-1]:
                    vs = [self.model.addVar(lb=self.lower_out_init[i], ub=self.upper_out_init[i], name=f'y[{i}]') for i in range(self.net.n_output)]
                    variables = [None] * len(vs)
                else:
                    variables = self.net.layers_mapping[relu_idx]
                    vs = [self.model.addVar(lb=self.mapping_bounds[i][0], ub=self.mapping_bounds[i][1], name=f'a[{i}]') for i in variables]

                exprs = [grb.LinExpr(layer.weight[i], gurobi_vars[-1]) + layer.bias[i] for i in range(len(vs))]
                self.model.update()
                for v, node, expr in zip(vs, variables, exprs):
                    self.model.addLConstr(v == expr)
                    if node is not None:
                        self.mapping_assignment[node] = expr

                new_layer_gurobi_vars = vs

            elif isinstance(layer, nn.ReLU):
                # This is convolutional relus
                if isinstance(gurobi_vars[-1][0], list):
                    print('cac', gurobi_vars[-1][0])
                    pass
                else:
                    for node, pre_var in zip(variables, gurobi_vars[-1]):
                        lb, ub = self.mapping_bounds[node]
                        if lb >= 0:
                            v = pre_var
                        elif ub <= 0:
                            v = zero_var
                        else:
                            v = self.model.addVar(ub=ub, lb=lb, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'n[{node}]')
                            if model_type == "mip" or model_type == "lp_integer":
                                if model_type == "mip":
                                    a = self.model.addVar(vtype=grb.GRB.BINARY, name=f'{model_type}_n[{node}]')
                                else:
                                    a = self.model.addVar(ub=1, lb=0, vtype=grb.GRB.CONTINUOUS, name=f'{model_type}_n[{node}]')

                                self.model.addLConstr(pre_var - lb * (1 - a) >= v)
                                self.model.addLConstr(v >= pre_var)
                                self.model.addLConstr(ub * a >= v)
                                self.model.addLConstr(v >= 0)

                            elif model_type == "lp":
                                self.model.addLConstr(v >= 0)
                                self.model.addLConstr(v >= pre_var)
                                self.model.addLConstr(ub * pre_var - (ub - lb) * v >= ub * lb)

                            else:
                                raise NotImplementedError

                        new_layer_gurobi_vars.append(v)
                relu_idx += 1

            elif isinstance(layer, nn.Flatten):
                if isinstance(gurobi_vars[-1][0], list):
                    # last layer is conv
                    raise
                else:
                    continue
            else:
                print(layer)
                raise NotImplementedError

            gurobi_vars.append(new_layer_gurobi_vars)
            self.model.update()
            # self.model.write('gurobi/example.lp')


        self.model.update()
        out_vars = gurobi_vars[-1]
        self.output_constraints = self.spec.get_output_property(out_vars)
        # print(out_vars)
        # print(self.output_constraints)


    def optimize(self):
        self.model.update()
        # self.model.write(f'gurobi/clause_shortener.lp')
        self.model.reset()
        self.model.optimize()

    def check_sat(self, assignment):
        self.optimize()
        print('Solving:', len(self.model.getConstrs()), 'constraints', GRB_STATUS_CODE[self.model.status])
        # exit()
        cc = []
        for node, status in assignment.items():
            # print(node, self.mapping_assignment[node])
            assert status is not None
            fv = self.model.getVarByName(f'n[{node}]')
            if status:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] >= 1e-8, name=f'c_b_{node}'))
                # cc.append(self.model.addLConstr(fv == self.mapping_assignment[node], name=f'c_f_{node}'))
            else:
                cc.append(self.model.addLConstr(self.mapping_assignment[node] <= 0, name=f'c_b_{node}'))
                # cc.append(self.model.addLConstr(fv == 0, name=f'c_f_{node}'))
        self.model.update()

        flag_sat = False
        for cnf in self.output_constraints:
            ci = [self.model.addLConstr(_) for _ in cnf]
            self.optimize()
            print('Solving:', len(self.model.getConstrs()), 'constraints', GRB_STATUS_CODE[self.model.status])
            if self.model.status == grb.GRB.OPTIMAL:
                flag_sat = True
                self.model.remove(ci)
                break

            self.model.remove(ci)
        self.model.remove(cc)

        return flag_sat


    # def shorten_conflict_clause(self, assignment):
    #     # print('shorten_conflict_clause:', len(self.model.getConstrs()), 'constraints')
    #     # print(len(assignment))

    #     cc = []
    #     for node, status in assignment.items():
    #         # print(node, self.mapping_assignment[node])
    #         fv = self.model.getVarByName(f'n_{node}')
    #         if status:
    #             cc.append(self.model.addLConstr(self.mapping_assignment[node] >= 1e-6, name=f'c_b_{node}'))
    #             cc.append(self.model.addLConstr(fv == self.mapping_assignment[node], name=f'c_f_{node}'))
    #         else:
    #             cc.append(self.model.addLConstr(self.mapping_assignment[node] <= 0, name=f'c_b_{node}'))
    #             cc.append(self.model.addLConstr(fv == 0, name=f'c_f_{node}'))
    #     self.model.update()

    #     conflict_clause = set()
    #     for cnf in self.output_constraints:
    #         ci = [self.model.addLConstr(_) for _ in cnf]
    #         self.optimize()
    #         # print('Solving:', len(self.model.getConstrs()), 'constraints', GRB_STATUS_CODE[self.model.status])
    #         if self.model.status == grb.GRB.OPTIMAL:
    #             self.model.remove(ci)
    #             self.model.remove(cc)
    #             return frozenset()
            
    #         if self.model.status == grb.GRB.INFEASIBLE:
    #             # print('Start computeIIS', GRB_STATUS_CODE[self.model.status])
    #             self.model.computeIIS()
    #             for node, value in assignment.items():
    #                 c = self.model.getConstrByName(f'c_f_{node}')
    #                 if c.IISConstr:
    #                     conflict_clause.add(-node if value else node)
    #                     continue
    #                 c = self.model.getConstrByName(f'c_b_{node}')
    #                 if c.IISConstr:
    #                     conflict_clause.add(-node if value else node)

    #         self.model.remove(ci)
    #     self.model.remove(cc)

    #     return frozenset(conflict_clause)
