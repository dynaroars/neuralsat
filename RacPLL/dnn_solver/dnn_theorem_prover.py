from pprint import pprint
import gurobipy as grb
import torch.nn as nn
import numpy as np
import torch
import time
import copy
import re
import os

from dnn_solver.utils import DNFConstraint
# from abstract.reluval import reluval
from abstract.deepz import deepz
# from abstract.eran import eran
import settings


class DNNTheoremProver:

    epsilon = 1e-5
    skip = 1e-4

    def __init__(self, dnn, layers_mapping, spec):
        self.dnn = dnn
        self.layers_mapping = layers_mapping
        self.spec = spec

        print(self.layers_mapping)

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 16)

        # bounds = self.get_intial_input_bounds()
        self.lbs_init = torch.Tensor(spec.lower)
        self.ubs_init = torch.Tensor(spec.upper)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
            for i in range(self.n_inputs)
        ]

        self.model.setObjective(0, grb.GRB.MAXIMIZE)
        
        self.count = 0 # debug

        self.solution = None
        self.constraints = []

        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = eran.ERAN(self.dnn.path[:-4] + 'onnx', 'deeppoly')

        # clean trash
        os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)


    @property
    def n_outputs(self):
        return self.dnn.output_shape[1]

    @property
    def n_inputs(self):
        return self.dnn.input_shape[1]


    def update_input_bounds(self, lbs, ubs):
        if lbs is None or ubs is None:
            return True
        lbs = np.array(lbs)
        ubs = np.array(ubs)
        flag = lbs > ubs

        if np.any(flag):
            if not np.all(lbs[flag] - ubs[flag] < DNNTheoremProver.skip):
                # self.model.write(f'gurobi/debug_{self.count}.lp')
                return False

        for i, var in enumerate(self.gurobi_vars):
            if abs(lbs[i] - ubs[i]) < DNNTheoremProver.skip: # concretize
                var.lb = lbs[i]
                var.ub = lbs[i]
                continue
            if (abs(var.lb - lbs[i]) > DNNTheoremProver.skip):
                var.lb = lbs[i]
            if (abs(var.ub - ubs[i]) > DNNTheoremProver.skip):
                var.ub = ubs[i]

        self.model.update()
        return True

    def _find_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes

    def _get_equation(self, coeffs):
        return sum([coeffs[i] * self.gurobi_vars[i] for i in range(len(self.gurobi_vars))]) + coeffs[-1]

    def __call__(self, assignment):

        # debug
        self.count += 1

        # reset constraints
        self.model.remove(self.constraints)
        self.constraints = []

        imply_nodes = self._find_nodes(assignment)
        return_output = True if imply_nodes is None else False

        substitute_dict_torch = {}

        inputs = torch.hstack([torch.eye(self.n_inputs), torch.zeros(self.n_inputs, 1)])
        layer_id = 0
        variables = self.layers_mapping.get(layer_id, None)
        flag_break = False
        for layer in self.dnn.layers:
            if variables is None: # output layer
                output = layer.weight.mm(inputs)
                output[:, -1] += layer.bias
                output_constraint = self.spec.get_output_property(
                    [self._get_equation(output[i]) for i in range(self.n_outputs)]
                )
            else:
                if isinstance(layer, nn.Linear):
                    output = layer.weight.mm(inputs)
                    output[:, -1] += layer.bias

                elif isinstance(layer, nn.ReLU):
                    inputs = torch.zeros(output.shape)
                    for i, v in enumerate(variables):
                        status = assignment.get(v, None)
                        if status is None: # unassigned node
                            flag_break = True
                        elif status:
                            inputs[i] = output[i]
                        else:
                            # inputs[i] = zero_torch
                            pass
                        substitute_dict_torch[v] = self._get_equation(output[i])

                    layer_id += 1
                    variables = self.layers_mapping.get(layer_id, None)
                else:
                    raise NotImplementedError

                if flag_break and (not return_output):
                    break

        for node in substitute_dict_torch:
            status = assignment.get(node, None)
            if status is None:
                continue
            if status:
                self.constraints.append(self.model.addConstr(substitute_dict_torch[node] >= DNNTheoremProver.epsilon))
            else:
                self.constraints.append(self.model.addConstr(substitute_dict_torch[node] <= 0))

        # debug
        if settings.DEBUG:
            self.model.write(f'gurobi/{self.count}.lp')


        self._optimize()
        if self.model.status == grb.GRB.INFEASIBLE:
            return False, None

        # output
        if return_output:
            if type(output_constraint) is DNFConstraint:
                flag_sat = False
                for cnf in output_constraint.constraints:
                    ci = [self.model.addConstr(_) for _ in cnf]
                    self._optimize()
                    if self.model.status == grb.GRB.OPTIMAL:
                        flag_sat = True
                    self.model.remove(ci)
                    if flag_sat:
                        break
                if flag_sat:
                    self.solution = self.get_solution()
                else:
                    return False, None

            else:
                if type(output_constraint) is list:
                    self.constraints += [self.model.addConstr(_) for _ in output_constraint]
                else:
                    self.constraints.append(self.model.addConstr(output_constraint))

                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    return False, None

                self.solution = self.get_solution()

            return True, {}


        if settings.TIGHTEN_BOUND: # compute new input lower/upper bounds

            if settings.DEBUG:
                print('[+] TIGHTEN_BOUND ')

            # upper
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MAXIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                ubs = [var.X for var in self.gurobi_vars]
            else:
                ubs = None

            # lower
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MINIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                lbs = [var.X for var in self.gurobi_vars]
            else:
                lbs = None
                
            if not self.update_input_bounds(lbs, ubs): # conflict
                self.restore_input_bounds()
                return False, None

            # reset objective
            self.model.setObjective(0, grb.GRB.MAXIMIZE)

            if settings.HEURISTIC_DEEPZONO: 
                lbs = torch.Tensor([var.lb for var in self.gurobi_vars])
                ubs = torch.Tensor([var.ub for var in self.gurobi_vars])
                
                # eran deepzono
                (lower, upper), hidden_bounds = deepz.forward(self.dnn, lbs, ubs)

                # TODO: reluval
                # lower2, upper2 = reluval.forward_nnet(self.dnn, lbs, ubs)

                if settings.DEBUG:
                    print('[+] HEURISTIC_DEEPZONO input')
                    print('\t- lower:', lbs.data)
                    print('\t- upper:', ubs.data)

                    print('[+] HEURISTIC_DEEPZONO output')
                    print('\t- lower:', lower)
                    print('\t- upper:', upper)

                self.restore_input_bounds()
                if not self.spec.check_output_reachability(lower, upper): # conflict
                    return False, None

                signs = {}
                for idx, (lb, ub) in enumerate(hidden_bounds):
                    sign = 2 * torch.ones(len(lb), dtype=int) 
                    sign[lb >= 0] = 1
                    sign[ub == 0] = -1
                    signs.update(dict(zip(self.layers_mapping[idx], sign.numpy())))

                for node, status in assignment.items():
                    if signs[node] == 2:
                        continue
                    abt_status = signs[node] == 1
                    if abt_status != status:
                        return False, None
                        raise

            self.restore_input_bounds()
        # imply next hidden nodes
        implications = {}
        if imply_nodes:
            for node in imply_nodes:
                implications[node] = {'pos': False, 'neg': False}
                # neg
                ci = self.model.addConstr(substitute_dict_torch[node] >= DNNTheoremProver.epsilon)
                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['neg'] = True
                    self.model.remove(ci)
                    continue
                self.model.remove(ci)

                # pos
                ci = self.model.addConstr(substitute_dict_torch[node] <= 0)
                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['pos'] = True
                    self.model.remove(ci)
                else:
                    self.model.remove(ci)

        if settings.TIGHTEN_BOUND and settings.HEURISTIC_DEEPZONO:
            for node, value in signs.items():
                if node in implications or node in assignment:
                    continue
                if node != 2:
                    implications[node] = {'pos': value==1, 'neg': value==-1}


        return True, implications

    def _optimize(self):
        self.model.update()
        self.model.reset()
        self.model.optimize()


    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self._optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.Tensor([var.X for var in self.gurobi_vars])
        return None


    def restore_input_bounds(self):
        for i, var in enumerate(self.gurobi_vars):
            var.lb = self.lbs_init[i]
            var.ub = self.ubs_init[i]
        self.model.update()
