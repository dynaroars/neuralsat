from pprint import pprint
import gurobipy as grb
import numpy as np
import torch
import time
import copy
import re
import os

from utils.read_nnet import ReLU, Linear
from abstract.reluval import reluval
from abstract.deepz import deepz
from abstract.eran import eran
import settings


class DNFConstraint:

    def __init__(self, constraints):
        self.constraints = constraints


class DNNTheoremProver:

    epsilon = 1e-5
    skip = 1e-4

    def __init__(self, dnn, layers_mapping, p=1):
        self.dnn = dnn
        self.layers_mapping = layers_mapping

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 16)

        try:
            self.gurobi_vars = [self.model.addVar(
                    name=f'x{i}', 
                    lb=(dnn.input_lower_bounds[i] - dnn.input_means[i]) / dnn.input_ranges[i], 
                    ub=(dnn.input_upper_bounds[i] - dnn.input_means[i]) / dnn.input_ranges[i])
                for i in range(self.n_inputs)]
        except AttributeError:
            self.gurobi_vars = [self.model.addVar(
                    name=f'x{i}', 
                    lb=-grb.GRB.INFINITY, 
                    ub=grb.GRB.INFINITY) 
                for i in range(self.n_inputs)]


        self.model.setObjective(0, grb.GRB.MAXIMIZE)
        
        self.p = p # property

        self.restore_input_bounds(intial=True)
        
        self.count = 0 # debug

        self.solution = None
        self.constraints = []


        if settings.HEURISTIC_DEEPPOLY:
            self.deeppoly = eran.ERAN(self.dnn.path[:-4] + 'onnx', 'deeppoly')


        # clean trash
        os.system('rm -rf gurobi/*')


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
                # print('lower:', lbs)
                # print('upper:', ubs)
                # self.model.write(f'gurobi/debug_{self.count}.lp')
                # raise
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
                output_constraint = self.get_output_property_torch(output)
            else:
                if type(layer) is Linear:
                    output = layer.weight.mm(inputs)
                    output[:, -1] += layer.bias

                elif type(layer) is ReLU:
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

        self._optimize()
        if self.model.status == grb.GRB.INFEASIBLE:
            self.restore_input_bounds()
            return False, None

        if settings.TIGHTEN_BOUND: # compute new input lower/upper bounds

            if settings.DEBUG:
                print('[+] TIGHTEN_BOUND ')

            # upper
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MAXIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                ubs = [var.X for var in self.gurobi_vars]
                # TODO
                # self.constraints += [self.model.addConstr(var <= ubs[i]) for i, var in enumerate(self.gurobi_vars)]
            else:
                # print(self.model.status)
                ubs = None

            # lower
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MINIMIZE)
            self._optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                lbs = [var.X for var in self.gurobi_vars]
                # TODO
                # self.constraints += [self.model.addConstr(var >= lbs[i]) for i, var in enumerate(self.gurobi_vars)]
            else:
                # print(self.model.status)
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

                # self.restore_input_bounds()
                if not self.check_output_reachability(lower, upper): # conflict
                    self.restore_input_bounds()
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
                        self.restore_input_bounds()
                        return False, None
                        raise

            if settings.HEURISTIC_DEEPPOLY: 
                lbs = torch.Tensor([var.lb for var in self.gurobi_vars])
                ubs = torch.Tensor([var.ub for var in self.gurobi_vars])
                
                # eran deeppoly
                lower, upper = self.deeppoly(lbs, ubs)

                if settings.DEBUG:
                    print('[+] HEURISTIC_DEEPPOLY input')
                    print('\t- lower:', lbs.data)
                    print('\t- upper:', ubs.data)

                    print('[+] HEURISTIC_DEEPPOLY output')
                    print('\t- lower:', lower)
                    print('\t- upper:', upper)

                # self.restore_input_bounds()
                if not self.check_output_reachability(lower, upper): # conflict
                    self.restore_input_bounds()
                    return False, None

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

        # debug
        if settings.DEBUG:
            self.model.write(f'gurobi/{self.count}.lp')

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
                    self.restore_input_bounds()
                    return False, None

            else:
                if type(output_constraint) is list:
                    self.constraints += [self.model.addConstr(_) for _ in output_constraint]
                else:
                    self.constraints.append(self.model.addConstr(output_constraint))

                self._optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    self.restore_input_bounds()
                    return False, None

                self.solution = self.get_solution()

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


    def restore_input_bounds(self, intial=False):

        properties = {
            0: { # debug
                'lbs' : [-30, -30, -30],
                'ubs' : [30, 30, 30],
            },
            1: {
                'lbs' : [55947.691, -3.141593,  -3.141593, 1145,  0],
                'ubs' : [60760,      3.141593,   3.141593, 1200, 60],
            }, 
            2: {
                'lbs' : [55947.691, -3.141593, -3.141593, 1145,  0],
                'ubs' : [60760,      3.141593,  3.141593, 1200, 60],
            },  
            3: {
                'lbs' : [1500, -0.06, 3.1,                980,  960],
                'ubs' : [1800,  0.06, 3.141592653589793, 1200, 1200],
            },  
            4: {
                'lbs' : [1500, -0.06, 0, 1000, 700],
                'ubs' : [1800,  0.06, 0, 1200, 800],
            }, 
            5: {
                'lbs' : [250, 0.2, -3.141592, 100,   0],
                'ubs' : [400, 0.4, -3.136592, 400, 400],
            }, 
            7: {
                'lbs' : [0,     -3.141592, -3.141592,  100,    0],
                'ubs' : [60760,  3.141592,  3.141592, 1200, 1200],
            },
            8: {
                'lbs' : [    0, -3.141592, -0.1,  600,  600],
                'ubs' : [60760, -2.356194,  0.1, 1200, 1200],
            },
            9: {
                'lbs' : [2000, -0.40, -3.141592, 100,   0],
                'ubs' : [7000, -0.14, -3.131592, 150, 150],
            },
            10: {
                'lbs' : [36000, 0.7,     -3.141592,  900,  600],
                'ubs' : [60760, 3.141592,-3.131592, 1200, 1200],
            },
        }

        lbs = properties[self.p]['lbs']
        ubs = properties[self.p]['ubs']
        if intial:
            for i, var in enumerate(self.gurobi_vars):
                self.model.addConstr(var >= (lbs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i])
                self.model.addConstr(var <= (ubs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i])
        else:
            for i, var in enumerate(self.gurobi_vars):
                var.lb = (lbs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i]
                var.ub = (ubs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i]
        self.model.update()


    def get_output_property_torch(self, output):
        if self.p == 0:
            return self._get_equation(output[0]) >= 1e-6

        if self.p == 1:
            return self._get_equation(output[0]) >= (1500 - self.dnn.output_mean) / self.dnn.output_range

        if self.p == 2:
            return [self._get_equation(output[0]) >= self._get_equation(output[1]),
                    self._get_equation(output[0]) >= self._get_equation(output[2]),
                    self._get_equation(output[0]) >= self._get_equation(output[3]),
                    self._get_equation(output[0]) >= self._get_equation(output[4])]

        if self.p == 3 or self.p == 4:
            return [self._get_equation(output[0]) <= self._get_equation(output[1]),
                    self._get_equation(output[0]) <= self._get_equation(output[2]),
                    self._get_equation(output[0]) <= self._get_equation(output[3]),
                    self._get_equation(output[0]) <= self._get_equation(output[4])]

        if self.p == 5:
            return DNFConstraint([ # or
                [self._get_equation(output[0]) <= self._get_equation(output[4])], # and
                [self._get_equation(output[1]) <= self._get_equation(output[4])], # and
                [self._get_equation(output[2]) <= self._get_equation(output[4])], # and
                [self._get_equation(output[3]) <= self._get_equation(output[4])], # and
            ]) 

        if self.p == 7:
            return DNFConstraint([ # or
                [ # and
                    self._get_equation(output[3]) <= self._get_equation(output[0]),
                    self._get_equation(output[3]) <= self._get_equation(output[1]),
                    self._get_equation(output[3]) <= self._get_equation(output[2]),
                ], 
                [ # and
                    self._get_equation(output[4]) <= self._get_equation(output[0]),
                    self._get_equation(output[4]) <= self._get_equation(output[1]),
                    self._get_equation(output[4]) <= self._get_equation(output[2]),

                ], 
            ])

        if self.p == 8:
            return DNFConstraint([ # or
                [ # and
                    self._get_equation(output[2]) <= self._get_equation(output[0]),
                    self._get_equation(output[2]) <= self._get_equation(output[1]),
                ], 
                [ # and
                    self._get_equation(output[3]) <= self._get_equation(output[0]),
                    self._get_equation(output[3]) <= self._get_equation(output[1]),
                ], 
                [ # and
                    self._get_equation(output[4]) <= self._get_equation(output[0]),
                    self._get_equation(output[4]) <= self._get_equation(output[1]),
                ], 
            ])

        if self.p == 9:
            return DNFConstraint([ # or
                [self._get_equation(output[0]) <= self._get_equation(output[3])], 
                [self._get_equation(output[1]) <= self._get_equation(output[3])], 
                [self._get_equation(output[2]) <= self._get_equation(output[3])], 
                [self._get_equation(output[4]) <= self._get_equation(output[3])], 
            ]) 


        if self.p == 10:
            return DNFConstraint([ # or
                [self._get_equation(output[1]) <= self._get_equation(output[0])], 
                [self._get_equation(output[2]) <= self._get_equation(output[0])], 
                [self._get_equation(output[3]) <= self._get_equation(output[0])], 
                [self._get_equation(output[4]) <= self._get_equation(output[0])], 
            ]) 


        raise NotImplementedError


    def check_output_reachability(self, lbs, ubs):
        if self.p == 0:
            return ubs[0] >= 1e-6

        if self.p == 1:
            return ubs[0] >= (1500 - self.dnn.output_mean) / self.dnn.output_range

        if self.p == 2:
            return all([ubs[0] >= lbs[1],
                        ubs[0] >= lbs[2],
                        ubs[0] >= lbs[3],
                        ubs[0] >= lbs[4]])
            
        if self.p == 3 or self.p == 4:
            return all([ubs[1] >= lbs[0],
                        ubs[2] >= lbs[0],
                        ubs[3] >= lbs[0],
                        ubs[4] >= lbs[0]])

        if self.p == 5:
            return any([ubs[4] >= lbs[0],
                        ubs[4] >= lbs[1],
                        ubs[4] >= lbs[2],
                        ubs[4] >= lbs[3]])

        if self.p == 7:
            return any([ # or
                all([ubs[0] >= lbs[3], # and
                     ubs[1] >= lbs[3],
                     ubs[2] >= lbs[3]]
                ), 
                all([ubs[0] >= lbs[4], # and
                     ubs[1] >= lbs[4],
                     ubs[2] >= lbs[4]]
                ),
            ])

        if self.p == 8:
            return any([ # or
                all([ubs[0] >= lbs[2], # and
                     ubs[1] >= lbs[2]]
                ), 
                all([ubs[0] >= lbs[3], # and
                     ubs[1] >= lbs[3]]
                ), 
                all([ubs[0] >= lbs[4], # and
                     ubs[1] >= lbs[4]]
                ),
            ])

        if self.p == 9:
            return any([ubs[3] >= lbs[0],
                        ubs[3] >= lbs[1],
                        ubs[3] >= lbs[2],
                        ubs[3] >= lbs[4]])

        if self.p == 10:
            return any([ubs[0] >= lbs[1],
                        ubs[0] >= lbs[2],
                        ubs[0] >= lbs[3],
                        ubs[0] >= lbs[4]])

        raise NotImplementedError

