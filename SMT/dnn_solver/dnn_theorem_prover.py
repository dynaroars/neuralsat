from pprint import pprint
import gurobipy as grb
import numpy as np
import torch
import time
import copy
import re
import os

from utils.read_nnet import NetworkDeepZono, ReLU, Linear
from abstract.deepz import deepz
import settings

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
        self.model.update()
        
        self.p = p

        self.restore_input_bounds(intial=True)

        # self.dnn = NetworkDeepZono(dnn.path)
        
        self.count = 0

        self.solution = None

        # clean trash
        os.system('rm -rf gurobi/*')


    @property
    def n_outputs(self):
        return self.dnn.output_shape[1]

    @property
    def n_inputs(self):
        return self.dnn.input_shape[1]

    # @property
    # def input_symbols(self):
    #     return np.eye(self.n_inputs)

    # @property
    # def output_symbols(self):
    #     return [f"y{n}" for n in range(self.n_outputs)]

    def update_input_bounds(self, lbs, ubs):
        if lbs is None or ubs is None:
            return True

        if np.any(np.array(lbs) > np.array(ubs)):
            return False

        for i, var in enumerate(self.gurobi_vars):
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

        self.count += 1
        imply_nodes = self._find_nodes(assignment)
        return_output = True if imply_nodes is None else False
        # imply_nodes = copy.deepcopy(nodes)
        # zero = np.zeros(self.n_inputs+1)
        # zero_torch = torch.zeros(self.n_inputs+1)
        constraints = []

        # prev_nodes = np.concatenate([self.input_symbols, np.zeros([1, self.n_inputs])]).T
        # # print(prev_nodes)

        # substitute_dict = {}

        # print('\n===================== substitute_dict ===================== ')

        # for layer_id, layer in enumerate(self.dnn.layers):
        #     weight, bias = layer.get_weights()
        #     variables = self.layers_mapping.get(layer_id, None)

        #     # print('- layer:', layer_id)
        #     # print('- variables:', variables)
        #     # print('- input:', prev_nodes)
        #     # print('- weight:', weight.shape, weight)
        #     # print('- bias:', bias.shape, bias)

        #     if variables is None: # output layer
        #         output = np.array(prev_nodes).T.dot(weight)
        #         output[-1] += bias
        #         output_constraint = self.get_output_property(output)
        #         print('------------- output:', output_constraint)
        #     else:
        #         output = np.array(prev_nodes).T.dot(weight)
        #         output[-1] += bias
        #         prev_nodes = []
        #         for i, v in enumerate(variables):
        #             status = assignment.get(v, None)
        #             if status is None:
        #                 nodes.remove(v)
        #             elif status:
        #                 prev_nodes.append(output[:, i])
        #             else:
        #                 prev_nodes.append(zero)
        #             substitute_dict[v] = self._get_equation(output[:, i])

        #         if (not nodes) and (not return_output):
        #             break

        # # pprint(substitute_dict)
        # # print('\n==============================================\n\n ')



        # print('\n===================== substitute_dict_torch ===================== ')
        substitute_dict_torch = {}
        # print('nodes', nodes)

        inputs = torch.hstack([torch.eye(self.n_inputs), torch.zeros(self.n_inputs, 1)])
        layer_id = 0
        variables = self.layers_mapping.get(layer_id, None)
        flag_break = False
        for layer in self.dnn.layers:
            # print('layer:', layer_id, 'variables:', variables)

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
                            # nodes.remove(v)
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

        for node, status in assignment.items():
            if node not in substitute_dict_torch:
                print('[!] Missing node:', node, 'status:', status)
                continue
            if status:
                if type(substitute_dict_torch[node]) == np.float64:
                    if substitute_dict_torch[node] > 0:
                        continue
                    else:
                        self.model.remove(constraints)
                        self.model.update()
                        return False, None 
                c = self.model.addConstr(substitute_dict_torch[node] >= DNNTheoremProver.epsilon)
            else:
                if type(substitute_dict_torch[node]) == np.float64:
                    if substitute_dict_torch[node] == 0:
                        continue
                    else:
                        self.model.remove(constraints)
                        self.model.update()
                        return False, None 

                c = self.model.addConstr(substitute_dict_torch[node] <= 0)
            constraints.append(c)



        # for node, status in assignment.items():
        #     if node not in substitute_dict:
        #         print('[!] Missing node:', node, 'status:', status)
        #         continue
        #     if status:
        #         if type(substitute_dict[node]) == np.float64:
        #             if substitute_dict[node] > 0:
        #                 continue
        #             else:
        #                 self.model.remove(constraints)
        #                 self.model.update()
        #                 return False, None 
        #         c = self.model.addConstr(substitute_dict[node] >= DNNTheoremProver.epsilon)
        #     else:
        #         if type(substitute_dict[node]) == np.float64:
        #             if substitute_dict[node] == 0:
        #                 continue
        #             else:
        #                 self.model.remove(constraints)
        #                 self.model.update()
        #                 return False, None 

        #         c = self.model.addConstr(substitute_dict[node] <= 0)
        #     constraints.append(c)

        self.model.update()
        self.model.reset()
        self.model.optimize()

        if self.model.status == grb.GRB.INFEASIBLE:
            self.model.remove(constraints)
            self.model.update()
            return False, None

        if settings.TIGHTEN_BOUND: # and self.count % settings.TIGHTEN_BOUND_INTERVAL == 0:

            if settings.DEBUG:
                print('[+] TIGHTEN_BOUND ')
            # compute new input lower bounds
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MINIMIZE)
            self.model.reset()
            self.model.optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                lbs = [var.X for var in self.gurobi_vars]
            else:
                lbs = None

            # compute new input upper bounds
            self.model.setObjective(sum(self.gurobi_vars), grb.GRB.MAXIMIZE)
            self.model.reset()
            self.model.optimize()
            if self.model.status == grb.GRB.OPTIMAL:
                ubs = [var.X for var in self.gurobi_vars]
            else:
                ubs = None
                
            if not self.update_input_bounds(lbs, ubs):
                # conflict
                self.model.remove(constraints)
                self.model.update()
                self.restore_input_bounds()
                return False, None


            # reset objective
            self.model.setObjective(0, grb.GRB.MAXIMIZE)
            self.model.update()

            if settings.HEURISTIC_DEEPZONO: 
                lbs = torch.Tensor([var.lb for var in self.gurobi_vars])
                ubs = torch.Tensor([var.ub for var in self.gurobi_vars])

                center, error = deepz.forward_nnet(self.dnn, lbs, ubs)
                error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
                upper = center + error_apt
                lower = center - error_apt
                self.restore_input_bounds()
                    
                if settings.DEBUG:
                    print('[+] HEURISTIC_DEEPZONO input')
                    print('\t- lower:', lbs.data)
                    print('\t- upper:', ubs.data)

                    print('[+] HEURISTIC_DEEPZONO output')
                    print('\t- lower:', lower.data)
                    print('\t- upper:', upper.data)

                if not self.check_deep_zono(lower, upper):
                    # conflict
                    self.model.remove(constraints)
                    self.model.update()
                    return False, None

        # print('substitute_dict_torch', list(substitute_dict_torch.keys()))
        # print()
        # implications
        implications = {}
        if imply_nodes:
            for node in imply_nodes:
                implications[node] = {'pos': False, 'neg': False}
                # neg
                if type(substitute_dict_torch[node]) == np.float64:
                    if substitute_dict_torch[node] == 0:
                        implications[node]['neg'] = True
                    else:
                        implications[node]['pos'] = True
                    continue

                ci = self.model.addConstr(substitute_dict_torch[node] >= DNNTheoremProver.epsilon)
                self.model.update()
                self.model.reset()
                self.model.optimize()

                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['neg'] = True
                    self.model.remove(ci)
                    continue
                self.model.remove(ci)
                # pos
                ci = self.model.addConstr(substitute_dict_torch[node] <= 0)
                self.model.update()
                self.model.reset()
                self.model.optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implications[node]['pos'] = True
                    self.model.remove(ci)
                else:
                    self.model.remove(ci)

        # debug
        if settings.DEBUG:
            self.model.write(f'gurobi/{self.count}.lp')

        # output
        if return_output:

            if type(output_constraint) is list:
                output_constraint_tmp = []
                for _ in output_constraint:
                    if type(_) is np.bool_:
                        if _:
                            continue
                        else:
                            self.model.remove(constraints)
                            self.model.update()
                            return False, None
                    else:
                        output_constraint_tmp.append(_)

                if len(output_constraint_tmp):
                    co = [self.model.addConstr(_) for _ in output_constraint_tmp]
                else:
                    self.solution = self.get_solution()
                    self.model.remove(constraints)
                    self.model.update()
                    # if self.check_solution():
                    #     pass
                    return True, []

            elif type(output_constraint) is np.bool_:
                if output_constraint:
                    self.solution = self.get_solution()
                self.model.remove(constraints)
                self.model.update()
                # if self.solution is not None and self.check_solution():
                #     pass
                return output_constraint, []
            else:
                co = self.model.addConstr(output_constraint)

            self.model.update()
            self.model.reset()
            self.model.optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                self.model.remove(constraints)
                self.model.remove(co)
                self.model.update()
                return False, None
            self.solution = self.get_solution()
            self.model.remove(co)

        self.model.remove(constraints)
        self.model.update()
        # if self.solution is not None and self.check_solution():
        #     pass
        return True, implications

    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self.model.reset()
            self.model.optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.Tensor([var.X for var in self.gurobi_vars])
        return None

    def check_solution(self):
        output = torch.zeros(self.n_outputs, self.n_inputs+1)
        dnn_output = self.dnn(self.solution)
        output[:, -1] = dnn_output
        print('check_solution', self.n_outputs, self.n_inputs+1, dnn_output)
        print(self.get_output_property_torch(output))

        return True

    def restore_input_bounds(self, intial=False):

        properties = {
            0: { # debug
                'lbs': [-30, -30, -30],
                'ubs' : [30, 30, 30],
            },
            1: {
                'lbs': [55947.691, -3.141593, -3.141593, 1145, 0],
                'ubs' : [60760, 3.141593, 3.141593, 1200, 60],
            }, 
            2: {
                'lbs': [55947.691, -3.141593, -3.141593, 1145, 0],
                'ubs' : [60760, 3.141593, 3.141593, 1200, 60],
            },  
            3: {
                'lbs': [1500, -0.06, 3.1, 980, 960],
                'ubs' : [1800, 0.06, 3.141592653589793, 1200, 1200],
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


    # def get_output_property(self, output):
    #     properties = {
    #         0: self._get_equation(output[:, 0]) >= 0,
    #         1: self._get_equation(output[:, 0]) >= (1500 - self.dnn.output_mean) / self.dnn.output_range,
    #         2: [
    #             self._get_equation(output[:, 0]) >= self._get_equation(output[:, 1]),
    #             self._get_equation(output[:, 0]) >= self._get_equation(output[:, 2]),
    #             self._get_equation(output[:, 0]) >= self._get_equation(output[:, 3]),
    #             self._get_equation(output[:, 0]) >= self._get_equation(output[:, 4])
    #         ],
    #         3: [
    #             self._get_equation(output[:, 0]) <= self._get_equation(output[:, 1]),
    #             self._get_equation(output[:, 0]) <= self._get_equation(output[:, 2]),
    #             self._get_equation(output[:, 0]) <= self._get_equation(output[:, 3]),
    #             self._get_equation(output[:, 0]) <= self._get_equation(output[:, 4])

    #         ],
    #     }

    #     return properties[self.p]

    def get_output_property_torch(self, output):
        properties = {
            0: self._get_equation(output[0]) >= -1e-6,
            1: self._get_equation(output[0]) >= (1500 - self.dnn.output_mean) / self.dnn.output_range,
            2: [
                self._get_equation(output[0]) >= self._get_equation(output[1]),
                self._get_equation(output[0]) >= self._get_equation(output[2]),
                self._get_equation(output[0]) >= self._get_equation(output[3]),
                self._get_equation(output[0]) >= self._get_equation(output[4])
            ],
            3: [
                self._get_equation(output[0]) <= self._get_equation(output[1]),
                self._get_equation(output[0]) <= self._get_equation(output[2]),
                self._get_equation(output[0]) <= self._get_equation(output[3]),
                self._get_equation(output[0]) <= self._get_equation(output[4])

            ],
        }

        return properties[self.p]


    def check_deep_zono(self, lbs, ubs):
        properties = {
            1: ubs[0][0] >= (1500 - self.dnn.output_mean) / self.dnn.output_range,
            2: all([
                    ubs[0][0] >= lbs[0][1],
                    ubs[0][0] >= lbs[0][2],
                    ubs[0][0] >= lbs[0][3],
                    ubs[0][0] >= lbs[0][4],
                ]),
            3: all([
                    ubs[0][1] >= lbs[0][0],
                    ubs[0][2] >= lbs[0][0],
                    ubs[0][3] >= lbs[0][0],
                    ubs[0][4] >= lbs[0][0],
                ]),
        }
        return properties[self.p]

