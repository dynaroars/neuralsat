import gurobipy as grb
import numpy as np
import torch
import time
import copy
import re
import os

from utils.read_nnet import NetworkDeepZono
from abstract.deepz import deepz
import settings

class DNNConstraintGurobi:

    epsilon = 1e-5
    skip = 1e-4

    def __init__(self, dnn, layers_mapping):
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
        
        self.p = 1

        self.init_input_property()

        # hardcode input onditions
        # self.model.addConstr(self.gurobi_vars[0] - self.gurobi_vars[1] >= DNNConstraintGurobi.epsilon)
        # self.model.addConstr(self.gurobi_vars[0] + self.gurobi_vars[1] <= -DNNConstraintGurobi.epsilon)
        # self.model.addConstr(self.gurobi_vars[0] <= 0)

        if settings.HEURISTIC_DEEPZONO:
            self.dnn_deepzono = NetworkDeepZono(dnn.path)

        
        self.count = 0

        # clean trash
        os.system('rm -rf gurobi/*')


    @property
    def n_outputs(self):
        return self.dnn.output_shape[1]

    @property
    def n_inputs(self):
        return self.dnn.input_shape[1]

    @property
    def input_symbols(self):
        return np.eye(self.n_inputs)

    @property
    def output_symbols(self):
        return [f"y{n}" for n in range(self.n_outputs)]

    def update_input_bounds(self, lbs, ubs):
        if lbs is None or ubs is None:
            return True

        if np.any(np.array(lbs) > np.array(ubs)):
            return False

        for i, var in enumerate(self.gurobi_vars):
            # if lbs is not None:
            if (abs(var.lb - lbs[i]) > DNNConstraintGurobi.skip):
                var.lb = lbs[i]
                # print('updated lb:', lbs[i])
            # if ubs is not None:
            if (abs(var.ub - ubs[i]) > DNNConstraintGurobi.skip):
                var.ub = ubs[i]
                # print('updated ub:', ubs[i])

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
        return sum([coeffs[i] * self.gurobi_vars[i] 
            for i in np.nonzero(coeffs[:-1])[0]]) + coeffs[-1]


    def __call__(self, assignment):

        self.count += 1
        nodes = self._find_nodes(assignment)
        return_output = True if nodes is None else False
        imply_nodes = copy.deepcopy(nodes)
        zero = np.zeros(self.n_inputs+1)
        constraints = []

        prev_nodes = np.concatenate([self.input_symbols, np.zeros([1, self.n_inputs])]).T
        # print(prev_nodes)

        substitute_dict = {}

        for layer_id, layer in enumerate(self.dnn.layers):
            weight, bias = layer.get_weights()
            variables = self.layers_mapping.get(layer_id, None)

            if variables is None: # output layer
                output = np.array(prev_nodes).T.dot(weight)
                output[-1] += bias

                # hardcode output onditions
                # output_constraint = self._get_equation(output[:, 0]) - self._get_equation(output[:, 1]) <= -DNNConstraintGurobi.epsilon
                output_constraint = self.get_output_property(output)
                # print('output_constraint:', output_constraint)
                
            else:
                before = np.array(prev_nodes).T.dot(weight)
                before[-1] += bias
                prev_nodes = []
                for i, v in enumerate(variables):
                    status = assignment.get(v, None)
                    if status is None:
                        nodes.remove(v)
                    elif status:
                        prev_nodes.append(before[:, i])
                    else:
                        prev_nodes.append(zero)
                    substitute_dict[v] = self._get_equation(before[:, i])

                if (not nodes) and (not return_output):
                    break

        for node, status in assignment.items():
            if status:
                if type(substitute_dict[node]) == np.float64:
                    if substitute_dict[node] > 0:
                        continue
                    else:
                        self.model.remove(constraints)
                        self.model.update()
                        self.restore_input_bounds()
                        return False, None 
                c = self.model.addConstr(substitute_dict[node] >= DNNConstraintGurobi.epsilon)
            else:
                if type(substitute_dict[node]) == np.float64:
                    if substitute_dict[node] == 0:
                        continue
                    else:
                        self.model.remove(constraints)
                        self.model.update()
                        self.restore_input_bounds()
                        return False, None 

                c = self.model.addConstr(substitute_dict[node] <= 0)
            constraints.append(c)

        self.model.update()
        self.model.reset()
        self.model.optimize()

        if self.model.status == grb.GRB.INFEASIBLE:
            self.model.remove(constraints)
            self.model.update()
            self.restore_input_bounds()
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

                center, error = deepz.forward_nnet(self.dnn_deepzono, lbs, ubs)
                error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
                upper = center + error_apt
                lower = center - error_apt
                    
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
                    self.restore_input_bounds()
                    return False, None

        # implies
        implies = {}
        if imply_nodes:
            for node in imply_nodes:
                implies[node] = {'pos': False, 'neg': False}
                # neg
                if type(substitute_dict[node]) == np.float64:
                    if substitute_dict[node] == 0:
                        implies[node]['neg'] = True
                    else:
                        implies[node]['pos'] = True
                    continue

                ci = self.model.addConstr(substitute_dict[node] >= DNNConstraintGurobi.epsilon)
                self.model.update()
                self.model.reset()
                self.model.optimize()

                if self.model.status == grb.GRB.INFEASIBLE:
                    implies[node]['neg'] = True
                    self.model.remove(ci)
                    continue
                self.model.remove(ci)
                # pos
                ci = self.model.addConstr(substitute_dict[node] <= 0)
                self.model.update()
                self.model.reset()
                self.model.optimize()
                if self.model.status == grb.GRB.INFEASIBLE:
                    implies[node]['pos'] = True
                    self.model.remove(ci)
                else:
                    self.model.remove(ci)

        # debug
        if settings.DEBUG:
            self.model.write(f'gurobi/{self.count}.lp')

        # output
        if return_output:
            if type(output_constraint) is np.bool_:
                self.model.remove(constraints)
                self.model.update()
                return output_constraint, []

            co = self.model.addConstr(output_constraint)
            self.model.update()
            self.model.reset()
            self.model.optimize()
            if self.model.status == grb.GRB.INFEASIBLE:
                self.model.remove(constraints)
                self.model.remove(co)
                self.model.update()
                self.restore_input_bounds()
                return False, None
            self.model.remove(co)

        self.model.remove(constraints)
        self.model.update()
        # self.restore_input_bounds()
        return True, implies


    def restore_input_bounds(self):

        properties = {
            1: {
                'lbs': [55947.691, -3.141593, -3.141593, 1145, 0],
                'ubs' : [60760, 3.141593, 3.141593, 1200, 60],
            } 
        }
        lbs = properties[self.p]['lbs']
        ubs = properties[self.p]['ubs']
        for i, var in enumerate(self.gurobi_vars):
            var.lb = (lbs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i]
            var.ub = (ubs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i]
        self.model.update()


    def init_input_property(self):
        properties = {
            1: {
                'lbs': [55947.691, -3.141593, -3.141593, 1145, 0],
                'ubs' : [60760, 3.141593, 3.141593, 1200, 60],
            } 
        }
        lbs = properties[self.p]['lbs']
        ubs = properties[self.p]['ubs']
        for i, var in enumerate(self.gurobi_vars):
            self.model.addConstr(var >= (lbs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i])
            self.model.addConstr(var <= (ubs[i] - self.dnn.input_means[i]) / self.dnn.input_ranges[i])

        self.model.update()

    def get_output_property(self, output):
        properties = {
            1: self._get_equation(output[:, 0]) >= (1500 - self.dnn.output_mean) / self.dnn.output_range
        }

        return properties[self.p]

    def check_deep_zono(self, lbs, ubs):
        properties = {
            1: ubs[0][0] >= (1500 - self.dnn.output_mean) / self.dnn.output_range
        }
        return properties[self.p]


if __name__ == '__main__':
        
    from utils import model_pa4

    dnn = model_pa4()

    assignment = {
        1: True,
        2: False,
        3: False,
        4: True
        # 'a1_0': True,
        # 'a1_1': False,
    }


    vars_mapping = {
        'a0_0': 1,
        'a0_1': 2,
        'a1_0': 3,
        'a1_1': 4,
    }

    layers_mapping = {
        0: [1, 2],
        1: [3, 4]
    }


    conditions = {
        'in': '(and (x0 < 0) (x1 > 1))',
        'out': '(y0 > y1)'
    }


    # dnn_constraint = DNNConstraint2(dnn, layers_mapping, conditions)
    # constraint, implies = dnn_constraint._get_substitution(assignment)

    # print('\n\nconstraints 2:', constraint)

    # print('implies')
    # for k, v in implies.items():
    #     print(k)
    #     for _ in v:
    #         print('    [+]', _)


    dnn_constraint = DNNConstraintGurobi(dnn, layers_mapping, conditions)
    constraint, implies = dnn_constraint._get_substitution(assignment)

    print('\n\nconstraints 3:', constraint)

    # print('implies')
    # for k, v in implies.items():
    #     print(k)
    #     for _ in v:
    #         print('    [+]', _)
