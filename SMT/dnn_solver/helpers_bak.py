from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import activations
from tensorflow.keras import Input
from tensorflow import keras
import tensorflow as tf
import numpy as np


import time
import copy
import re


class Utils:

    def And(term1, term2):
        return f'(and {term1} {term2})'

    def Not(term1):
        return f'(not {term1})'

    def Or(term1, term2):
        return f'(or {term1} {term2})'

    def Prove(term1, term2):
        return Utils.And(term1, Utils.Not(term2))

def clean(string):
    string = string.replace('?', '')
    string = re.sub(r'\n', ' ', string)
    return string


class DNNConstraint3:

    def __init__(self, dnn, layers_mapping, conditions):
        self.dnn = dnn
        self.conditions = conditions
        self.layers_mapping = layers_mapping

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
        return ' + '.join([f'{coeffs[i]}*x{i}' for i in np.nonzero(coeffs[:-1])[0]] + [str(coeffs[-1])])


    def _get_substitution(self, assignment):
        nodes = self._find_nodes(assignment)
        return_output = True if nodes is None else False
        imply_nodes = copy.deepcopy(nodes)
        zero = np.zeros(self.n_inputs+1)

        prev_nodes = np.concatenate([self.input_symbols, np.zeros([1, self.n_inputs])]).T
        # print(prev_nodes)

        substitute_dict = {}

        for layer_id, layer in enumerate(self.dnn.layers):
            layer = self.dnn.layers[layer_id]
            weight, bias = layer.get_weights()
            variables = self.layers_mapping.get(layer_id, None)

            if variables is None: # output layer
                output = np.array(prev_nodes).T.dot(weight)
                output[-1] += bias
                output_condition = self.conditions['out']
                for i, o in enumerate(self.output_symbols):
                    output_condition = output_condition.replace(o, self._get_equation(output[:, i]))
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
        # current

        constraint = self.conditions['in']
        for node, status in assignment.items():
            if status:
                constraint = Utils.And(constraint, '(%s > 0)' % str(substitute_dict[node]))
            else:
                if str(substitute_dict[node]) == '0.0':
                    continue
                constraint = Utils.And(constraint, '(%s <= 0)' % str(substitute_dict[node]))

        # implies
        implies = {}
        if imply_nodes:
            for node in imply_nodes:
                implies[node] = [
                    clean(Utils.Prove(constraint, '(%s <= 0)' % str(substitute_dict[node]))),
                    clean(Utils.Prove(constraint, '(%s > 0)' % str(substitute_dict[node]))),
                ]

        # output
        if return_output:
            constraint = Utils.And(constraint, output_condition) # prove(f, not(g)) = f and g

        return clean(constraint), implies


    def __call__(self, assignment):
        return self._get_substitution(assignment)


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
