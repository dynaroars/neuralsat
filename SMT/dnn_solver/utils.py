import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from pprint import pprint
import numpy as np
import z3

z3.set_option(rational_to_decimal=True)


def model_random(input_shape, hidden_shapes, output_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,), dtype='float32'))
    for unit in hidden_shapes:
        model.add(Dense(units=unit, activation='relu', dtype='float32'))
    model.add(Dense(units=output_shape, activation=None, dtype='float32'))
    return model

class InputParser:

    def neuron_name(layer_id, neuron_id, n_layers):
        if layer_id < n_layers - 1:
            return f'n{layer_id}_{neuron_id}'
        return f'y{neuron_id}'

    def parse(model):
        dnn = {}
        vars_mapping = {}
        idx = 1
        n_inputs = model.input_shape[1]
        n_layers = len(model.layers)

        prev_nodes = [f"x{n}" for n in range(n_inputs)]
        for lid, layer in enumerate(model.layers):
            weights, biases = layer.get_weights()
            cur_nodes = []
            for i in range(layer.output_shape[1]): # #nodes in layer
                node = InputParser.neuron_name(lid, i, n_layers)
                cur_nodes.append(node)

                node = node.replace('n', 'a')
                if node not in dnn:
                    dnn[node] = []
                if node not in vars_mapping and node.startswith('a'):
                    vars_mapping[node] = idx
                    idx += 1
                for p, q in zip(weights[:, i], prev_nodes):
                    dnn[node].append((p, q))
            prev_nodes = cur_nodes

        return dnn, vars_mapping


dnn = {
    'a0_0': [(1.0, 'x0'), (-1.0, 'x1')],
    'a0_1': [(1.0, 'x0'), (1.0, 'x1')],
    'a1_0': [(0.5, 'n0_0'), (-0.2, 'n0_1')],
    'a1_1': [(-0.5, 'n0_0'), (0.1, 'n0_1')],
    'y0' : [(1.0, 'n1_0'), (-1.0, 'n1_1')],
    'y1' : [(-1.0, 'n1_0'), (1.0, 'n1_1')],
}


if __name__ == '__main__':
    model = model_random(2, [4, 2], 2)
    dnn, vars_mapping = InputParser.parse(model)

    pprint(dnn)
    print()
    pprint(vars_mapping)