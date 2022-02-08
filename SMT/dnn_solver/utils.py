import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras import Input
from tensorflow import keras
from pprint import pprint
import sortedcontainers
import numpy as np
import z3

z3.set_option(rational_to_decimal=True)


def model_random(input_shape, hidden_shapes, output_shape):
    bias_initializer = initializers.HeNormal()
    model = Sequential()
    model.add(Input(shape=(input_shape,), dtype='float32'))
    for unit in hidden_shapes:
        model.add(Dense(
            units=unit, 
            activation='relu', 
            bias_initializer=bias_initializer, 
            dtype='float32'))
    model.add(Dense(
        units=output_shape, 
        bias_initializer=bias_initializer, 
        activation=None, 
        dtype='float32'))
    return model

class InputParser:

    def neuron_name(layer_id, neuron_id, n_layers):
        if layer_id < n_layers - 1:
            return f'n{layer_id}_{neuron_id}'
        return f'y{neuron_id}'

    def parse(model):
        dnn = {}
        vars_mapping = {}
        layers_mapping = {}
        idx = 1
        n_inputs = model.input_shape[1]
        n_layers = len(model.layers)

        prev_nodes = [f"x{n}" for n in range(n_inputs)]
        for lid, layer in enumerate(model.layers):
            if lid < n_layers - 1:
                layers_mapping[lid] = sortedcontainers.SortedList()
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
                    layers_mapping[lid].add(idx)
                    idx += 1
                for p, q in zip(weights[:, i], prev_nodes):
                    dnn[node].append((p, q))
                dnn[node].append(biases[i])
            prev_nodes = cur_nodes

        return dnn, vars_mapping, layers_mapping



if __name__ == '__main__':


    # model = model_random(2, [2, 2], 2)
    # model.save('../example/model.keras')
    model = keras.models.load_model('../example/model.keras')

    dnn, vars_mapping, layers_mapping = InputParser.parse(model)

    pprint(dnn)
    # print()
    # pprint(vars_mapping)

    reversed_layers_mapping = {i: k for k, v in layers_mapping.items() for i in v}
    pprint(layers_mapping)
    pprint(reversed_layers_mapping)
