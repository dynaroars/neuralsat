import torch.nn.functional as F
import torch.nn as nn
import torch
from pprint import pprint
import tensorflow as tf
import sortedcontainers
import numpy as np

from utils.read_nnet import Network, Linear

class InputParser:

    def neuron_name(layer_id, neuron_id):
        return f'a{layer_id}_{neuron_id}'

    def parse(model):
        vars_mapping = {}
        layers_mapping = {}
        idx = 1

        lid = 0
        for layer in model.layers[1:]: # discard inputlayer
            if isinstance(layer, nn.Linear):
                layers_mapping[lid] = sortedcontainers.SortedList()
                for i in range(layer.weight.shape[1]): # #nodes in layer
                    node = InputParser.neuron_name(lid, i)
                    if node not in vars_mapping:
                        vars_mapping[node] = idx
                        layers_mapping[lid].add(idx)
                        idx += 1
                lid += 1

        return vars_mapping, layers_mapping

class DNFConstraint:

    def __init__(self, constraints):
        self.constraints = constraints

if __name__ == '__main__':


    # model = model_random(3, [7, 5, 6], 5)
    # model.save('../example/model.keras')
    # model = keras.models.load_model('../example/model.keras')
    model = Network('example/random.nnet')
    vars_mapping, layers_mapping = InputParser.parse(model)

    # pprint(dnn)
    # print()
    pprint(vars_mapping)

    reversed_layers_mapping = {i: k for k, v in layers_mapping.items() for i in v}
    pprint(layers_mapping)
    # pprint(reversed_layers_mapping)
