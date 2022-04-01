import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import Input
from tensorflow import keras
from pprint import pprint
import tensorflow as tf
import sortedcontainers
import numpy as np

from utils.read_nnet import NetworkDeepZono, ReLU, Linear

def model_pa4():
    model = Sequential()
    model.add(Input(shape=(2,), dtype='float32'))
    model.add(Dense(units=2,
                    activation=activations.relu,
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, 1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer(
                        [[0.0], [0.0]]),
                    dtype='float32'
                    ))
    model.add(Dense(units=2,
                    activation=activations.relu,
                    kernel_initializer=tf.constant_initializer(
                        [[0.5, -0.5], [-0.2, 0.1]]),
                    bias_initializer=tf.constant_initializer(
                        [[0.0], [0.0]]),
                    dtype='float32'
                    ))
    model.add(Dense(units=2,
                    activation=None,
                    kernel_initializer=tf.constant_initializer(
                        [[1.0, -1.0], [-1.0, 1.0]]),
                    bias_initializer=tf.constant_initializer(
                        [[0.0], [0.0]]),
                    dtype='float32'
                    ))
    return model


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

    def neuron_name(layer_id, neuron_id):
        return f'a{layer_id}_{neuron_id}'

    def parse(model):
        vars_mapping = {}
        layers_mapping = {}
        idx = 1
        n_inputs = model.input_shape[1]

        lid = 0
        for layer in model.layers[1:]: # discard inputlayer
            if type(layer) is Linear:
                layers_mapping[lid] = sortedcontainers.SortedList()
                for i in range(layer.output_shape[1]): # #nodes in layer
                    node = InputParser.neuron_name(lid, i)
                    if node not in vars_mapping:
                        vars_mapping[node] = idx
                        layers_mapping[lid].add(idx)
                        idx += 1
                lid += 1

        return vars_mapping, layers_mapping



if __name__ == '__main__':


    # model = model_random(3, [7, 5, 6], 5)
    # model.save('../example/model.keras')
    # model = keras.models.load_model('../example/model.keras')
    model = NetworkDeepZono('example/random.nnet')
    vars_mapping, layers_mapping = InputParser.parse(model)

    # pprint(dnn)
    # print()
    pprint(vars_mapping)

    reversed_layers_mapping = {i: k for k, v in layers_mapping.items() for i in v}
    pprint(layers_mapping)
    # pprint(reversed_layers_mapping)
