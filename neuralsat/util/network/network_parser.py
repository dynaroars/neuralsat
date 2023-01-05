from pprint import pprint
import sortedcontainers
import torch.nn as nn
import numpy as np
import torch
import math

from util.network.read_onnx import ONNXParser
import arguments

class NetworkParser:

    def parse(filename, device=torch.device('cpu')):
        if filename.lower().endswith('.onnx'):
            net = NetworkParser.parse_onnx(filename)
        else:
            print(f'Error extention: {filename}')
            raise NotImplementedError
        net.device = device
        return net.to(device)

    def parse_nnet(filename):
        model = NetworkNNET(filename)
        layers_mapping = {}
        idx = 1
        lid = 0
        for layer in model.layers[1:]: # discard input layer
            if isinstance(layer, nn.Linear):
                layers_mapping[lid] = sortedcontainers.SortedList()
                for i in range(layer.weight.shape[1]): # #nodes in layer
                    layers_mapping[lid].add(idx)
                    idx += 1
                lid += 1
        model.layers_mapping = layers_mapping
        return model

    def parse_onnx(filename):

        model = ONNXParser(filename).pytorch_model
        x = torch.randn(model.input_shape, dtype=arguments.Config['dtype'])

        count = 1
        layers_mapping = {}
        idx = 0
        
        for layer in model.layers:
            if isinstance(layer, nn.ReLU):
                layers_mapping[idx] = sortedcontainers.SortedList(range(count, count+x.numel()))
                idx += 1
                count += x.numel()
            x = layer(x)

        model.layers_mapping = layers_mapping
        return model
