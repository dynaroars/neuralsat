import sortedcontainers
import torch.nn as nn
import numpy as np
import torch

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
        model = ONNXParser(filename)

        # forward to record relu shapes
        x = torch.randn(model.input_shape, dtype=arguments.Config['dtype'])
        assert x.shape[0] == 1
        output_pytorch = model(x)
        
        # extract boolean abstraction
        count = 1
        layers_mapping = {}
        idx = 0
        for k, v in model.activation.items():
            layers_mapping[idx] = sortedcontainers.SortedList(range(count, count+np.prod(v)))
            idx += 1
            count += np.prod(v)
        model.layers_mapping = layers_mapping
        
        return model
