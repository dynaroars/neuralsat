import sortedcontainers
import torch.nn as nn
import numpy as np
import collections
import warnings
import torch
import os

from util.network.read_onnx import parse_onnx
from extra import *
import arguments


def get_activation_shape(name, result):
    def hook(model, input, output):
        result[name] = output.shape
    return hook


class ONNXParser(nn.Module):

    def __init__(self, filename):
        super().__init__()

        self.layers, self.input_shape, self.output_shape = parse_onnx(filename)
        self.n_input = np.prod(self.input_shape)
        self.n_output = np.prod(self.output_shape)
        
        self.activation = collections.OrderedDict()
        for name, layer in self.layers.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation_shape(name, self.activation))
    
    
    def forward(self, x):
        return self.layers(x)
    

class PytorchParser(nn.Module):

    def __init__(self, net):
        super().__init__()

        self.layers = net
        self.input_shape = net.input_shape
        self.output_shape = net.output_shape

        self.n_input = np.prod(self.input_shape)
        self.n_output = np.prod(self.output_shape)
        
        self.activation = collections.OrderedDict()
        for name, layer in self.layers.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(get_activation_shape(name, self.activation))

    
    def forward(self, x):
        return self.layers(x)
    
    
class NetworkParser:

    def parse(filename, device=torch.device('cpu'), ckpt=None):
        
        if filename.lower().endswith('.onnx'):
            assert os.path.exists(filename)
            net = NetworkParser.parse_onnx(filename)
        elif (ckpt is not None) and os.path.exists(ckpt):
            net = NetworkParser.parse_pytorch(filename, ckpt)
        else:
            raise NotImplementedError("Checkpoint is missing")
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


    def add_layer_mapping(model):
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
        assert count > 1, "Only supports ReLU activation"
        print("Number of SAT variables:", count - 1)
        return model
        
        
    def parse_onnx(filename):
        model = ONNXParser(filename)
        return NetworkParser.add_layer_mapping(model)
    

    def parse_pytorch(model_name, ckpt):
        try:
            net = eval(model_name)
        except SyntaxError:
            raise ValueError(model_name)
        else:
            sd = torch.load(ckpt, map_location=torch.device('cpu'))
            if 'state_dict' in sd:
                sd = sd['state_dict']
            if isinstance(sd, list):
                sd = sd[0]
            if not isinstance(sd, dict):
                raise NotImplementedError("Unknown model format.")
            net.layers.load_state_dict(sd)
            
            model = PytorchParser(net)
            
        return NetworkParser.add_layer_mapping(model)
           