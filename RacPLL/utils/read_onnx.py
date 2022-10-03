import torch.nn as nn
import numpy as np
import torch
import math

from abstract.crown.utils import load_model_onnx


class PyTorchModelWrapper(nn.Module):

    def __init__(self, layers):
        super().__init__()

        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = layers

        self.layers_mapping = None
        self.input_shape = None

        self.n_input = None
        self.n_output = None
        

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


    def forward_grad(self, x):
        return self.layers(x)



    @torch.no_grad()
    def get_assignment(self, x):
        idx = 0
        implication = {}
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                s = torch.zeros_like(x, dtype=int) 
                s[x > 0] = 1
                implication.update(dict(zip(self.layers_mapping[idx], s.flatten().numpy().astype(dtype=bool))))
                idx += 1
        return implication

    @torch.no_grad()
    def get_concrete(self, x):
        x = x.view(self.input_shape)
        idx = 0
        implication = {}
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                implication.update(dict(zip(self.layers_mapping[idx], x.view(-1))))
                idx += 1
            x = layer(x)
        return implication

    @torch.no_grad()
    def forward_layer(self, x, lid):
        relu_idx = 0
        # print(lid)
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                relu_idx += 1
            if relu_idx <= lid:
                continue
            # print(layer)
            x = layer(x)
        return x



class ONNXParser:

    def __init__(self, filename, dataset):

        force_convert = False
        if dataset == 'mnist':
            input_shape = (1, 1, 28, 28)
            n_output = 10
        elif dataset == 'cifar':
            input_shape = (1, 3, 32, 32)
            n_output = 10
        elif dataset == 'acasxu':
            input_shape = (1, 5)
            n_output = 5
            force_convert = True
        else:
            raise 

        model, is_channel_last = load_model_onnx(filename, input_shape[1:], force_convert=force_convert)
        model = model.eval()

        if is_channel_last:
            input_shape = input_shape[:1] + input_shape[2:] + input_shape[1:2]
            print(f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {input_shape}')

        self.pytorch_model = PyTorchModelWrapper(model)
        self.pytorch_model.n_input = math.prod(input_shape)
        self.pytorch_model.n_output = n_output
        self.pytorch_model.input_shape = input_shape
