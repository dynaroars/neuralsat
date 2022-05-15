import torch.nn as nn
import torch

import settings


class DNNLayer:

    def __init__(self, net, layers_mapping):
        self.net = net
        self.layers_mapping = layers_mapping
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                layers.append(DNNLinear(layer))
            elif isinstance(layer, nn.ReLU):
                layers.append(DNNReLU(self.layers_mapping[idx]))
                idx += 1
            else:
                raise NotImplementedError

        return layers

    @torch.no_grad()
    def __call__(self, x, assignment):
        backsub_dict = {}
        for layer in self.layers:
            x, flag_break, backsub = layer(x, assignment)
            backsub_dict.update(backsub)
            if flag_break:
                break
        return x, backsub_dict



class DNNLinear:

    def __init__(self, layer):
        self.weight = layer.weight
        self.bias = layer.bias


    def __call__(self, x, assignment):
        x = self.weight @ x
        x[:, -1] += self.bias
        return x, False, {}


class DNNReLU:

    def __init__(self, variables):
        self.variables = variables
        self.set_variables = set(variables)


    def __call__(self, x, assignment):
        output = torch.zeros_like(x, dtype=settings.DTYPE)
        flag_break = not self.set_variables.issubset(set(assignment.keys()))
        backsub_dict = {}
        for i, v in enumerate(self.variables):
            if assignment.get(v, False): 
                output[i] = x[i]
            backsub_dict[v] = x[i]
        return output, flag_break, backsub_dict



class DNNConv2d:

    def __init__(self, layer):
        self.weight = layer.weight # OUT x IN x K1 x K2
        self.bias = layer.bias
        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.groups = layer.groups

    def __call__(self, x, assignment):
        # conv2d
        return x, False, {}