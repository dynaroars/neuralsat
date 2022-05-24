import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import settings


class SymbolicNetwork:

    def __init__(self, net):
        self.net = net
        self.layers_mapping = net.layers_mapping

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        idx = 0
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                l = SymbolicLinear(layer)
            elif isinstance(layer, nn.ReLU):
                l = SymbolicReLU(self.layers_mapping[idx])
                idx += 1
            elif isinstance(layer, nn.Conv2d):
                l = SymbolicConv2d(layer)
            elif isinstance(layer, nn.Flatten):
                l = SymbolicFlatten()
            else:
                print(layer)
                raise NotImplementedError
            layers.append(l)
        return layers


    @property
    def symbolic_input(self):
        default_input = F.one_hot(torch.arange(0, self.net.n_input)).view(*self.net.input_shape[1:], self.net.n_input)
        return torch.concat([default_input, torch.zeros(*self.net.input_shape[1:], 1)], dim=-1).to(settings.DTYPE)


    @torch.no_grad()
    def __call__(self, assignment):
        x = self.symbolic_input
        backsub_dict = {}
        for layer in self.layers:
            x, flag_break, backsub = layer(x, assignment)
            backsub_dict.update(backsub)
            if flag_break:
                break
        return x, backsub_dict



class SymbolicLinear:

    def __init__(self, layer):
        self.weight = layer.weight.to(settings.DTYPE)
        self.bias = layer.bias.to(settings.DTYPE)


    def __call__(self, x, assignment):
        x = self.weight @ x
        x[:, -1] += self.bias
        return x, False, {}


class SymbolicReLU:

    def __init__(self, variables):
        self.variables = variables
        self.set_variables = set(variables)


    def __call__(self, x, assignment):
        output = torch.zeros_like(x, dtype=settings.DTYPE)
        assert math.prod(output.shape[:-1]) == len(self.variables), f'layers_mapping sai me roi: {x.shape}'

        flag_break = not self.set_variables.issubset(set(assignment.keys()))
        mask = torch.tensor([assignment.get(v, False) for v in self.variables]).view(x.shape[:-1])
        tmp_x = x.view(-1, x.shape[-1])
        backsub_dict = {v: tmp_x[i] for i, v in enumerate(self.variables)}
        output[mask] = x[mask]

        return output, flag_break, backsub_dict



class SymbolicConv2d:

    def __init__(self, layer):
        self.weight = layer.weight.to(settings.DTYPE) # OUT x IN x K1 x K2
        self.bias = layer.bias.to(settings.DTYPE)

        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.groups = layer.groups

        self.run_conv2d = True


    def __call__(self, x, assignment):
        if self.padding[0] != 0 or self.padding[1] != 0:
            p3d = (0, 0, self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            x = F.pad(x, p3d, 'constant', 0)

        if self.run_conv2d:
            x = x.permute(3, 0, 1, 2)
            x = F.conv2d(x, self.weight, bias=None, stride=self.stride)
            x = x.permute(1, 2, 3, 0)
        else:
            x = F.conv3d(x.unsqueeze(0), 
                         self.weight.unsqueeze(-1), 
                         bias=None, 
                         stride=(*self.stride, 1)).squeeze(0)
        
        x[..., -1] += self.bias[:, None, None]
        return x, False, {}




class SymbolicFlatten:

    def __init__(self):
        pass

    def __call__(self, x, assignment):
        x = x.view(-1, x.shape[-1])
        return x, False, {}
