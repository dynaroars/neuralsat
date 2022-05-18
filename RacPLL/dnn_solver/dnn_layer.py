import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import settings


class DNNLayer:

    def __init__(self, net):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.linear_input = None

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        idx = 0
        last = None
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                if self.linear_input is None:
                    self.linear_input = True
                l = DNNLinear(layer, last)
            elif isinstance(layer, nn.ReLU):
                l = DNNReLU(self.layers_mapping[idx], last)
                idx += 1
            elif isinstance(layer, nn.Conv2d):
                if self.linear_input is None:
                    self.linear_input = False
                l = DNNConv2d(layer, last)
            elif isinstance(layer, nn.Flatten):
                l = DNNFlatten(last)
            else:
                print(layer)
                raise NotImplementedError
            layers.append(l)
        return layers


    @property
    def symbolic_input(self):
        if self.linear_input: #linear
            return torch.hstack([torch.eye(self.net.n_input), torch.zeros(self.net.n_input, 1)]).to(settings.DTYPE)
        else: # conv2d
            _, C, H, W = self.net.input_shape
            # assert N == 1, f'batch size {N} > 1'
            default_input = F.one_hot(torch.arange(0, C*H*W)).view(C, H, W, C*H*W)
            return torch.concat([default_input, torch.zeros(C, H, W, 1)], dim=-1).to(settings.DTYPE)


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



class DNNLinear:

    def __init__(self, layer, last):
        self.weight = layer.weight
        self.bias = layer.bias
        self.last = last


    def __call__(self, x, assignment):
        x = self.weight @ x
        x[:, -1] += self.bias
        return x, False, {}


class DNNReLU:

    def __init__(self, variables, last):
        self.variables = variables
        self.set_variables = set(variables)
        self.last = last


    def __call__(self, x, assignment):
        output = torch.zeros_like(x, dtype=settings.DTYPE)
        assert math.prod(output.shape[:-1]) == len(self.variables), f'layers_mapping sai me roi: {x.shape}'

        flag_break = not self.set_variables.issubset(set(assignment.keys()))
        mask = torch.tensor([assignment.get(v, False) for v in self.variables]).view(x.shape[:-1])
        tmp_x = x.view(-1, x.shape[-1])
        backsub_dict = {v: tmp_x[i] for i, v in enumerate(self.variables)}
        output[mask] = x[mask]

        return output, flag_break, backsub_dict



class DNNConv2d:

    def __init__(self, layer, last):
        self.weight = layer.weight # OUT x IN x K1 x K2
        self.bias = layer.bias

        self.in_channels = layer.in_channels
        self.out_channels = layer.out_channels
        self.kernel_size = layer.kernel_size
        self.stride = layer.stride
        self.padding = layer.padding
        self.groups = layer.groups

        self.run_conv2d = True
        self.last = last

    def __call__(self, x, assignment):
        C, H, W, _ = x.shape
        out = [] 
        for cin in range(C):
            o = self._forward_symbolic_one_in_channel(x[cin], self.weight[:, cin], run_conv2d=self.run_conv2d)
            out.append(o)

        out = torch.stack(out).sum(dim=0)
        out[..., -1] += self.bias[:, None, None]
        return out, False, {}


    def _forward_symbolic_one_in_channel(self, x, weight, run_conv2d=True):
        x = x.unsqueeze(0)
        weight = weight.unsqueeze(1)

        if run_conv2d:
            x = x.permute(3, 0, 1, 2)
            x = F.conv2d(x, weight, bias=None, stride=self.stride)
            x = x.permute(1, 2, 3, 0)
        else:
            x = F.conv3d(x.unsqueeze(0), weight.unsqueeze(-1), bias=None, stride=(*self.stride, 1))
            x = x.squeeze(0)
        return x



class DNNFlatten:

    def __init__(self, last):
        self.last = last

    def __call__(self, x, assignment):
        x = x.view(-1, x.shape[-1])
        return x, False, {}
