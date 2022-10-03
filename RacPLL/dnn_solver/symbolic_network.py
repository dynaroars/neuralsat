import torch.nn.functional as F
import torch.nn as nn
import onnx2pytorch
import torch
import math

import settings
import utils


class SymbolicNetwork:

    def __init__(self, net):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.device = net.device

        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        idx = 0
        for layer in self.net.layers:
            print(type(layer))
            if isinstance(layer, nn.Linear):
                l = SymbolicLinear(layer, self.device)
            elif isinstance(layer, nn.ReLU):
                l = SymbolicReLU(self.layers_mapping[idx], self.device)
                idx += 1
            elif isinstance(layer, nn.Conv2d):
                l = SymbolicConv2d(layer, self.device)
            elif isinstance(layer, nn.Flatten) or isinstance(layer, onnx2pytorch.operations.flatten.Flatten):
                l = SymbolicFlatten(self.device)
            # elif isinstance(layer, onnx2pytorch.operations.Reshape):
            #     l = SymbolicReshape(layer, self.device)
            # elif isinstance(layer, onnx2pytorch.operations.Transpose):
            #     l = SymbolicTranspose(layer, self.device)
            # elif isinstance(layer, utils.read_onnx.Sub):
            #     l = SymbolicSub(layer, self.device)
            # elif isinstance(layer, utils.read_onnx.Div):
            #     l = SymbolicDiv(layer, self.device)
            elif isinstance(layer, nn.BatchNorm2d):
                l = SymbolicBatchNorm2d(layer)
            else:
                print(layer, type(layer))
                raise NotImplementedError
            layers.append(l)
        return layers


    @property
    def symbolic_input(self):
        default_input = F.one_hot(torch.arange(0, self.net.n_input)).view(*self.net.input_shape[1:], self.net.n_input)
        return torch.concat([default_input, torch.zeros(*self.net.input_shape[1:], 1)], dim=-1).to(settings.DTYPE).to(self.device)


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

    def __init__(self, layer, device):
        self.weight = layer.weight.to(settings.DTYPE).to(device)
        self.bias = layer.bias.to(settings.DTYPE).to(device)


    def __call__(self, x, assignment):
        x = self.weight @ x
        x[:, -1] += self.bias
        return x, False, {}


class SymbolicReLU:

    def __init__(self, variables, device):
        self.variables = variables
        self.set_variables = set(variables)
        self.device = device


    def __call__(self, x, assignment):
        output = torch.zeros_like(x, dtype=settings.DTYPE, device=self.device)
        assert math.prod(output.shape[:-1]) == len(self.variables), f'layers_mapping sai me roi: {x.shape}'

        flag_break = not self.set_variables.issubset(set(assignment.keys()))
        mask = torch.tensor([assignment.get(v, False) for v in self.variables], device=self.device).view(x.shape[:-1])
        tmp_x = x.view(-1, x.shape[-1])
        backsub_dict = {v: tmp_x[i] for i, v in enumerate(self.variables)}
        output[mask] = x[mask]

        return output, flag_break, backsub_dict



class SymbolicConv2d:

    def __init__(self, layer, device):
        self.weight = layer.weight.to(settings.DTYPE).to(device) # OUT x IN x K1 x K2
        self.bias = layer.bias.to(settings.DTYPE).to(device)

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

    def __init__(self, device):
        pass

    def __call__(self, x, assignment):
        x = x.view(-1, x.shape[-1])
        return x, False, {}



class SymbolicTranspose:

    def __init__(self, layer, device):
        # discard batch_size dimension
        self.dims = tuple([d-1 for d in layer.dims[1:]] + [len(layer.dims)-1])

    def __call__(self, x, assignment):
        if not self.dims:
            dims = tuple(reversed(range(x.dim())))
            raise
        else:
            dims = self.dims
        x = x.permute(dims)
        return x, False, {}



class SymbolicReshape:

    def __init__(self, layer, device):
        self.shape = layer.shape

    def __call__(self, x, assignment):
        shape = [s if s != 0 else x.size(i) for i, s in enumerate(self.shape)]
        if len(shape)==2 and shape[0]==-1:
            shape = (shape[1], -1)
        else:
            raise NotImplementedError
        x = x.reshape(shape)
        return x, False, {}




class SymbolicSub:

    def __init__(self, layer, device):
        self.constant = layer.constant[0].unsqueeze(-1)

    def __call__(self, x, assignment):
        x = x - self.constant
        return x, False, {}



class SymbolicDiv:

    def __init__(self, layer, device):
        self.constant = layer.constant[0].unsqueeze(-1)

    def __call__(self, x, assignment):
        x = torch.div(x, self.constant)
        return x, False, {}





class SymbolicBatchNorm2d:

    def __init__(self, layer):
        self.running_mean = layer.running_mean.repeat(1, 1, 1).permute(2, 0, 1)
        self.running_var = layer.running_var.repeat(1, 1, 1, 1).permute(3, 0, 1, 2)

    def __call__(self, x, assignment):
        x[..., -1] -= self.running_mean
        x /= self.running_var
        return x, False, {}
