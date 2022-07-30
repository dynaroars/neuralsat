import torch
import torch.nn as nn
from functools import reduce


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dim=None):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.dim = dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward_concrete(self, x):
        return self.conv(x)

    def forward_abstract(self, x):
        return x.conv2d(self.conv.weight, self.conv.bias, self.stride, self.conv.padding, self.dilation, self.conv.groups)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            ret = self.forward_concrete(x)
        else:
            ret = self.forward_abstract(x)
        return ret


class Sequential(nn.Module):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward_until(self, i, x):
        for layer in self.layers[:i+1]:
            x = layer(x)
        return x

    def forward_from(self, i, x):
        for layer in self.layers[i+1:]:
            x = layer(x)
        return x

    def total_abs_l1(self, x):
        ret = 0
        for layer in self.layers:
            x = layer(x)
            ret += x.l1()
        return ret

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(self, x, init_lambda=False, skip_norm=False):
        for layer in self.layers:
            if isinstance(layer, Normalization) and skip_norm:
                continue
            if isinstance(layer, ReLU):
                x = layer(x, init_lambda)
            else:
                x = layer(x)
        return x
        
    
class ReLU(nn.Module):

    def __init__(self, dims=None):
        super(ReLU, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(torch.ones(dims))
        self.bounds = None

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x, init_lambda=False):
        return x.relu()


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view((x.size()[0], -1))


class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.linear(x)
        else:
            return x.linear(self.linear.weight, self.linear.bias)


class Normalization(nn.Module):

    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = mean
        self.sigma = sigma

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.sigma
        ret = x.normalize(self.mean, self.sigma)
        return ret
