# Convolution layer ``z = conv2d(a, weight) + bias``.

import torch
import torch.nn.functional as functional

from . import affine, patches

TYPES = (torch.nn.Conv2d,)


def parse(module):
    # Layer description of an ``nn.Conv2d`` (zero padding only).
    if module.padding_mode != 'zeros' or isinstance(module.padding, str):
        raise TypeError('Conv2d needs zero padding given as numbers')
    bias = module.bias
    return {'op': 'conv', 'weight': module.weight.detach(),
            'bias': None if bias is None else bias.detach(),
            'stride': module.stride, 'padding': module.padding,
            'dilation': module.dilation, 'groups': module.groups}


def forward(layer, x):
    # Concrete evaluation on a ``(rows, C, H, W)`` batch.
    return functional.conv2d(
        x, layer['weight'], layer['bias'], layer['stride'], layer['padding'],
        layer['dilation'], layer['groups'])


def substitute(coef, off, layer):
    # Substitute the convolution into patch coefficients (exact).
    new = patches.compose(coef, layer['weight'], layer['stride'],
                          layer['dilation'], layer['groups'],
                          layer['padding'], layer['in_shape'])
    if layer['bias'] is not None:
        bias = layer['bias'].reshape(1, 1, 1, -1, 1, 1)
        off = off + patches.row_sum(patches.masked(coef) * bias, coef)
    return new, off


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the convolution (exact, no relaxation needed).
    if patches.is_patches(lo[0]):
        return substitute(*lo, layer), substitute(*hi, layer)
    return affine.backward(lambda x: forward(layer, x), layer['in_shape'],
                           layer['out_shape'], lo, hi)
