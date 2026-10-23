# Average pooling ``z = avg_pool2d(a)``; an affine layer.

import math

import torch
import torch.nn.functional as functional

from . import affine, patches

TYPES = (torch.nn.AvgPool2d,)


def parse(module):
    if module.ceil_mode or module.divisor_override is not None:
        raise TypeError('AvgPool2d needs ceil_mode=False and no divisor')
    return {'op': 'avgpool', 'kernel': patches.pair(module.kernel_size),
            'stride': patches.pair(module.stride),
            'padding': patches.pair(module.padding),
            'count_include_pad': module.count_include_pad}


def forward(layer, x):
    return functional.avg_pool2d(x, layer['kernel'], layer['stride'],
                        layer['padding'], False, layer['count_include_pad'])


def substitute(coef, off, layer):
    # Substitute the pooling into patch coefficients (a depthwise conv).
    channels = layer['in_shape'][0]
    weight = torch.full((channels, 1, *layer['kernel']),
                        1 / math.prod(layer['kernel']),
                        dtype=coef['p'].dtype, device=coef['p'].device)
    new = patches.compose(coef, weight, layer['stride'], (1, 1), channels,
                          layer['padding'], layer['in_shape'])
    return new, off


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the pooling (exact, no relaxation needed).
    if patches.is_patches(lo[0]) and (
            layer['count_include_pad'] or layer['padding'] == (0, 0)):
        return substitute(*lo, layer), substitute(*hi, layer)
    return affine.backward(lambda x: forward(layer, x), layer['in_shape'],
                           layer['out_shape'], lo, hi)
