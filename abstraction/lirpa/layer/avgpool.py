# Average pooling ``z = avg_pool2d(a)``; an affine layer.

import torch
import torch.nn.functional as functional

from . import affine

TYPES = (torch.nn.AvgPool2d,)


def parse(module):
    if module.ceil_mode or module.divisor_override is not None:
        raise TypeError('AvgPool2d needs ceil_mode=False and no divisor')
    return {'op': 'avgpool', 'kernel': module.kernel_size,
            'stride': module.stride, 'padding': module.padding,
            'count_include_pad': module.count_include_pad}


def forward(layer, x):
    return functional.avg_pool2d(x, layer['kernel'], layer['stride'],
                        layer['padding'], False, layer['count_include_pad'])


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the pooling (exact, no relaxation needed).
    return affine.backward(lambda x: forward(layer, x), layer['in_shape'],
                           layer['out_shape'], lo, hi)
