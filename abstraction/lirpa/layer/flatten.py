# Flatten layer: only the shape changes, the values keep their order.

import torch

TYPES = (torch.nn.Flatten,)


def parse(module):
    if module.start_dim != 1 or module.end_dim != -1:
        raise TypeError('Flatten must flatten every dimension after the batch')
    return {'op': 'flatten'}


def forward(layer, x):
    return x.reshape(x.shape[0], -1)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # The coefficients are over the flattened values already.
    return lo, hi
