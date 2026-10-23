'''Linear layer ``z = weight a + bias``.

Layers are dicts; every layer module provides ``TYPES``, ``parse``,
``forward`` and ``backward``.
'''

import torch

TYPES = (torch.nn.Linear,)


def parse(module):
    # Layer description of an ``nn.Linear``.
    return {'op': 'linear', 'weight': module.weight.detach(),
            'bias': module.bias.detach()}


def forward(layer, x):
    # Concrete evaluation on a ``(rows, in)`` batch.
    return x @ layer['weight'].t() + layer['bias']


def substitute(coef, off, layer):
    # Substitute ``v = weight w + bias`` into ``coef v + off``.
    return coef @ layer['weight'], coef @ layer['bias'] + off


def backward(layer, lo, hi, pre_bounds, alphas=None):
    '''Back-substitute the exact affine map into both symbolic bounds.

    ``lo`` and ``hi`` are ``(coef, off)`` pairs over this layer's output;
    the result is over its input. ``pre_bounds`` and ``alphas`` are unused:
    affine maps need no relaxation.
    '''
    return substitute(*lo, layer), substitute(*hi, layer)
