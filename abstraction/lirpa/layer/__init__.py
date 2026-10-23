'''Per-layer handling: one module per supported layer type.

To support a new layer, add a module providing ``TYPES`` (the torch module
classes), ``parse``, ``forward`` and ``backward``, and register it below.
A layer is a dict with the keys ``op``, ``in_shape`` and ``out_shape`` (shapes
without the batch dimension) plus the parameters its module needs.
'''

import torch

from . import (
    avgpool,
    batchnorm,
    conv,
    flatten,
    identity,
    leakyrelu,
    linear,
    maxpool,
    relu,
    sigmoid,
    tanh,
)

HANDLERS = {'linear': linear, 'conv': conv, 'batchnorm': batchnorm,
            'flatten': flatten, 'identity': identity, 'avgpool': avgpool,
            'maxpool': maxpool, 'relu': relu, 'leakyrelu': leakyrelu,
            'sigmoid': sigmoid, 'tanh': tanh}
BY_TYPE = {kind: handler for handler in HANDLERS.values()
           for kind in handler.TYPES}

# layers whose input needs concrete bounds (they are not affine)
NONLINEAR = ('relu', 'leakyrelu', 'sigmoid', 'tanh', 'maxpool')
# layers whose neurons the search can split into two linear pieces
SPLITTABLE = ('relu', 'leakyrelu')


def dtype_of(layer):
    # Floating dtype of the layer's tensors, or torch's default.
    for value in layer.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return torch.get_default_dtype()


def parse(module, in_shape):
    '''Layer description of a torch module that takes ``in_shape`` inputs.

    Adds ``in_shape`` and ``out_shape`` (without the batch dimension).
    '''
    handler = BY_TYPE.get(type(module))
    if handler is None:
        raise TypeError(f'unsupported layer {type(module).__name__}')
    layer = handler.parse(module)
    layer['in_shape'] = tuple(in_shape)
    probe = torch.zeros(1, *in_shape, dtype=dtype_of(layer))
    layer['out_shape'] = tuple(forward(layer, probe).shape[1:])
    return layer


def forward(layer, x):
    # Concrete evaluation of one layer on a batch ``(rows, *in_shape)``.
    return HANDLERS[layer['op']].forward(layer, x)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute symbolic bounds ``lo`` and ``hi`` through one layer.
    return HANDLERS[layer['op']].backward(layer, lo, hi, pre_bounds, alphas)
