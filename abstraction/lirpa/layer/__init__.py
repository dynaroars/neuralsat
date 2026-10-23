'''Per-layer handling: one module per supported layer type.

To support a new layer, add a module providing ``TYPES`` (the torch module
classes), ``parse``, ``forward`` and ``backward``, and register it below.
A layer is a dict with the keys ``op``, ``in_shape`` and ``out_shape`` (shapes
without the batch dimension) plus the parameters its module needs.

The coefficients that ``backward`` receives are dense matrices, or patches (see
``patches``) when the start node is conv-shaped; Conv2d, BatchNorm, AvgPool2d,
MaxPool2d and the pointwise activations keep the patch form, the others only
see dense coefficients.
'''

import torch

from . import (
    affine,
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


def interval(layer, lower, upper):
    # Output interval of a non-affine layer from flat input bounds.
    return HANDLERS[layer['op']].interval(layer, lower, upper)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute symbolic bounds ``lo`` and ``hi`` through one layer.
    return HANDLERS[layer['op']].backward(layer, lo, hi, pre_bounds, alphas)


def absolute(layer):
    '''An affine layer with absolute weights and no bias.

    It maps the radius of a box to the radius of the image box.
    '''
    out = dict(layer)
    for key in ('weight', 'scale'):
        if key in out:
            out[key] = out[key].abs()
    for key in ('bias', 'shift'):
        if out.get(key) is not None:
            out[key] = torch.zeros_like(out[key])
    return out


def zonotope(layer, center, gens, lower, upper):
    '''Push the flat zonotope ``(center, gens)`` through one layer.

    ``lower`` and ``upper`` are the flat bounds of the layer's input; only a
    non-affine layer reads them. Returns the new center and generators.
    '''
    if layer['op'] in NONLINEAR:
        return HANDLERS[layer['op']].zonotope(layer, center, gens, lower,
                                              upper)
    return affine.zonotope(lambda x: forward(layer, x), layer['in_shape'],
                           center, gens)
