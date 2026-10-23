'''Zonotope abstraction (DeepZ) for sequential networks.

A zonotope is a center and a list of generators,
``z = c + sum_k eps_k g_k`` with every ``eps_k`` in ``[-1, 1]``. The input
box starts as one generator per input. An affine layer maps the center and
the generators exactly, so the dependence between neurons is kept. A
non-affine layer maps a neuron to a strip ``slope z + [low, high]`` (see
``layer.activation.parallelogram``), which adds one generator per neuron
whose strip has a width.

The bounds of a layer are ``c -/+ sum_k |g_k|``. The interface is the one of
``lirpa.crown``, so the searches can use ``layer_bounds`` as ``bound_fn``.
The center and the generators are flat; the model's input keeps its shape.
'''

import torch

from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops


def concretize(center, gens):
    # The box around a zonotope.
    radius = gens.abs().sum(0)
    return center - radius, center + radius


def layer_bounds(layers, x0, eps, splits=None, core=False):
    '''Concrete bounds ``[(l, u)]`` of the layers in ``bounded_layers``.

    ``splits`` and ``core`` work as in ``crown.layer_bounds``. A split only
    changes the bounds the next layer sees: a neuron fixed active or inactive
    is stable there, so its transformer is exact inside the subdomain.
    '''
    splits = splits or {}
    flat = x0.reshape(-1)
    center = flat
    gens = eps * torch.eye(flat.numel(), dtype=flat.dtype, device=flat.device)
    box = (flat - eps, flat + eps)
    bounded = crown.bounded_layers(layers)
    effective, bounds = {}, {}
    for i, layer in enumerate(layers):
        pre = bounds.get(i - 1, box)
        center, gens = ops.zonotope(layer, center, gens, *pre)
        if i in bounded:
            bound = concretize(center, gens)
            split = splits.get(i + 1)
            if core and split is not None:
                effective[i + 1] = crown.split_effect(bound, split)
            bounds[i] = crown.clamp_split(bound, split)
    return (list(bounds.values()), effective) if core else list(
        bounds.values())


def bounds(model, x0, eps):
    '''Concrete bounds ``[(l, u)]`` of every pre-activation of ``model``.

    ``x0`` has the shape of one model input; the last entry is the output.
    '''
    return layer_bounds(crown.from_sequential(model, x0.shape), x0, eps)


def margin_lower_bounds(model, x0, eps, label):
    # Lower bounds of f_label(x) - f_k(x) for every k != label.
    layers = crown.margin_layers(model, x0.shape, label)
    return layer_bounds(layers, x0, eps)[-1][0]


def verify(model, x0, eps, label):
    # True iff label is certified to be the argmax on the ball.
    return bool((margin_lower_bounds(model, x0, eps, label) > 0).all())
