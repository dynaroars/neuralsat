'''Minimal CROWN bound propagation for sequential networks.

Functional implementation of CROWN over an l_inf input ball.

The model is an ``nn.Sequential`` of the layers in the ``layer`` package
(Linear, Conv2d, BatchNorm, Flatten, Identity/Dropout, AvgPool2d, MaxPool2d,
ReLU, LeakyReLU, Sigmoid, Tanh). The input point keeps the shape the model
takes without the batch dimension, for example ``(1, 28, 28)``; the
coefficients below are over the flattened values.

For every layer output ``z_i`` that feeds a non-affine layer (and for the
last one) we keep symbolic bounds
``coef_lo x + off_lo <= z_i <= coef_hi x + off_hi`` (``backsub``,
Algorithm 1) and concretize them over the ball ``||x - x0||_inf <= eps``
(``concretize``) with Hoelder's inequality.
'''

import math

import torch

from . import layer as ops

# -- Network -----------------------------------------------------------------


def from_sequential(model, input_shape=None):
    '''Layer descriptions of an ``nn.Sequential``.

    ``input_shape`` is the shape of one input without the batch dimension; it
    may be left out when the first layer is Linear.
    '''
    if not isinstance(model, torch.nn.Sequential):
        name = type(model).__name__
        raise TypeError(f'only nn.Sequential is supported, got {name}')
    modules = list(model)
    if not modules:
        raise TypeError('the network has no layers')
    shape = input_shape
    if shape is None:
        if not isinstance(modules[0], torch.nn.Linear):
            raise TypeError('input_shape is needed unless the first layer '
                            'is Linear')
        shape = (modules[0].in_features,)
    layers = []
    for module in modules:
        layers.append(ops.parse(module, shape))
        shape = layers[-1]['out_shape']
    return layers


def bounded_layers(layers):
    '''Indices of the layers whose output needs concrete bounds.

    These are the inputs of the non-affine layers, and the last layer.
    '''
    last = len(layers) - 1
    return [i for i in range(len(layers))
            if i == last or layers[i + 1]['op'] in ops.NONLINEAR]


# -- Symbolic bounds (Algorithm 1) -------------------------------------------

def backsub(layers, bounds, i, like, alphas=None):
    '''Symbolic bounds of the output of layer ``i`` over the input.

    Returns ``((coef_lo, off_lo), (coef_hi, off_hi))`` over the flattened
    input. ``bounds`` maps a layer index to the flat concrete bounds
    ``(l, u)`` of that layer's output; non-affine layers read the entry of
    the layer before them. ``like`` is a tensor with the dtype and device of
    the coefficients. ``alphas`` optionally maps a ReLU layer index to its
    alpha slopes (see ``alphacrown``); None is vanilla CROWN.
    '''
    alphas = alphas or {}
    size = math.prod(layers[i]['out_shape'])
    eye = torch.eye(size, dtype=like.dtype, device=like.device)
    lo = hi = (eye, torch.zeros_like(eye[0]))
    for j in range(i, -1, -1):
        lo, hi = ops.backward(layers[j], lo, hi, bounds.get(j - 1),
                              alphas.get(j))
    return lo, hi


# -- Concretization ----------------------------------------------------------

def concretize(coef, off, x0, eps, sign):
    '''Bound ``coef x + off`` row-wise over ``||x - x0||_inf <= eps``.

    Returns a lower bound for ``sign=+1`` and an upper bound for ``sign=-1``.
    '''
    signed = sign * coef
    lower = signed @ x0 - (eps * signed.abs()).sum(-1) + sign * off
    return sign * lower


# -- Splits and entry points -------------------------------------------------


def clamp_split(bound, split):
    '''Restrict ``bound = (l, u)`` to a ReLU phase split.

    ``split`` is a tensor in {-1, 0, +1} per neuron (inactive, free, active)
    or None. Sound inside the subdomain where the phases hold; a neuron with
    ``l > u`` afterwards means the subdomain is empty.
    '''
    if split is None:
        return bound
    lower, upper = bound
    return (torch.where(split > 0, lower.clamp(min=0), lower),
            torch.where(split < 0, upper.clamp(max=0), upper))


def split_effect(bound, split):
    '''Mask of the splits that actually changed ``bound = (l, u)``.

    A split on a neuron that was already stable the same way is a no-op:
    leaving it out gives bit-identical bounds in every later layer.
    '''
    lower, upper = bound
    return ((split > 0) & (lower < 0)) | ((split < 0) & (upper > 0))


def layer_bounds(layers, x0, eps, splits=None, core=False):
    '''Concrete bounds ``[(l, u)]`` of the layers in ``bounded_layers``.

    The bounds are flat tensors. ``splits`` maps a layer index of a ReLU or
    LeakyReLU to its phase split (see ``clamp_split``); None is the whole
    input region. With ``core=True`` the result is ``(bounds, effective)``
    where ``effective`` maps such a layer index to the mask of its splits
    that changed a bound (``split_effect``).
    '''
    splits = splits or {}
    flat = x0.reshape(-1)
    effective = {}
    bounds = {}
    for i in bounded_layers(layers):
        (coef_lo, off_lo), (coef_hi, off_hi) = backsub(layers, bounds, i, flat)
        bound = (concretize(coef_lo, off_lo, flat, eps, +1),
                 concretize(coef_hi, off_hi, flat, eps, -1))
        split = splits.get(i + 1)
        if core and split is not None:
            effective[i + 1] = split_effect(bound, split)
        bounds[i] = clamp_split(bound, split)
    return (list(bounds.values()), effective) if core else list(
        bounds.values())


def bounds(model, x0, eps):
    '''Concrete bounds ``[(l, u)]`` of every pre-activation of ``model``.

    ``x0`` has the shape of one model input (no batch dimension). The bounds
    cover the inputs of the non-affine layers and the network output, which
    is the last entry.
    '''
    return layer_bounds(from_sequential(model, x0.shape), x0, eps)


def margin_layers(model, input_shape, label):
    '''Layers of ``model`` followed by the margin specification.

    The appended Linear layer outputs ``f_label - f_k`` for every
    ``k != label``.
    '''
    layers = from_sequential(model, input_shape)
    dtype = next(model.parameters()).dtype
    size = math.prod(layers[-1]['out_shape'])
    eye = torch.eye(size, dtype=dtype)
    margin = (eye[label] - eye)[torch.arange(size) != label]
    return layers + [{'op': 'linear', 'weight': margin,
                      'bias': torch.zeros(size - 1, dtype=dtype),
                      'in_shape': (size,), 'out_shape': (size - 1,)}]


def margin_lower_bounds(model, x0, eps, label):
    # Lower bounds of ``f_label(x) - f_k(x)`` for every ``k != label``.
    layers = margin_layers(model, x0.shape, label)
    return layer_bounds(layers, x0, eps)[-1][0]


def verify(model, x0, eps, label):
    # True iff ``label`` is certified to be the argmax on the ball.
    return bool((margin_lower_bounds(model, x0, eps, label) > 0).all())
