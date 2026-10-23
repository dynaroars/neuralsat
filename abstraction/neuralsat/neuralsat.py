'''Neuralsat abstraction for sequential networks.

Like CROWN (see ``abstraction.lirpa.crown``) but the input region is an l_inf
ball cut by convex constraints ``h_j(x) <= 0``. The symbolic bounds come from
the same backward substitution; each one is concretized over the constrained
region by a projected primal-dual method (``concretize``).

The model is an ``nn.Sequential`` of the layers in ``lirpa.layer``; the
input keeps the model's shape, the constraints act on its flattened values.
'''

from abstraction.lirpa import layer as ops
from abstraction.lirpa.crown import (
    backsub,
    bounded_layers,
    clamp_split,
    from_sequential,
    margin_layers,
    split_effect,
    use_interval,
)
from abstraction.lirpa.layer.patches import as_dense
from abstraction.neuralsat.concretize import concretize


def layer_bounds(layers, x0, eps, constraints=(), splits=None, core=False,
                 patches=True, **kwargs):
    '''Concrete bounds ``[(l, u)]`` of the layers in ``bounded_layers``.

    ``splits`` maps a ReLU or LeakyReLU layer index to its phase split (see
    ``crown.clamp_split``); None is the whole input region. With ``core=True``
    the result is ``(bounds, effective)``, see ``crown.layer_bounds``. The
    constraints act on the flattened input. ``patches`` keeps conv-shaped
    coefficients in patch form until the input, where they are made dense.
    '''
    splits = splits or {}
    flat = x0.reshape(-1)
    effective = {}
    bounds = {}
    for i in bounded_layers(layers):
        if use_interval(layers, i):
            bound = ops.interval(layers[i], *bounds[i - 1])
        else:
            (coef_lo, off_lo), (coef_hi, off_hi) = backsub(
                layers, bounds, i, flat, patches=patches)
            coef_lo, coef_hi = as_dense(coef_lo), as_dense(coef_hi)
            bound = (concretize(coef_lo, off_lo, flat, eps, constraints, +1,
                                **kwargs),
                     concretize(coef_hi, off_hi, flat, eps, constraints, -1,
                                **kwargs))
        split = splits.get(i + 1)
        if core and split is not None:
            effective[i + 1] = split_effect(bound, split)
        bounds[i] = clamp_split(bound, split)
    return (list(bounds.values()), effective) if core else list(
        bounds.values())


def bounds(model, x0, eps, constraints=(), **kwargs):
    '''Concrete bounds ``[(l, u)]`` of every pre-activation of ``model``.

    ``x0`` has the shape of one model input; the last entry is the network
    output. ``kwargs`` are passed to ``concretize``.
    '''
    return layer_bounds(from_sequential(model, x0.shape), x0, eps,
                        constraints, **kwargs)


def margin_lower_bounds(model, x0, eps, label, constraints=(), **kwargs):
    # Lower bounds of f_label(x) - f_k(x) for every k != label.
    layers = margin_layers(model, x0.shape, label)
    return layer_bounds(layers, x0, eps, constraints, **kwargs)[-1][0]


def verify(model, x0, eps, label, constraints=(), **kwargs):
    # True iff label is certified to be the argmax on the feasible set.
    lower = margin_lower_bounds(model, x0, eps, label, constraints, **kwargs)
    return bool((lower > 0).all())
