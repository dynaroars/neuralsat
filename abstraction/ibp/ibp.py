'''Interval bound propagation (IBP) for sequential networks.

The box of every layer is computed from the box of the layer before: an
affine layer maps the center normally and the radius with absolute weights,
and a non-affine layer gives the exact image of the interval
(``layer.interval``). It is the cheapest abstraction and the loosest: it
forgets how the neurons depend on each other.

The model is an ``nn.Sequential`` of the layers in ``lirpa.layer``; the input
keeps the model's shape, the bounds are flat. The interface is the one of
``lirpa.crown``, so the searches can use ``layer_bounds`` as ``bound_fn``.
'''

from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops


def step(layer, lower, upper):
    # The box of one layer's output from the box of its input.
    if layer['op'] in ops.NONLINEAR:
        return ops.interval(layer, lower, upper)
    shape = (1, *layer['in_shape'])
    center, radius = (lower + upper) / 2, (upper - lower) / 2
    center = ops.forward(layer, center.reshape(shape))
    radius = ops.forward(ops.absolute(layer), radius.reshape(shape))
    return (center - radius).reshape(-1), (center + radius).reshape(-1)


def layer_bounds(layers, x0, eps, splits=None, core=False):
    '''Concrete bounds ``[(l, u)]`` of the layers in ``bounded_layers``.

    ``splits`` and ``core`` work as in ``crown.layer_bounds``: a split clamps
    the bounds that feed the split layer, and the clamped bounds go on to the
    next layers.
    '''
    splits = splits or {}
    lower, upper = (x0 - eps).reshape(-1), (x0 + eps).reshape(-1)
    bounded = crown.bounded_layers(layers)
    effective, bounds = {}, {}
    for i, layer in enumerate(layers):
        lower, upper = step(layer, lower, upper)
        if i in bounded:
            split = splits.get(i + 1)
            if core and split is not None:
                effective[i + 1] = crown.split_effect((lower, upper), split)
            lower, upper = crown.clamp_split((lower, upper), split)
            bounds[i] = (lower, upper)
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
