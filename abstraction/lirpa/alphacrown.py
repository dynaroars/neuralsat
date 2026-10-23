'''alpha-CROWN bound propagation for FC + ReLU nets.

Same backward substitution as ``crown`` but the lower relaxation slope of
every unstable ReLU is a free parameter in [0, 1] (alpha). For each Linear
layer, alphas of all earlier ReLUs are tuned by Adam to tighten that layer's
bounds, with the bounds of earlier layers already tightened. Every alpha in
[0, 1] gives a sound relaxation, so any iterate is sound; the best bound seen
(starting from vanilla CROWN) is kept.

The lower and the upper symbolic bound have separate alphas, shared by all
specification rows.
'''

import torch

from abstraction.lirpa import layer as ops
from abstraction.lirpa.crown import (
    backsub,
    bounded_layers,
    clamp_split,
    concretize,
    from_sequential,
    margin_layers,
    split_effect,
    use_interval,
)


def init_alphas(layers, bounds, i):
    # Alphas of every ReLU before layer i, set to the CROWN heuristic slope.
    alphas = {}
    for j in range(i):
        if layers[j]['op'] == 'relu':
            lower, upper = bounds[j - 1]
            slope = (upper > -lower).to(upper.dtype)
            alphas[j] = {'lo': slope.clone().requires_grad_(True),
                         'hi': slope.clone().requires_grad_(True)}
    return alphas


def evaluate(layers, bounds, i, x0, eps, alphas, patches=True):
    # Concrete bounds of layer i for the given alphas.
    (coef_lo, off_lo), (coef_hi, off_hi) = backsub(
        layers, bounds, i, x0, alphas, patches)
    return (concretize(coef_lo, off_lo, x0, eps, +1),
            concretize(coef_hi, off_hi, x0, eps, -1))


def tune_layer(layers, bounds, i, x0, eps, steps, lr, patches=True):
    '''Tightest bounds of layer ``i`` found by tuning alphas with Adam.'''
    if use_interval(layers, i):
        return ops.interval(layers[i], *bounds[i - 1])
    alphas = init_alphas(layers, bounds, i)
    params = [a for pair in alphas.values() for a in pair.values()]
    best_lower, best_upper = evaluate(
        layers, bounds, i, x0, eps, alphas, patches)
    best_lower, best_upper = best_lower.detach(), best_upper.detach()
    if not params:
        return best_lower, best_upper
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        lower, upper = evaluate(
            layers, bounds, i, x0, eps, alphas, patches)
        best_lower = torch.maximum(best_lower, lower.detach())
        best_upper = torch.minimum(best_upper, upper.detach())
        optimizer.zero_grad()
        (upper - lower).sum().backward()
        optimizer.step()
        with torch.no_grad():
            for a in params:
                a.clamp_(0.0, 1.0)
    with torch.no_grad():
        lower, upper = evaluate(
            layers, bounds, i, x0, eps, alphas, patches)
    return torch.maximum(best_lower, lower), torch.minimum(best_upper, upper)


def layer_bounds(layers, x0, eps, steps=20, lr=0.1, splits=None, core=False,
                 patches=True):
    '''Concrete bounds ``[(l, u)]`` of the layers in ``bounded_layers``.

    ``splits`` maps a ReLU or LeakyReLU layer index to its phase split (see
    ``crown.clamp_split``); None is the whole input region. With ``core=True``
    the result is ``(bounds, effective)``, see ``crown.layer_bounds``.
    '''
    splits = splits or {}
    flat = x0.reshape(-1)
    effective = {}
    bounds = {}
    for i in bounded_layers(layers):
        bound = tune_layer(layers, bounds, i, flat, eps, steps, lr, patches)
        split = splits.get(i + 1)
        if core and split is not None:
            effective[i + 1] = split_effect(bound, split)
        bounds[i] = clamp_split(bound, split)
    return (list(bounds.values()), effective) if core else list(
        bounds.values())


def bounds(model, x0, eps, steps=20, lr=0.1):
    '''Concrete bounds ``[(l, u)]`` of every pre-activation of ``model``.

    ``x0`` has the shape of one model input. Only ReLU slopes are tuned;
    ``steps`` and ``lr`` configure Adam on them. The last entry is the output.
    '''
    return layer_bounds(from_sequential(model, x0.shape), x0, eps, steps, lr)


def margin_lower_bounds(model, x0, eps, label, steps=20, lr=0.1):
    # Lower bounds of ``f_label(x) - f_k(x)`` for every ``k != label``.
    layers = margin_layers(model, x0.shape, label)
    return layer_bounds(layers, x0, eps, steps, lr)[-1][0]


def verify(model, x0, eps, label, steps=20, lr=0.1):
    # True iff ``label`` is certified to be the argmax on the ball.
    lower = margin_lower_bounds(model, x0, eps, label, steps, lr)
    return bool((lower > 0).all())
