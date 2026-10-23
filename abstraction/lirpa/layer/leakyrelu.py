# Leaky ReLU ``a = max(z, slope z)``, relaxed like ReLU (auto_LiRPA rules).

import torch
import torch.nn.functional as functional

from . import activation

TYPES = (torch.nn.LeakyReLU,)


def parse(module):
    # Layer description of an ``nn.LeakyReLU`` with a slope in [0, 1].
    if not 0 <= module.negative_slope <= 1:
        raise TypeError('LeakyReLU needs a negative slope in [0, 1]')
    return {'op': 'leakyrelu', 'slope': module.negative_slope}


def forward(layer, x):
    return functional.leaky_relu(x, layer['slope'])


def interval(layer, lower, upper):
    # Exact output interval of the monotone leaky relu.
    return forward(layer, lower), forward(layer, upper)


def relax(slope, lower, upper, alpha=None):
    '''Linear relaxations of leaky relu on ``[lower, upper]``.

    Same shape of result as ``relu.relax``. Unstable neurons use the chord as
    upper line; the lower line has slope ``alpha`` clipped to [slope, 1], or
    1 when the chord slope exceeds 0.5 and ``slope`` otherwise.
    '''
    active, inactive = lower >= 0, upper <= 0
    unstable = ~(active | inactive)
    gap = torch.where(unstable, upper - lower, torch.ones_like(upper))
    stable = torch.where(active, 1.0, slope).to(upper.dtype)
    chord = (upper - slope * lower) / gap
    slope_hi = torch.where(unstable, chord, stable)
    shift_hi = torch.where(unstable, slope * lower - chord * lower,
                           torch.zeros_like(upper))
    if alpha is None:
        alpha = torch.where(chord > 0.5, 1.0, slope).to(upper.dtype)
    slope_lo = torch.where(unstable, alpha.clamp(slope, 1.0), stable)
    return (slope_lo, torch.zeros_like(upper)), (slope_hi, shift_hi)


def zonotope(layer, center, gens, lower, upper):
    '''Zonotope of the leaky relu output: the chord and its parallel through 0.

    The function minus ``chord_slope * z`` is smallest at the kink, where it
    is 0, so the strip runs from 0 up to the chord's intercept.
    '''
    _, (slope, shift) = relax(layer['slope'], lower, upper)
    return activation.parallelogram(center, gens, slope, 0.0, shift)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the relaxation of leaky relu into both bounds.
    def relax_layer(lower, upper, alpha):
        return relax(layer['slope'], lower, upper, alpha)

    return activation.backward(relax_layer, lo, hi, pre_bounds, alphas)
