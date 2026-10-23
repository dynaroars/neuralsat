# ReLU layer ``a = relu(z)``, relaxed linearly from the bounds of ``z``.

import torch

from . import activation

TYPES = (torch.nn.ReLU,)
substitute = activation.substitute


def parse(module):
    # Layer description of an ``nn.ReLU``.
    return {'op': 'relu'}


def forward(layer, x):
    # Concrete evaluation.
    return torch.relu(x)


def relax(lower, upper, alpha=None):
    '''Linear relaxations of relu on ``[lower, upper]``.

    Returns ``((d_lo, e_lo), (d_hi, e_hi))`` such that
    ``d_lo z + e_lo <= relu(z) <= d_hi z + e_hi``. For unstable neurons the
    lower slope is ``alpha`` clipped to [0, 1] (alpha-CROWN), or the CROWN
    heuristic (1 if ``upper > -lower`` else 0) when ``alpha`` is None.
    '''
    active, inactive = lower >= 0, upper <= 0
    unstable = ~(active | inactive)
    ones, zeros = torch.ones_like(upper), torch.zeros_like(upper)
    gap = torch.where(unstable, upper - lower, ones)
    slope_hi = torch.where(unstable, upper / gap, active.to(upper.dtype))
    shift_hi = torch.where(unstable, -lower * upper / gap, zeros)
    if alpha is None:
        alpha = (upper > -lower).to(upper.dtype)
    slope_lo = torch.where(unstable, alpha.clamp(0.0, 1.0),
                           active.to(upper.dtype))
    return (slope_lo, zeros), (slope_hi, shift_hi)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the relaxation of relu into both symbolic bounds.
    return activation.backward(relax, lo, hi, pre_bounds, alphas)
