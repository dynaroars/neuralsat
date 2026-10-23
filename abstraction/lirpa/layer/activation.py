'''Substitution shared by the pointwise activation layers.

An activation is relaxed per neuron by a lower and an upper line,
``(slope, shift)`` each, from the concrete bounds of its input. The
coefficients are dense matrices or patches (see ``patches``).
'''

import torch

from . import patches


def substitute(coef, off, relax_same, relax_flip):
    '''Substitute an activation relaxation into ``coef v + off``.

    Non-negative coefficients take ``relax_same``, negative ones
    ``relax_flip``. Pass ``(lower, upper)`` to bound from below and
    ``(upper, lower)`` to bound from above.
    '''
    if patches.is_patches(coef):
        return patches.substitute(coef, off, relax_same, relax_flip)
    (slope, shift), (slope_flip, shift_flip) = relax_same, relax_flip
    pos, neg = coef.clamp(min=0), coef.clamp(max=0)
    return (pos * slope + neg * slope_flip,
            pos @ shift + neg @ shift_flip + off)


def backward(relax, lo, hi, pre_bounds, alphas=None):
    '''Back-substitute ``relax(lower, upper, alpha)`` into both bounds.

    ``pre_bounds = (l, u)`` are the concrete bounds of the activation input.
    ``alphas = {'lo': a, 'hi': a}`` are the lower slopes used for the lower
    and the upper symbolic bound (alpha-CROWN); None means vanilla CROWN.
    '''
    alpha_lo, alpha_hi = (None, None) if alphas is None else (
        alphas['lo'], alphas['hi'])
    relax_lo, relax_hi = relax(*pre_bounds, alpha_lo)
    relax_flip, relax_up = relax(*pre_bounds, alpha_hi)
    return (substitute(*lo, relax_lo, relax_hi),
            substitute(*hi, relax_up, relax_flip))


def new_generators(half):
    # One generator per neuron whose half-width is positive: ``(m, n)``.
    index = (half > 0).nonzero().flatten()
    rows = torch.zeros(len(index), half.numel(), dtype=half.dtype,
                       device=half.device)
    rows[torch.arange(len(index), device=half.device), index] = half[index]
    return rows


def parallelogram(center, gens, slope, low, high):
    '''Zonotope of ``y`` with ``slope z + low <= y <= slope z + high``.

    ``z`` is the zonotope ``center + sum_k eps_k gens[k]`` over the flat
    neurons. Every neuron whose strip has a width gets one new generator, so
    the result is ``y = slope z + (low + high) / 2 + half eps_new``.
    '''
    half = ((high - low) / 2).clamp(min=0)
    new_center = slope * center + (low + high) / 2
    return new_center, torch.cat([slope * gens, new_generators(half)], 0)
