'''Substitution shared by the pointwise activation layers.

An activation is relaxed per neuron by a lower and an upper line,
``(slope, shift)`` each, from the concrete bounds of its input.
'''


def substitute(coef, off, relax_same, relax_flip):
    '''Substitute an activation relaxation into ``coef v + off``.

    Non-negative coefficients take ``relax_same``, negative ones
    ``relax_flip``. Pass ``(lower, upper)`` to bound from below and
    ``(upper, lower)`` to bound from above.
    '''
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
