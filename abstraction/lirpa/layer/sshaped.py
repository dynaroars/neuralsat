'''Linear relaxation of an S-shaped function (sigmoid, tanh) on ``[l, u]``.

Follows auto_LiRPA (``BoundSShaped``, vanilla CROWN): the function is convex
for ``x < 0`` and concave for ``x > 0``.

- ``u <= 0``: chord above, tangent at the midpoint below.
- ``l >= 0``: chord below, tangent at the midpoint above.
- ``l < 0 < u``: the chord when it is a valid bound, otherwise the tangent
  from the endpoint (``tangent_points``), and the midpoint tangent when that
  point lies beyond the middle (auto_LiRPA's smoothness rule).

Where auto_LiRPA reads the tangent points from a table with step 0.01, they
are found here by bisection. Both keep a point on the safe side, so the
result is sound; the lines can differ in the third decimal.
'''

import torch

ITERATIONS = 60


def double_until(holds, point):
    # Double ``point`` until ``holds(point)`` (elementwise); the first true.
    for _ in range(ITERATIONS):
        point = torch.where(holds(point), point, point * 2)
    return point


def tangent_points(func, dfunc, lower, upper):
    '''Tangent points ``(d_lower, d_upper)`` for intervals with ``l < 0 < u``.

    ``d_lower <= 0`` is the point closest to 0 whose tangent stays below
    ``func(upper)``; ``d_upper >= 0`` is the point closest to 0 whose tangent
    stays above ``func(lower)``. The returned points are on the safe side.
    '''
    def below_upper(d):
        return dfunc(d) * (upper - d) + func(d) <= func(upper)

    def above_lower(d):
        return dfunc(d) * (lower - d) + func(d) >= func(lower)

    far = double_until(below_upper, -torch.ones_like(upper))
    near = torch.zeros_like(upper)
    for _ in range(ITERATIONS):
        mid = (far + near) / 2
        ok = below_upper(mid)
        far = torch.where(ok, mid, far)
        near = torch.where(ok, near, mid)
    d_lower = far

    far = double_until(above_lower, torch.ones_like(upper))
    near = torch.zeros_like(upper)
    for _ in range(ITERATIONS):
        mid = (far + near) / 2
        ok = above_lower(mid)
        far = torch.where(ok, mid, far)
        near = torch.where(ok, near, mid)
    return d_lower, far


def line(slope, x0, y0):
    # The line through ``(x0, y0)`` as ``(slope, shift)``.
    return slope, y0 - slope * x0


def select(base, *overrides):
    # Start from the line ``base``; each ``(mask, line)`` after it overrides.
    slope, shift = base
    for mask, (k, b) in overrides:
        slope, shift = torch.where(mask, k, slope), torch.where(
            mask, b, shift)
    return slope, shift


def relax(func, dfunc, lower, upper, alpha=None):
    '''Linear relaxations ``((d_lo, e_lo), (d_hi, e_hi))`` on ``[l, u]``.

    ``alpha`` is accepted so the function can be used like ``relu.relax``; the
    S-shaped relaxations are not tuned.
    '''
    neg, pos = upper <= 0, lower >= 0
    both = ~(neg | pos)
    y_l, y_u = func(lower), func(upper)
    k_direct = (y_u - y_l) / (upper - lower).clamp(min=1e-8)
    mid = (lower + upper) / 2
    k_mid, y_mid = dfunc(mid), func(mid)
    safe_l = torch.where(both, lower, -torch.ones_like(lower))
    safe_u = torch.where(both, upper, torch.ones_like(upper))
    d_lower, d_upper = tangent_points(func, dfunc, safe_l, safe_u)
    direct_lower = both & (k_direct < dfunc(lower))
    direct_upper = both & (k_direct < dfunc(upper))

    chord = line(k_direct, lower, y_l)
    middle = line(k_mid, mid, y_mid)
    tan_lower = line(dfunc(d_lower), d_lower, func(d_lower))
    tan_upper = line(dfunc(d_upper), d_upper, func(d_upper))

    lower_line = select(
        chord, (neg, middle), (both, tan_lower), (direct_lower, chord),
        (both & (d_lower >= mid), middle))
    upper_line = select(
        chord, (pos, middle), (both, tan_upper), (direct_upper, chord),
        (both & (d_upper < mid), middle))
    return lower_line, upper_line
