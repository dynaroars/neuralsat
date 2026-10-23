'''Dense back-substitution through an affine layer given as a function.

The coefficients are matrices ``(rows, size)`` over the flattened layer
output. For an affine ``f(x) = J x + c`` the new coefficients are
``coef J`` and the new offset ``coef c + off``. ``J`` is never built: each row
of ``coef`` goes through the vector-Jacobian product of ``f``.
'''

import torch

from . import patches


def backward(fn, in_shape, out_shape, lo, hi):
    '''Back-substitute the affine ``fn`` into both bounds.

    ``fn`` maps a batch ``(1, *in_shape)`` to ``(1, *out_shape)``.
    '''
    lo = (patches.as_dense(lo[0]), lo[1])
    hi = (patches.as_dense(hi[0]), hi[1])
    ref = lo[0]
    zero = torch.zeros(1, *in_shape, dtype=ref.dtype, device=ref.device)
    const = fn(zero).reshape(-1)
    _, pull = torch.func.vjp(fn, zero)

    def substitute(coef, off):
        rows = coef.reshape(coef.shape[0], 1, *out_shape)
        new = torch.func.vmap(lambda row: pull(row)[0])(rows)
        return new.reshape(coef.shape[0], -1), coef @ const + off

    return substitute(*lo), substitute(*hi)


def zonotope(fn, in_shape, center, gens):
    '''Push a flat zonotope ``(center, gens)`` through the affine ``fn``.

    The center goes through ``fn``; the generators go through its linear
    part, ``fn(g) - fn(0)``. ``fn`` maps ``(rows, *in_shape)`` to a batch.
    '''
    zero = torch.zeros(1, *in_shape, dtype=center.dtype, device=center.device)
    base = fn(zero)
    new_center = fn(center.reshape(1, *in_shape)).reshape(-1)
    shifted = fn(gens.reshape(gens.shape[0], *in_shape)) - base
    return new_center, shifted.reshape(gens.shape[0], -1)
