'''Dense back-substitution through an affine layer given as a function.

The coefficients are matrices ``(rows, size)`` over the flattened layer
output. For an affine ``f(x) = J x + c`` the new coefficients are
``coef J`` and the new offset ``coef c + off``. ``J`` is never built: each row
of ``coef`` goes through the vector-Jacobian product of ``f``.
'''

import torch


def backward(fn, in_shape, out_shape, lo, hi):
    '''Back-substitute the affine ``fn`` into both bounds.

    ``fn`` maps a batch ``(1, *in_shape)`` to ``(1, *out_shape)``.
    '''
    ref = lo[0]
    zero = torch.zeros(1, *in_shape, dtype=ref.dtype, device=ref.device)
    const = fn(zero).reshape(-1)
    _, pull = torch.func.vjp(fn, zero)

    def substitute(coef, off):
        rows = coef.reshape(coef.shape[0], 1, *out_shape)
        new = torch.func.vmap(lambda row: pull(row)[0])(rows)
        return new.reshape(coef.shape[0], -1), coef @ const + off

    return substitute(*lo), substitute(*hi)
