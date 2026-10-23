'''Patch form of the coefficients, as in auto_LiRPA.

Starting from a conv-shaped layer output ``(Cs, Hs, Ws)``, the symbolic bound
of one start neuron over an earlier conv-shaped layer ``(C, H, W)`` only
involves a small window of that layer. The dense coefficients
``(Cs*Hs*Ws, C*H*W)`` are then stored as a dict of

- ``p``: ``(Cs, Hp, Wp, C, kh, kw)``, the window of every start neuron;
  ``Hp`` and ``Wp`` are 1 while the window is the same for every position
  (it broadcasts), else ``Hs`` and ``Ws``;
- ``start``: ``(Cs, Hs, Ws)``, ``shape``: ``(C, H, W)`` of the current layer;
- ``stride`` and ``padding``: ``(h, w)`` pairs. Start neuron
  ``(cs, hs, ws)`` reads the window whose top-left corner is at
  ``(hs * stride_h - padding_h, ws * stride_w - padding_w)`` in the layer;
  entries outside the layer lie in the zero padding and never count.

Rows are ordered ``(cs, hs, ws)`` like the dense rows. Every function here
returns new dicts and tensors; nothing is changed in place.
'''

import torch
import torch.nn.functional as functional


def pair(value):
    # An int or a pair as an ``(h, w)`` tuple.
    return (value, value) if isinstance(value, int) else tuple(value)


def is_patches(coef):
    return isinstance(coef, dict)


def identity(shape, like):
    # Patches of the identity over a conv-shaped layer: one channel window.
    channels = shape[0]
    eye = torch.eye(channels, dtype=like.dtype, device=like.device)
    return {'p': eye.reshape(channels, 1, 1, channels, 1, 1),
            'start': tuple(shape), 'shape': tuple(shape),
            'stride': (1, 1), 'padding': (0, 0)}


def windows(image, patches):
    '''Windows ``(Hs, Ws, C, kh, kw)`` of ``image`` ``(C, H, W)``.

    They have the geometry of ``patches`` and read zero in the padding.
    '''
    kh, kw = patches['p'].shape[-2:]
    (sh, sw), (ph, pw) = patches['stride'], patches['padding']
    padded = functional.pad(image, (pw, pw, ph, ph))
    out = padded.unfold(1, kh, sh).unfold(2, kw, sw).permute(1, 2, 0, 3, 4)
    return out[:patches['start'][1], :patches['start'][2]]


def masked(patches):
    # The windows with the entries in the padding set to zero.
    if patches['padding'] == (0, 0):
        return patches['p']
    p = patches['p']
    ones = torch.ones(patches['shape'], dtype=p.dtype, device=p.device)
    return p * windows(ones, patches)


def row_sum(values, patches):
    # Sum over each window, one value per start neuron (flat rows).
    return values.sum((3, 4, 5)).expand(*patches['start']).reshape(-1)


def to_dense(patches):
    # The dense coefficients ``(Cs*Hs*Ws, C*H*W)`` of the patches.
    cs, hs, ws = patches['start']
    c, h, w = patches['shape']
    (sh, sw), (ph, pw) = patches['stride'], patches['padding']
    p = patches['p']
    kh, kw = p.shape[-2:]
    device = p.device
    ys = (torch.arange(hs, device=device)[:, None] * sh
          + torch.arange(kh, device=device)[None, :] - ph)
    xs = (torch.arange(ws, device=device)[:, None] * sw
          + torch.arange(kw, device=device)[None, :] - pw)
    valid = (((ys >= 0) & (ys < h))[:, None, :, None]
             & ((xs >= 0) & (xs < w))[None, :, None, :])
    index = (ys.clamp(0, h - 1)[:, None, :, None] * w
             + xs.clamp(0, w - 1)[None, :, None, :])
    values = p.expand(cs, hs, ws, c, kh, kw) * valid[None, :, :, None]
    index = index.reshape(1, hs, ws, 1, kh * kw).expand(
        cs, hs, ws, c, kh * kw)
    dense = torch.zeros(cs, hs, ws, c, h * w, dtype=p.dtype, device=device)
    dense.scatter_add_(4, index, values.reshape(cs, hs, ws, c, kh * kw))
    return dense.reshape(cs * hs * ws, c * h * w)


def as_dense(coef):
    # Dense coefficients of either form.
    return to_dense(coef) if is_patches(coef) else coef


def substitute(patches, off, relax_same, relax_flip):
    '''Substitute a pointwise relaxation into patch coefficients.

    Same rule as ``activation.substitute``: each window entry multiplies one
    neuron of the layer, so the slopes and shifts are read through the same
    windows.
    '''
    (slope, shift), (slope_flip, shift_flip) = relax_same, relax_flip
    p = patches['p']
    pos, neg = p.clamp(min=0), p.clamp(max=0)

    def read(flat):
        return windows(flat.reshape(patches['shape']), patches)

    new = pos * read(slope) + neg * read(slope_flip)
    add = row_sum(pos * read(shift) + neg * read(shift_flip), patches)
    return {**patches, 'p': new}, off + add


def scale(patches, off, slope, shift):
    '''Substitute the pointwise affine map ``slope v + shift`` (exact).

    The same as ``substitute`` with equal lines, without splitting the window
    by sign, which keeps the gradient of the coefficients single.
    '''
    def read(flat):
        return windows(flat.reshape(patches['shape']), patches)

    add = row_sum(patches['p'] * read(shift), patches)
    return {**patches, 'p': patches['p'] * read(slope)}, off + add


def compose(patches, weight, stride, dilation, groups, padding, in_shape):
    '''Patches over the input of a convolution, from patches over its output.

    The new windows are the transposed convolution of the old ones with the
    kernel; kernel size, stride and padding of the patches are composed.
    '''
    window = masked(patches)
    cs, hp, wp, c, kh, kw = window.shape
    new = functional.conv_transpose2d(
        window.reshape(cs * hp * wp, c, kh, kw), weight, stride=stride,
        dilation=dilation, groups=groups)
    (sh, sw), (ph, pw) = patches['stride'], patches['padding']
    return {**patches, 'p': new.reshape(cs, hp, wp, -1, *new.shape[-2:]),
            'stride': (sh * stride[0], sw * stride[1]),
            'padding': (ph * stride[0] + padding[0],
                        pw * stride[1] + padding[1]),
            'shape': tuple(in_shape)}


def concretize(patches, off, x0, eps, sign):
    '''Bound the patch coefficients over ``||x - x0||_inf <= eps``.

    Returns a lower bound for ``sign=+1`` and an upper bound for ``sign=-1``,
    like ``crown.concretize``. ``x0`` is flat.
    '''
    image = x0.reshape(patches['shape'])
    signed = sign * patches['p']
    valid = windows(torch.ones_like(image), patches)
    center = (signed * windows(image, patches)).sum((3, 4, 5))
    radius = (signed.abs() * valid).sum((3, 4, 5))
    lower = (center - eps * radius).expand(*patches['start']).reshape(-1)
    return sign * (lower + sign * off)
