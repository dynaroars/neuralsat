'''Max pooling over non-overlapping windows, relaxed like auto_LiRPA.

Per window, with ``i`` the input of largest lower bound:

- lower line: ``y >= x_i``;
- upper line: ``y <= x_i`` when ``l_i`` is at least every other upper bound
  (the window is linear), else the constant ``max_j u_j``.

Each output neuron therefore reads one input: ``y ~ slope x[index] + shift``.
'''

import torch
import torch.nn.functional as functional

from . import activation, patches

TYPES = (torch.nn.MaxPool2d,)


def parse(module):
    # Layer description of an ``nn.MaxPool2d`` with non-overlapping windows.
    kernel = patches.pair(module.kernel_size)
    stride = patches.pair(module.stride)
    if (stride != kernel or patches.pair(module.padding) != (0, 0)
            or patches.pair(module.dilation) != (1, 1) or module.ceil_mode):
        raise TypeError('MaxPool2d needs stride == kernel, no padding, '
                        'no dilation and ceil_mode=False')
    return {'op': 'maxpool', 'kernel': kernel}


def forward(layer, x):
    return functional.max_pool2d(x, layer['kernel'])


def interval(layer, lower, upper):
    # Exact output interval of max pooling (monotone), flat in and out.
    shape = (1, *layer['in_shape'])
    return (forward(layer, lower.reshape(shape)).flatten(),
            forward(layer, upper.reshape(shape)).flatten())


def relaxation(layer, lower, upper):
    '''Per output neuron ``(index, lower line, upper line)``.

    ``index`` is the flat input position that the lines read.
    '''
    channels, height, width = layer['in_shape']
    kernel = layer['kernel']
    low = lower.reshape(1, channels, height, width)
    up = upper.reshape(1, channels, height, width)
    max_lower, argmax = functional.max_pool2d(low, kernel, return_indices=True)
    others = up.flatten(2).scatter(2, argmax.flatten(2), -torch.inf)
    max_others = functional.max_pool2d(others.view_as(up), kernel)
    linear = (max_lower >= max_others).flatten()
    max_upper = functional.max_pool2d(up, kernel).flatten()
    offset = torch.arange(channels, device=lower.device).view(
        1, channels, 1, 1) * height * width
    index = (argmax + offset).flatten()
    ones, zeros = torch.ones_like(max_upper), torch.zeros_like(max_upper)
    return (index, (ones, zeros),
            (linear.to(upper.dtype), torch.where(linear, zeros, max_upper)))


def substitute(coef, off, index, relax_same, relax_flip, size):
    # Substitute a window relaxation into ``coef v + off``; ``size`` inputs.
    (slope, shift), (slope_flip, shift_flip) = relax_same, relax_flip
    pos, neg = coef.clamp(min=0), coef.clamp(max=0)
    weights = pos * slope + neg * slope_flip
    new = torch.zeros(coef.shape[0], size, dtype=coef.dtype,
                      device=coef.device)
    new.scatter_add_(1, index.expand(coef.shape[0], -1), weights)
    return new, pos @ shift + neg @ shift_flip + off


def window_offsets(layer, index):
    # Position inside its window of the input read by each output neuron.
    channels, height, width = layer['in_shape']
    out_h, out_w = layer['out_shape'][1:]
    k_h, k_w = layer['kernel']
    spatial = index.reshape(channels, out_h, out_w) % (height * width)
    rows = torch.arange(out_h, device=index.device).view(1, -1, 1) * k_h
    cols = torch.arange(out_w, device=index.device).view(1, 1, -1) * k_w
    return spatial // width - rows, spatial % width - cols


def substitute_patches(coef, off, layer, index, relax_same, relax_flip):
    '''Substitute a window relaxation into patch coefficients.

    Each entry of a window over the pooled values moves to the input pixel it
    reads, so the windows grow by the pooling kernel.
    '''
    (slope, shift), (slope_flip, shift_flip) = relax_same, relax_flip
    p = coef['p']
    pos, neg = p.clamp(min=0), p.clamp(max=0)
    shape = layer['out_shape']
    k_h, k_w = layer['kernel']
    (start_c, start_h, start_w), (kh, kw) = coef['start'], p.shape[-2:]
    channels = shape[0]

    def read(flat):
        return patches.windows(flat.reshape(shape), coef)

    dy, dx = (patches.windows(offset, coef)
              for offset in window_offsets(layer, index))
    weights = pos * read(slope) + neg * read(slope_flip)
    off = off + patches.row_sum(pos * read(shift) + neg * read(shift_flip),
                                coef)
    where = ((torch.arange(kh, device=p.device).view(-1, 1) * k_h + dy)
             * (kw * k_w) + torch.arange(kw, device=p.device) * k_w + dx)
    full = (start_c, start_h, start_w, channels, kh * kw)
    new = torch.zeros(*full[:4], kh * k_h * kw * k_w, dtype=p.dtype,
                      device=p.device)
    new.scatter_add_(4, where.reshape(1, *full[1:]).expand(*full),
                     weights.expand(start_c, start_h, start_w, channels, kh,
                                    kw).reshape(*full))
    (sh, sw), (ph, pw) = coef['stride'], coef['padding']
    return {**coef, 'p': new.reshape(*full[:4], kh * k_h, kw * k_w),
            'stride': (sh * k_h, sw * k_w), 'padding': (ph * k_h, pw * k_w),
            'shape': tuple(layer['in_shape'])}, off


def zonotope(layer, center, gens, lower, upper):
    '''Zonotope of the max pool output.

    Per window, ``y`` is at least the input ``x_i`` of the largest lower
    bound, and at most the largest upper bound ``U``, which is at most
    ``x_i + (U - l_i)``. So ``y`` lies in a strip of width ``U - l_i`` above
    ``x_i``; the width is 0 when the window is linear.
    '''
    index, _, (slope, shift) = relaxation(layer, lower, upper)
    width = torch.where(slope == 1, torch.zeros_like(shift),
                        shift - lower[index])
    return (center[index] + width / 2,
            torch.cat([gens[:, index],
                       activation.new_generators(width / 2)], 0))


def backward(layer, lo, hi, pre_bounds, alphas=None):
    '''Back-substitute the max-pool relaxation into both symbolic bounds.

    ``pre_bounds = (l, u)`` are the flat concrete bounds of the pool input.
    '''
    index, relax_lo, relax_hi = relaxation(layer, *pre_bounds)
    if patches.is_patches(lo[0]):
        return (substitute_patches(*lo, layer, index, relax_lo, relax_hi),
                substitute_patches(*hi, layer, index, relax_hi, relax_lo))
    size = int(torch.tensor(layer['in_shape']).prod())
    return (substitute(*lo, index, relax_lo, relax_hi, size),
            substitute(*hi, index, relax_hi, relax_lo, size))
