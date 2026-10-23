'''Max pooling over non-overlapping windows, relaxed like auto_LiRPA.

Per window, with ``i`` the input of largest lower bound:

- lower line: ``y >= x_i``;
- upper line: ``y <= x_i`` when ``l_i`` is at least every other upper bound
  (the window is linear), else the constant ``max_j u_j``.

Each output neuron therefore reads one input: ``y ~ slope x[index] + shift``.
'''

import torch
import torch.nn.functional as functional

TYPES = (torch.nn.MaxPool2d,)


def pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def parse(module):
    # Layer description of an ``nn.MaxPool2d`` with non-overlapping windows.
    kernel, stride = pair(module.kernel_size), pair(module.stride)
    if (stride != kernel or pair(module.padding) != (0, 0)
            or pair(module.dilation) != (1, 1) or module.ceil_mode):
        raise TypeError('MaxPool2d needs stride == kernel, no padding, '
                        'no dilation and ceil_mode=False')
    return {'op': 'maxpool', 'kernel': kernel}


def forward(layer, x):
    return functional.max_pool2d(x, layer['kernel'])


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


def backward(layer, lo, hi, pre_bounds, alphas=None):
    '''Back-substitute the max-pool relaxation into both symbolic bounds.

    ``pre_bounds = (l, u)`` are the flat concrete bounds of the pool input.
    '''
    index, relax_lo, relax_hi = relaxation(layer, *pre_bounds)
    size = int(torch.tensor(layer['in_shape']).prod())
    return (substitute(*lo, index, relax_lo, relax_hi, size),
            substitute(*hi, index, relax_hi, relax_lo, size))
