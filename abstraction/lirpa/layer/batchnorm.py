# Batch normalization in evaluation mode: ``z = scale a + shift`` per channel.

import torch

TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)


def parse(module):
    # Fold the running statistics and affine parameters into scale and shift.
    if module.running_mean is None:
        raise TypeError('BatchNorm needs running statistics')
    scale = torch.rsqrt(module.running_var + module.eps)
    if module.weight is not None:
        scale = scale * module.weight
    shift = -module.running_mean * scale
    if module.bias is not None:
        shift = shift + module.bias
    return {'op': 'batchnorm', 'scale': scale.detach(),
            'shift': shift.detach()}


def channels(layer, tensor):
    # View a per-channel vector so that it broadcasts over ``(rows, C, ...)``.
    rest = len(layer['in_shape']) - 1
    return tensor.reshape(-1, *([1] * rest))


def forward(layer, x):
    # Concrete evaluation on a ``(rows, C, ...)`` batch.
    return x * channels(layer, layer['scale']) + channels(
        layer, layer['shift'])


def substitute(coef, off, layer):
    # Substitute ``v = scale w + shift`` into ``coef v + off``.
    rows = coef.reshape(coef.shape[0], *layer['out_shape'])
    scale = channels(layer, layer['scale'])
    shift = channels(layer, layer['shift'])
    return ((rows * scale).reshape(coef.shape[0], -1),
            (rows * shift).reshape(coef.shape[0], -1).sum(-1) + off)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the per-channel affine map (exact).
    return substitute(*lo, layer), substitute(*hi, layer)
