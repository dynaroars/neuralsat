# Identity layers: ``nn.Identity`` and ``nn.Dropout`` (evaluation mode).

import torch

TYPES = (torch.nn.Identity, torch.nn.Dropout)


def parse(module):
    return {'op': 'identity'}


def forward(layer, x):
    return x


def backward(layer, lo, hi, pre_bounds, alphas=None):
    return lo, hi
