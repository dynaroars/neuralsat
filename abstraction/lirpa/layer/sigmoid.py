# Sigmoid layer ``a = sigmoid(z)`` (S-shaped, see ``sshaped``).

import torch

from . import activation, sshaped

TYPES = (torch.nn.Sigmoid,)


def parse(module):
    return {'op': 'sigmoid'}


def forward(layer, x):
    return torch.sigmoid(x)


def derivative(x):
    value = torch.sigmoid(x)
    return value * (1 - value)


def relax(lower, upper, alpha=None):
    # Linear relaxations of sigmoid on ``[lower, upper]``.
    return sshaped.relax(torch.sigmoid, derivative, lower, upper)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the relaxation of sigmoid into both symbolic bounds.
    return activation.backward(relax, lo, hi, pre_bounds, alphas)
