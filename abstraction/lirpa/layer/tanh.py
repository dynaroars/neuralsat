# Tanh layer ``a = tanh(z)`` (S-shaped, see ``sshaped``).

import torch

from . import activation, sshaped

TYPES = (torch.nn.Tanh,)


def parse(module):
    return {'op': 'tanh'}


def forward(layer, x):
    return torch.tanh(x)


def interval(layer, lower, upper):
    # Exact output interval of the monotone tanh.
    return torch.tanh(lower), torch.tanh(upper)


def derivative(x):
    return 1 - torch.tanh(x) ** 2


def relax(lower, upper, alpha=None):
    # Linear relaxations of tanh on ``[lower, upper]``.
    return sshaped.relax(torch.tanh, derivative, lower, upper)


def zonotope(layer, center, gens, lower, upper):
    # Zonotope of the tanh output.
    return sshaped.zonotope(torch.tanh, derivative, center, gens, lower,
                            upper)


def backward(layer, lo, hi, pre_bounds, alphas=None):
    # Back-substitute the relaxation of tanh into both symbolic bounds.
    return activation.backward(relax, lo, hi, pre_bounds, alphas)
