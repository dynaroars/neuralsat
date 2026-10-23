'''Pieces shared by the searches: bounds checks, branching, falsification.

A search works on a phase assignment ``{(j, k): sign}`` where ``j`` is the
index of a ReLU or LeakyReLU layer, ``k`` a flat neuron index, and ``sign``
+1 (active) or -1 (inactive). The abstraction (``bound_fn``) bounds the
network on the subdomain given by an assignment and plays the role of the
theory solver. ``bound_fn(layers, x0, eps, splits=...)`` returns the list of
bounds of ``lirpa.crown.bounded_layers``; ``lirpa.crown.layer_bounds`` is the
default.
'''

import math

import torch

from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops

TOL = 1e-9


def split_tensors(layers, assignment):
    # Per ReLU layer tensor in {-1, 0, +1} for the abstraction's ``splits``.
    splits = {}
    for (j, k), sign in assignment.items():
        if j not in splits:
            width = math.prod(layers[j]['in_shape'])
            splits[j] = torch.zeros(width, dtype=torch.int8)
        splits[j][k] = sign
    return splits


def margin_proven(bounds):
    # The margin of the folded network is positive on the whole subdomain.
    return bool((bounds[-1][0] > 0).all())


def first_infeasible(bounds):
    # Position of the first layer with an empty neuron interval, or None.
    for position, (lower, upper) in enumerate(bounds):
        if bool((lower > upper + TOL).any()):
            return position
    return None


def infeasible(bounds):
    # Some neuron has an empty interval: the subdomain has no point.
    return first_infeasible(bounds) is not None


def proven(bounds):
    # No counterexample can exist in the subdomain.
    return infeasible(bounds) or margin_proven(bounds)


def pick_neuron(layers, bounds):
    '''Unstable neuron with the largest triangle-relaxation area, or None.

    Only ReLU and LeakyReLU layers can be split. The input bounds of layer
    ``j`` are the entry of layer ``j - 1`` in ``bounded_layers``.
    '''
    positions = {i: p for p, i in enumerate(crown.bounded_layers(layers))}
    best, best_score = None, 0.0
    for j, layer in enumerate(layers):
        if layer['op'] not in ops.SPLITTABLE:
            continue
        lower, upper = bounds[positions[j - 1]]
        unstable = (lower < 0) & (upper > 0)
        score = torch.where(unstable, -lower * upper / (upper - lower),
                            torch.zeros_like(lower))
        k = int(score.argmax())
        if score[k] > best_score:
            best, best_score = (j, k), float(score[k])
    return best


def network(layers, x):
    # Concrete output of the network on a single point ``x``.
    x = x.unsqueeze(0)
    for layer in layers:
        x = ops.forward(layer, x)
    return x[0]


def falsify(layers, x0, eps, label, constraints=(), steps=20):
    '''PGD for a point in the ball with ``f_label <= max_k f_k``, or None.

    A point only counts if it satisfies every constraint ``h(x) <= 0``.
    '''
    x = x0.clone()
    for _ in range(steps + 1):
        x.requires_grad_(True)
        out = network(layers, x)
        margin = out[label] - torch.cat([out[:label], out[label + 1:]]).max()
        flat = x.detach().reshape(-1)
        if margin <= 0 and all(h(flat) <= 0 for h in constraints):
            return x.detach()
        grad, = torch.autograd.grad(margin, x)
        x = (x.detach() - 2 * eps / steps * grad.sign())
        x = torch.minimum(torch.maximum(x, x0 - eps), x0 + eps)
    return None
