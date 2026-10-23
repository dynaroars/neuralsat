'''Instances of the experiment: random FC + ReLU net, point, radius, region.

An instance is ``(model, x0, eps, label, constraints)``: is ``label`` still
the argmax on the l_inf ball of radius ``eps`` around ``x0`` cut by the
convex ``constraints`` (a halfspace through ``x0`` and an l2 ball)? ``eps`` is
a multiple of the largest radius plain CROWN verifies, so every instance
starts just beyond what bounding alone can prove.
'''

import torch

from abstraction.lirpa import crown
from abstraction.neuralsat.constraint import halfspace, l2_ball
from writeup.exp.config import FACTORS, SIZES

DTYPE = torch.float64


def make_net(seed, sizes=SIZES):
    torch.manual_seed(seed)
    linears = [torch.nn.Linear(i, o).to(DTYPE)
               for i, o in zip(sizes[:-1], sizes[1:])]
    layers = [m for linear in linears[:-1]
              for m in (linear, torch.nn.ReLU())]
    return torch.nn.Sequential(*layers, linears[-1])


def verified_radius(model, x0, label, low=0.0, high=1.0, steps=12):
    # Largest eps (by bisection) for which plain CROWN verifies the ball.
    for _ in range(steps):
        mid = (low + high) / 2
        if crown.verify(model, x0, mid, label):
            low = mid
        else:
            high = mid
    return low


def make_instance(idx):
    # Instance number ``idx``; deterministic.
    model = make_net(idx)
    x0 = torch.randn(SIZES[0], dtype=DTYPE) * 0.5
    label = int(model(x0).argmax())
    eps = verified_radius(model, x0, label) * FACTORS[idx % len(FACTORS)]
    a = torch.randn(SIZES[0], dtype=DTYPE)
    constraints = [halfspace(a, -(a @ x0)),
                   l2_ball(x0, 0.8 * eps * SIZES[0] ** 0.5)]
    return model, x0, eps, label, constraints
