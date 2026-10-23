'''Convex input constraints for the neuralsat abstraction.

A constraint is a plain function ``h(x) -> (rows,)`` on a ``(rows, D)`` batch,
feasible iff ``h(x) <= 0``. It must be convex and use torch operations so its
subgradient can come from autograd. The Lagrangian penalizes ``relu(h(x))``.
'''

import torch


def halfspace(a, c):
    # Constraint a . x + c <= 0.
    return lambda x: x @ a + c


def l2_ball(center, radius):
    # Constraint ||x - center||_2 <= radius.
    return lambda x: (x - center).norm(dim=-1) - radius


def penalty_terms(x, constraints):
    '''Values ``h_j(x)`` and subgradients of ``relu(h_j)`` at ``x``.

    Returns ``([values], [grads])`` with one entry per constraint.
    '''
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        values = [h(x) for h in constraints]
        grads = [torch.autograd.grad(torch.relu(v).sum(), x)[0]
                 for v in values]
    return [v.detach() for v in values], grads
