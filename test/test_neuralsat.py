# Tests for abstraction/neuralsat/neuralsat.py. Run ``make test``.

import torch

from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops
from abstraction.neuralsat.constraint import halfspace, l2_ball
from abstraction.neuralsat.neuralsat import (
    bounds,
    margin_lower_bounds,
    verify,
)

TOL = 1e-6
DTYPE = torch.float64


def tensor(*values):
    return torch.tensor(values, dtype=DTYPE)


def make_net(sizes=(4, 16, 16, 3), seed=0):
    torch.manual_seed(seed)
    linears = [torch.nn.Linear(i, o).to(DTYPE)
               for i, o in zip(sizes[:-1], sizes[1:])]
    layers = [m for linear in linears[:-1]
              for m in (linear, torch.nn.ReLU())]
    return torch.nn.Sequential(*layers, linears[-1])


def pre_activations(model, x):
    zs = []
    for layer in crown.from_sequential(model):
        x = ops.forward(layer, x)
        if layer['op'] == 'linear':
            zs.append(x)
    return zs


def feasible_samples(x0, eps, constraints, n=200_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, x0.shape[0], generator=generator, dtype=x0.dtype)
    x = x0 + eps * (2 * noise - 1)
    feasible = torch.ones(n, dtype=torch.bool)
    for h in constraints:
        feasible &= h(x) <= 0
    return x[feasible]


def check_sound(model, x0, eps, constraints):
    x = feasible_samples(x0, eps, constraints)
    assert len(x) > 100
    zs = pre_activations(model, x)
    for (lower, upper), z in zip(bounds(model, x0, eps, constraints), zs):
        assert (z >= lower - TOL).all() and (z <= upper + TOL).all()


def test_matches_crown_without_constraints():
    model, x0 = make_net(), tensor(0.1, -0.2, 0.3, 0.4)
    for (l1, u1), (l2, u2) in zip(bounds(model, x0, 0.3),
                                  crown.bounds(model, x0, 0.3)):
        assert torch.equal(l1, l2) and torch.equal(u1, u2)


def test_constrained_sound():
    x0 = tensor(0.1, -0.2, 0.3, 0.4)
    a = tensor(1.0, 1.0, 0.0, 0.0)
    constraints = [halfspace(a, -(a @ x0)), l2_ball(x0, 0.4)]
    check_sound(make_net(seed=3), x0, 0.3, constraints)


def test_constraints_tighten():
    model, x0 = make_net(seed=5), torch.zeros(4, dtype=DTYPE)
    constraints = [halfspace(tensor(1.0, 1.0, 1.0, 0.0), tensor(0.0)[0])]
    free = bounds(model, x0, 0.5)
    cons = bounds(model, x0, 0.5, constraints)
    # layer 0 is a plain concretization, so it can only get tighter; deeper
    # layers may flip a relu's lower slope, so compare total width instead
    (lf, uf), (lc, uc) = free[0], cons[0]
    assert (lc >= lf - TOL).all() and (uc <= uf + TOL).all()
    width_free = sum((u - lo).sum() for lo, u in free)
    width_cons = sum((u - lo).sum() for lo, u in cons)
    assert width_cons < width_free


def test_tight_on_linear_example():
    # min -(x1 + x2) over [-1, 1]^2 with x1 + x2 <= 0 is exactly 0; the box
    # alone gives -2
    linear = torch.nn.Linear(2, 1).to(DTYPE)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[-1.0, -1.0]]))
        linear.bias.zero_()
    model = torch.nn.Sequential(linear)
    x0 = torch.zeros(2, dtype=DTYPE)
    constraints = [halfspace(torch.ones(2, dtype=DTYPE), tensor(0.0)[0])]
    assert bounds(model, x0, 1.0)[0][0].item() == -2.0
    (lower, _), = bounds(model, x0, 1.0, constraints)
    assert abs(lower.item()) <= TOL


def test_verify_margin_sound():
    model, x0 = make_net(seed=7), tensor(0.1, -0.2, 0.3, 0.4)
    a = tensor(1.0, 0.0, 0.0, 0.0)
    constraints = [halfspace(a, -(a @ x0))]
    label = int(pre_activations(model, x0)[-1].argmax())
    lower = margin_lower_bounds(model, x0, 0.05, label, constraints)
    f = pre_activations(model, feasible_samples(x0, 0.05, constraints))[-1]
    others = [k for k in range(f.shape[1]) if k != label]
    margin = f[:, [label]] - f[:, others]
    assert (margin >= lower - TOL).all()
    assert verify(model, x0, 1e-4, label, constraints)
