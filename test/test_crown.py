# Tests for abstraction/lirpa/crown.py. Run ``python -m pytest test``.

import torch

from abstraction.lirpa import layer as ops
from abstraction.lirpa.crown import (
    bounds,
    from_sequential,
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
    for layer in from_sequential(model):
        x = ops.forward(layer, x)
        if layer['op'] == 'linear':
            zs.append(x)
    return zs


def ball_samples(x0, eps, n=200_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, x0.shape[0], generator=generator, dtype=x0.dtype)
    return x0 + eps * (2 * noise - 1)


def test_bounds_sound():
    model, x0 = make_net(), tensor(0.1, -0.2, 0.3, 0.4)
    zs = pre_activations(model, ball_samples(x0, 0.3))
    for (lower, upper), z in zip(bounds(model, x0, 0.3), zs):
        assert (z >= lower - TOL).all() and (z <= upper + TOL).all()


def test_exact_on_linear_network():
    linear = torch.nn.Linear(2, 1).to(DTYPE)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[-1.0, 2.0]]))
        linear.bias.fill_(0.5)
    (lower, upper), = bounds(torch.nn.Sequential(linear),
                             torch.zeros(2, dtype=DTYPE), 1.0)
    assert abs(lower.item() + 2.5) <= TOL and abs(upper.item() - 3.5) <= TOL


def test_verify_margin_sound():
    model, x0 = make_net(seed=7), tensor(0.1, -0.2, 0.3, 0.4)
    label = int(pre_activations(model, x0)[-1].argmax())
    lower = margin_lower_bounds(model, x0, 0.05, label)
    f = pre_activations(model, ball_samples(x0, 0.05))[-1]
    others = [k for k in range(f.shape[1]) if k != label]
    margin = f[:, [label]] - f[:, others]
    assert (margin >= lower - TOL).all()
    assert verify(model, x0, 1e-4, label)


def test_only_supported_sequential():
    unsupported = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.ELU(), torch.nn.Linear(2, 2))
    for bad in (torch.nn.Linear(2, 2), unsupported, torch.nn.Sequential()):
        try:
            from_sequential(bad)
        except TypeError:
            continue
        raise AssertionError(f'accepted {bad}')
    # a network may end with an activation, and conv nets need a shape
    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    assert [layer['op'] for layer in from_sequential(net)] == [
        'linear', 'relu']
    conv = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3))
    try:
        from_sequential(conv)
    except TypeError:
        pass
    else:
        raise AssertionError('a conv net needs input_shape')
    assert from_sequential(conv, (1, 5, 5))[0]['out_shape'] == (2, 3, 3)
