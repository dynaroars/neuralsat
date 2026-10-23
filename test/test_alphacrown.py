# Tests for abstraction/lirpa/alphacrown.py. Run ``make test``.

import torch

from abstraction.lirpa import alphacrown, crown
from abstraction.lirpa import layer as ops

TOL = 1e-6
DTYPE = torch.float64


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


def ball_samples(x0, eps, n=100_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, x0.shape[0], generator=generator, dtype=x0.dtype)
    return x0 + eps * (2 * noise - 1)


def test_alphacrown_sound():
    model = make_net()
    x0 = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=DTYPE)
    zs = pre_activations(model, ball_samples(x0, 0.3))
    for (lower, upper), z in zip(alphacrown.bounds(model, x0, 0.3), zs):
        assert (z >= lower - TOL).all() and (z <= upper + TOL).all()


def test_alphacrown_not_looser_than_crown():
    model = make_net(seed=2)
    x0 = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=DTYPE)
    alpha = alphacrown.bounds(model, x0, 0.3)
    vanilla = crown.bounds(model, x0, 0.3)
    # first layer is affine, so equal; the output must not get looser
    for (la, ua), (lv, uv) in zip(alpha[:1] + alpha[-1:],
                                  vanilla[:1] + vanilla[-1:]):
        assert (la >= lv - TOL).all() and (ua <= uv + TOL).all()
    width_alpha = sum((u - lo).sum() for lo, u in alpha)
    width_vanilla = sum((u - lo).sum() for lo, u in vanilla)
    assert width_alpha < width_vanilla


def test_zero_steps_equals_crown():
    model = make_net(seed=4)
    x0 = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=DTYPE)
    for (la, ua), (lv, uv) in zip(alphacrown.bounds(model, x0, 0.3, steps=0),
                                  crown.bounds(model, x0, 0.3)):
        assert torch.allclose(la, lv) and torch.allclose(ua, uv)


def test_verify_margin_sound():
    model = make_net(seed=7)
    x0 = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=DTYPE)
    label = int(pre_activations(model, x0)[-1].argmax())
    lower = alphacrown.margin_lower_bounds(model, x0, 0.05, label)
    f = pre_activations(model, ball_samples(x0, 0.05))[-1]
    others = [k for k in range(f.shape[1]) if k != label]
    assert (f[:, [label]] - f[:, others] >= lower - TOL).all()
    assert alphacrown.verify(model, x0, 1e-4, label)
