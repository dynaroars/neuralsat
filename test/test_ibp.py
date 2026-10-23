# Tests for abstraction/ibp. Run ``make test``.

import torch

from abstraction.ibp import ibp
from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops
from search.bab import bab
from search.common import pick_neuron, split_tensors
from search.dpll import dpll

TOL = 1e-7
DTYPE = torch.float64
SHAPE = (1, 6, 6)


def make_cnn(seed=0):
    # Every supported layer type; batch norm gets non-trivial statistics.
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 3, padding=1), torch.nn.BatchNorm2d(3),
        torch.nn.LeakyReLU(0.1), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(3, 4, 2), torch.nn.Sigmoid(), torch.nn.AvgPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(4, 6), torch.nn.Tanh(),
        torch.nn.Dropout(0.5), torch.nn.Identity(), torch.nn.Linear(6, 3),
        torch.nn.ReLU()).to(DTYPE)
    with torch.no_grad():
        model[1].running_mean.normal_(0, 0.3)
        model[1].running_var.uniform_(0.5, 2.0)
        model[1].weight.uniform_(0.5, 1.5)
        model[1].bias.normal_(0, 0.3)
    return model.eval()


def ball_samples(x0, eps, n=20_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, *x0.shape, generator=generator, dtype=x0.dtype)
    return x0 + eps * (2 * noise - 1)


def check_sound(layer_bounds, keep=None):
    # Every bounded layer value of sampled inputs lies inside its bounds.
    model = make_cnn()
    x0 = torch.randn(*SHAPE, dtype=DTYPE) * 0.5
    layers = crown.from_sequential(model, SHAPE)
    out = layer_bounds(layers, x0, 0.3)
    a = ball_samples(x0, 0.3)
    bounded = crown.bounded_layers(layers)
    for i, layer in enumerate(layers):
        a = ops.forward(layer, a)
        if i in bounded:
            lower, upper = out[bounded.index(i)]
            flat = a.reshape(a.shape[0], -1)
            assert (flat >= lower - TOL).all() and (flat <= upper + TOL).all()


def affine_chain():
    # y = z1 + z2 = 2 x1, but the intervals of z1 and z2 do not know it.
    first = torch.nn.Linear(2, 2).to(DTYPE)
    second = torch.nn.Linear(2, 1).to(DTYPE)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        first.bias.zero_()
        second.weight.copy_(torch.tensor([[1.0, 1.0]]))
        second.bias.zero_()
    return torch.nn.Sequential(first, second)


def test_ibp_sound_on_cnn():
    check_sound(ibp.layer_bounds)


def test_ibp_first_layer_is_exact():
    model = make_cnn(seed=3)[:3]
    x0 = torch.randn(*SHAPE, dtype=DTYPE)
    (exact_lo, exact_hi), = crown.bounds(model[:1], x0, 0.2)
    (lower, upper), = ibp.bounds(model[:1], x0, 0.2)
    assert torch.allclose(lower, exact_lo) and torch.allclose(upper, exact_hi)


def test_ibp_forgets_dependencies():
    x0 = torch.zeros(2, dtype=DTYPE)
    lower, upper = ibp.bounds(affine_chain(), x0, 1.0)[-1]
    exact_lo, exact_hi = crown.bounds(affine_chain(), x0, 1.0)[-1]
    assert float(upper - lower) == 8.0 and float(exact_hi - exact_lo) == 4.0


def test_ibp_splits_sound():
    model = make_cnn(seed=2)[:-1]
    x0 = torch.randn(*SHAPE, dtype=DTYPE) * 0.5
    layers = crown.from_sequential(model, SHAPE)
    j, k = pick_neuron(layers, ibp.layer_bounds(layers, x0, 0.4))
    samples = ball_samples(x0, 0.4)
    pre = samples
    for layer in layers[:j]:
        pre = ops.forward(layer, pre)
    pre = pre.reshape(pre.shape[0], -1)
    bounded = crown.bounded_layers(layers)
    for sign in (+1, -1):
        out = ibp.layer_bounds(layers, x0, 0.4,
                               splits=split_tensors(layers, {(j, k): sign}))
        a = samples[pre[:, k] * sign >= 0]
        for i, layer in enumerate(layers):
            a = ops.forward(layer, a)
            if i in bounded:
                lower, upper = out[bounded.index(i)]
                flat = a.reshape(a.shape[0], -1)
                assert (flat >= lower - TOL).all()
                assert (flat <= upper + TOL).all()


def competing_inputs():
    # relu(x) with two outputs: the label flips once x2 can pass x1.
    first = torch.nn.Linear(2, 2).to(DTYPE)
    second = torch.nn.Linear(2, 2).to(DTYPE)
    with torch.no_grad():
        for linear in (first, second):
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()
    return torch.nn.Sequential(first, torch.nn.ReLU(), second)


def test_searches_with_ibp():
    model = competing_inputs()
    x0 = torch.tensor([0.3, -0.2], dtype=DTYPE)
    label = int(model(x0).argmax())
    for search in (bab, dpll):
        assert search(model, x0, 0.01, label,
                      bound_fn=ibp.layer_bounds)[0] == 'unsat'
        status, witness = search(model, x0, 1.0, label,
                                 bound_fn=ibp.layer_bounds)
        assert status == 'sat'
        out = model(witness)
        assert out[label] <= out[1 - label] + TOL  # a tie counts
