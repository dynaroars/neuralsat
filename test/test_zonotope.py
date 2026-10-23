# Tests for abstraction/zonotope. Run ``make test``.

import torch
from test_ibp import (
    SHAPE,
    TOL,
    affine_chain,
    ball_samples,
    check_sound,
    competing_inputs,
    make_cnn,
)

from abstraction.ibp import ibp
from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops
from abstraction.zonotope import zonotope
from search.bab import bab
from search.common import pick_neuron, split_tensors
from search.dpll import dpll

DTYPE = torch.float64


def random_zonotope(neurons, generators, seed):
    generator = torch.Generator().manual_seed(seed)
    center = torch.randn(neurons, generator=generator, dtype=DTYPE)
    gens = torch.randn(generators, neurons, generator=generator,
                       dtype=DTYPE) * 1.5
    return center, gens


def check_transformer(layer, shape, center, gens, samples=5000):
    '''The output of every sampled point lies in the new zonotope.

    For error terms ``eps`` of the input the new zonotope may still choose
    its new generators, which add ``+- |g_new|`` per neuron.
    '''
    lower, upper = zonotope.concretize(center, gens)
    new_center, new_gens = ops.zonotope(layer, center, gens, lower, upper)
    k = gens.shape[0]
    generator = torch.Generator().manual_seed(3)
    eps = 2 * torch.rand(samples, k, generator=generator, dtype=DTYPE) - 1
    z = center + eps @ gens
    y = ops.forward(layer, z.reshape(samples, *shape)).reshape(samples, -1)
    middle = new_center + eps @ new_gens[:k]
    slack = new_gens[k:].abs().sum(0)
    assert (y >= middle - slack - TOL).all()
    assert (y <= middle + slack + TOL).all()


def test_pointwise_transformers_sound():
    for seed in range(4):
        center, gens = random_zonotope(10, 3, seed)
        for layer in ({'op': 'relu'}, {'op': 'leakyrelu', 'slope': 0.2},
                      {'op': 'sigmoid'}, {'op': 'tanh'}):
            check_transformer(layer, (10,), center, gens)


def test_maxpool_transformer_sound():
    layer = {'op': 'maxpool', 'kernel': (2, 2), 'in_shape': (1, 4, 4)}
    for seed in range(4):
        center, gens = random_zonotope(16, 4, seed)
        check_transformer(layer, (1, 4, 4), center, gens)


def test_stable_relu_is_exact():
    center = torch.tensor([3.0, -3.0], dtype=DTYPE)
    gens = torch.tensor([[0.5, 0.5]], dtype=DTYPE)
    lower, upper = zonotope.concretize(center, gens)
    new_center, new_gens = ops.zonotope({'op': 'relu'}, center, gens, lower,
                                        upper)
    assert new_gens.shape[0] == 1  # no new generator
    assert torch.allclose(new_center, torch.tensor([3.0, 0.0], dtype=DTYPE))
    assert torch.allclose(new_gens, torch.tensor([[0.5, 0.0]], dtype=DTYPE))


def test_zonotope_sound_on_cnn():
    check_sound(zonotope.layer_bounds)


def test_zonotope_exact_on_affine_cnn():
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.BatchNorm2d(2),
        torch.nn.AvgPool2d(2), torch.nn.Flatten(),
        torch.nn.Linear(18, 3)).to(DTYPE).eval()
    x0 = torch.randn(*SHAPE, dtype=DTYPE)
    lower, upper = zonotope.bounds(model, x0, 0.2)[-1]
    exact_lo, exact_hi = crown.bounds(model, x0, 0.2)[-1]
    assert torch.allclose(lower, exact_lo, atol=1e-9)
    assert torch.allclose(upper, exact_hi, atol=1e-9)


def test_zonotope_keeps_dependencies():
    x0 = torch.zeros(2, dtype=DTYPE)
    lower, upper = zonotope.bounds(affine_chain(), x0, 1.0)[-1]
    ibp_lower, ibp_upper = ibp.bounds(affine_chain(), x0, 1.0)[-1]
    assert float(upper - lower) == 4.0  # exact: y = 2 x1
    assert float(ibp_upper - ibp_lower) == 8.0


def test_zonotope_splits_sound():
    model = make_cnn(seed=2)[:-1]
    x0 = torch.randn(*SHAPE, dtype=DTYPE) * 0.5
    layers = crown.from_sequential(model, SHAPE)
    j, k = pick_neuron(layers, zonotope.layer_bounds(layers, x0, 0.4))
    samples = ball_samples(x0, 0.4)
    pre = samples
    for layer in layers[:j]:
        pre = ops.forward(layer, pre)
    pre = pre.reshape(pre.shape[0], -1)
    bounded = crown.bounded_layers(layers)
    for sign in (+1, -1):
        out = zonotope.layer_bounds(
            layers, x0, 0.4, splits=split_tensors(layers, {(j, k): sign}))
        a = samples[pre[:, k] * sign >= 0]
        for i, layer in enumerate(layers):
            a = ops.forward(layer, a)
            if i in bounded:
                lower, upper = out[bounded.index(i)]
                flat = a.reshape(a.shape[0], -1)
                assert (flat >= lower - TOL).all()
                assert (flat <= upper + TOL).all()


def test_searches_with_zonotope():
    model = competing_inputs()
    x0 = torch.tensor([0.3, -0.2], dtype=DTYPE)
    label = int(model(x0).argmax())
    for search in (bab, dpll):
        assert search(model, x0, 0.01, label,
                      bound_fn=zonotope.layer_bounds)[0] == 'unsat'
        status, witness = search(model, x0, 1.0, label,
                                 bound_fn=zonotope.layer_bounds)
        assert status == 'sat'
        out = model(witness)
        assert out[label] <= out[1 - label] + TOL
