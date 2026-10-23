# Tests for the patch form of the coefficients. Run ``make test``.

import math

import torch

from abstraction.lirpa import alphacrown, crown
from abstraction.lirpa.layer import patches
from abstraction.neuralsat import neuralsat
from abstraction.neuralsat.constraint import halfspace

DTYPE = torch.float64
TOL = 1e-9
nn = torch.nn


def build(shape, *modules, seed=0):
    # A random double network; batch norm gets non-trivial statistics.
    torch.manual_seed(seed)
    model = nn.Sequential(*modules).to(DTYPE)
    for module in model:
        if isinstance(module, nn.BatchNorm2d):
            with torch.no_grad():
                module.running_mean.normal_(0, 0.3)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.normal_(0, 0.3)
    return model.eval(), torch.randn(*shape, dtype=DTYPE) * 0.5


def networks():
    return {
        'strided padded conv, bn, tanh': build(
            (1, 8, 8), nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3),
            nn.ReLU(), nn.Conv2d(3, 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(4), nn.Tanh(), nn.Conv2d(4, 2, 2), nn.Flatten(),
            nn.Linear(18, 3)),
        'conv conv without activation': build(
            (1, 6, 6), nn.Conv2d(1, 2, 3, padding=1),
            nn.Conv2d(2, 3, 3, padding=1), nn.ReLU(),
            nn.Conv2d(3, 2, 3, padding=1), nn.LeakyReLU(0.2), nn.Flatten(),
            nn.Linear(72, 3)),
        'dilation and groups': build(
            (2, 8, 8), nn.Conv2d(2, 4, 3, dilation=2, padding=2, groups=2),
            nn.LeakyReLU(0.1), nn.Conv2d(4, 4, 3, groups=4), nn.Sigmoid(),
            nn.Flatten(), nn.Linear(144, 3)),
        'avgpool with padding': build(
            (1, 7, 7), nn.Conv2d(1, 2, 3), nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1), nn.Conv2d(2, 2, 2),
            nn.Tanh(), nn.Flatten(), nn.Linear(8, 3)),
        'avgpool without padded count (dense fallback)': build(
            (1, 7, 7), nn.Conv2d(1, 2, 3), nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
            nn.Conv2d(2, 2, 2), nn.Tanh(), nn.Flatten(), nn.Linear(8, 3)),
        'maxpool with leftover row': build(
            (1, 7, 7), nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(2, 3, 2, padding=1), nn.Sigmoid(),
            nn.MaxPool2d(2), nn.Flatten(), nn.Linear(12, 3)),
        'maxpool kernel 3, bn': build(
            (1, 9, 9), nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2),
            nn.ReLU(), nn.MaxPool2d(3), nn.Conv2d(2, 2, 2), nn.Flatten(),
            nn.Linear(8, 3)),
    }


def reference(layers, x0, eps):
    # Dense bounds of every bounded layer, as a dict by layer index.
    dense = crown.layer_bounds(layers, x0, eps, patches=False)
    return dict(zip(crown.bounded_layers(layers), dense))


def test_to_dense_matches_a_loop():
    generator = torch.Generator().manual_seed(1)
    for hp, wp in ((1, 1), (3, 3)):
        p = torch.randn(2, hp, wp, 3, 3, 2, generator=generator, dtype=DTYPE)
        geometry = {'p': p, 'start': (2, 3, 3), 'shape': (3, 5, 5),
                    'stride': (2, 1), 'padding': (1, 0)}
        dense = patches.to_dense(geometry)
        expected = torch.zeros(2, 3, 3, 3, 5, 5, dtype=DTYPE)
        for cs in range(2):
            for hs in range(3):
                for ws in range(3):
                    for c in range(3):
                        for a in range(3):
                            for b in range(2):
                                y, x = hs * 2 + a - 1, ws * 1 + b - 0
                                if 0 <= y < 5 and 0 <= x < 5:
                                    expected[cs, hs, ws, c, y, x] += p[
                                        cs, min(hs, hp - 1), min(ws, wp - 1),
                                        c, a, b]
        assert torch.allclose(dense, expected.reshape(18, 75), atol=TOL)


def test_identity_patches_are_the_identity():
    like = torch.zeros(1, dtype=DTYPE)
    dense = patches.to_dense(patches.identity((3, 4, 5), like))
    assert torch.equal(dense, torch.eye(60, dtype=DTYPE))


def test_backsub_patches_equal_dense():
    for name, (model, x0) in networks().items():
        layers = crown.from_sequential(model, x0.shape)
        bounds = reference(layers, x0, 0.2)
        flat, used = x0.reshape(-1), 0
        for i in crown.bounded_layers(layers):
            if crown.use_interval(layers, i):
                continue
            new = crown.backsub(layers, bounds, i, flat, patches=True)
            old = crown.backsub(layers, bounds, i, flat, patches=False)
            used += patches.is_patches(new[0][0])
            for (coef, off), (ref_coef, ref_off) in zip(new, old):
                assert torch.allclose(patches.as_dense(coef), ref_coef,
                                      atol=TOL), name
                assert torch.allclose(off, ref_off, atol=TOL), name
            for sign, side in ((+1, 0), (-1, 1)):
                assert torch.allclose(
                    crown.concretize(*new[side], flat, 0.2, sign),
                    crown.concretize(*old[side], flat, 0.2, sign),
                    atol=TOL), name
        assert used > 0, name  # the patch form really was used


def test_layer_bounds_patches_equal_dense():
    constraint = None
    for name, (model, x0) in networks().items():
        layers = crown.from_sequential(model, x0.shape)
        pairs = {
            'crown': (crown.layer_bounds(layers, x0, 0.2),
                      crown.layer_bounds(layers, x0, 0.2, patches=False)),
            'alpha': (alphacrown.layer_bounds(layers, x0, 0.2, steps=3),
                      alphacrown.layer_bounds(layers, x0, 0.2, steps=3,
                                              patches=False)),
        }
        constraint = [halfspace(torch.ones(x0.numel(), dtype=DTYPE),
                                -(x0.sum()))]
        pairs['neuralsat'] = (
            neuralsat.layer_bounds(layers, x0, 0.2, constraint, steps=5),
            neuralsat.layer_bounds(layers, x0, 0.2, constraint, steps=5,
                                   patches=False))
        for kind, (new, old) in pairs.items():
            for (lower, upper), (ref_lower, ref_upper) in zip(new, old):
                assert torch.allclose(lower, ref_lower, atol=TOL), (
                    name, kind)
                assert torch.allclose(upper, ref_upper, atol=TOL), (
                    name, kind)


def test_patches_are_smaller_than_dense():
    shape = (1, 28, 28)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 10)).to(DTYPE)
    x0 = torch.rand(*shape, dtype=DTYPE)
    layers = crown.from_sequential(model, shape)
    bounds = reference(layers, x0, 1 / 255)
    (coef, _), _ = crown.backsub(layers, bounds, 2, x0.reshape(-1))
    assert patches.is_patches(coef)
    dense_size = (32 * 7 * 7) * math.prod(shape)
    assert coef['p'].numel() < dense_size / 5
