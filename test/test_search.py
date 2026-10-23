# Tests for abstraction/search. Run ``make test``.

import functools

import torch

from abstraction.lirpa import alphacrown, crown
from abstraction.lirpa import layer as ops
from abstraction.neuralsat import neuralsat
from abstraction.neuralsat.constraint import halfspace
from search.bab import bab
from search.common import (
    network,
    pick_neuron,
    proven,
    split_tensors,
)
from search.dpll import dpll, effective_literals, explain

TOL = 1e-6
DTYPE = torch.float64
X0 = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=DTYPE)


def make_net(seed, sizes=(4, 12, 12, 3)):
    torch.manual_seed(seed)
    linears = [torch.nn.Linear(i, o).to(DTYPE)
               for i, o in zip(sizes[:-1], sizes[1:])]
    layers = [m for linear in linears[:-1]
              for m in (linear, torch.nn.ReLU())]
    return torch.nn.Sequential(*layers, linears[-1])


def label_of(model):
    return int(model(X0).argmax())


def ball_samples(eps, n=100_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, X0.shape[0], generator=generator, dtype=DTYPE)
    return X0 + eps * (2 * noise - 1)


def robust_on_samples(model, label, eps):
    out = model(ball_samples(eps))
    return bool((out.argmax(-1) == label).all())


def split_helps_cases():
    # (model, label, eps): plain CROWN cannot verify, samples are all robust
    for seed in range(10):
        model = make_net(seed)
        label = label_of(model)
        for step in range(1, 30):
            eps = 0.02 * step
            if (not crown.verify(model, X0, eps, label)
                    and robust_on_samples(model, label, eps)):
                yield model, label, eps
                break


def test_split_bounds_sound():
    model = make_net(0)
    layers = crown.from_sequential(model)
    bounds = crown.layer_bounds(layers, X0, 0.3)
    j, k = pick_neuron(layers, bounds)
    for sign in (+1, -1):
        splits = split_tensors(layers, {(j, k): sign})
        out = crown.layer_bounds(layers, X0, 0.3, splits=splits)
        x = ball_samples(0.3)
        zs, a = [], x
        for layer in layers:
            a = ops.forward(layer, a)
            if layer['op'] == 'linear':
                zs.append(a)
        keep = (zs[(j - 1) // 2][:, k] * sign) >= 0
        for (lower, upper), z in zip(out, zs):
            assert (z[keep] >= lower - TOL).all()
            assert (z[keep] <= upper + TOL).all()


def test_bab_verifies_what_crown_cannot():
    cases = list(split_helps_cases())
    assert cases
    unsat = 0
    for model, label, eps in cases[:3]:
        status, witness = bab(model, X0, eps, label, max_nodes=2000)
        assert status != 'sat' and witness is None
        unsat += status == 'unsat'
    assert unsat >= 1


def test_dpll_verifies_what_crown_cannot():
    cases = list(split_helps_cases())
    assert cases
    unsat = 0
    for model, label, eps in cases[:3]:
        status, witness = dpll(model, X0, eps, label, max_checks=4000)
        assert status != 'sat' and witness is None
        unsat += status == 'unsat'
    assert unsat >= 1


def test_violation_has_valid_witness():
    model = make_net(3)
    label = label_of(model)
    for search in (bab, dpll):
        status, x = search(model, X0, 2.0, label)
        assert status == 'sat'
        assert (x - X0).abs().max() <= 2.0 + TOL
        out = network(crown.from_sequential(model), x)
        others = torch.cat([out[:label], out[label + 1:]])
        assert out[label] <= others.max() + TOL


def test_search_with_other_abstractions():
    model = make_net(0)
    label = label_of(model)
    constraints = [halfspace(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                             torch.tensor(-0.1, dtype=DTYPE))]
    bound_fns = [
        functools.partial(alphacrown.layer_bounds, steps=5),
        functools.partial(neuralsat.layer_bounds, constraints=constraints,
                          steps=20),
    ]
    for bound_fn in bound_fns:
        for search in (bab, dpll):
            status, _ = search(model, X0, 0.01, label, bound_fn=bound_fn)
            assert status == 'unsat'


def descend_until_proven(model, label, eps, seed):
    # Random splits of unstable neurons until the subdomain is proven.
    folded = crown.margin_layers(model, X0.shape, label)
    generator = torch.Generator().manual_seed(seed)
    assignment = {}
    for _ in range(40):
        bounds, effective = crown.layer_bounds(
            folded, X0, eps, splits=split_tensors(folded, assignment),
            core=True)
        if proven(bounds):
            return folded, assignment, bounds, effective
        neuron = pick_neuron(folded, bounds)
        if neuron is None:
            return None
        sign = 1 if torch.rand(1, generator=generator) < 0.5 else -1
        assignment[neuron] = sign
    return None


def stable_neuron(layers, bounds):
    # A neuron that is stable without any split, with its stable sign.
    for j, layer in enumerate(layers):
        if layer['op'] != 'relu':
            continue
        lower, upper = bounds[(j - 1) // 2]
        for k in range(len(lower)):
            if lower[k] >= 0:
                return (j, k), +1
            if upper[k] <= 0:
                return (j, k), -1
    return None


def test_dropping_ineffective_splits_keeps_bounds():
    checked = 0
    for seed in range(8):
        model = make_net(seed)
        folded = crown.margin_layers(model, X0.shape, label_of(model))
        root = crown.layer_bounds(folded, X0, 0.3)
        stable = stable_neuron(folded, root)
        free = pick_neuron(folded, root)
        if stable is None or free is None:
            continue
        assignment = {stable[0]: stable[1], free: 1}
        bounds, effective = crown.layer_bounds(
            folded, X0, 0.3, splits=split_tensors(folded, assignment),
            core=True)
        (j, k), _ = stable
        assert not bool(effective[j][k])  # the split was already implied
        assert bool(effective[free[0]][free[1]])
        again = crown.layer_bounds(
            folded, X0, 0.3, splits=split_tensors(folded, {free: 1}))
        assert all(torch.equal(a, c) and torch.equal(b, d)
                   for (a, b), (c, d) in zip(bounds, again))
        checked += 1
    assert checked > 0


def test_explained_splits_still_prove():
    checked = 0
    for seed in range(12):
        model = make_net(seed)
        found = descend_until_proven(model, label_of(model), 0.25, seed)
        if found is None or not found[1]:
            continue
        folded, assignment, bounds, effective = found
        literals = effective_literals(folded, bounds, effective, assignment)
        again = crown.layer_bounds(
            folded, X0, 0.25, splits=split_tensors(folded, dict(literals)))
        assert proven(again)
        checked += 1
    assert checked > 0


def test_explain_follows_reasons_to_decisions():
    a, b, c = (1, 0), (1, 1), (3, 0)
    # a, b decided; c implied by the clause (c or not a or not b)
    trail = [(a, 1, None), (b, -1, None), (c, 1, [(c, 1), (a, -1), (b, 1)])]
    assert explain(trail, [(c, 1)]) == [(a, 1), (b, -1)]
    assert explain(trail, [(b, -1)]) == [(b, -1)]
    assert explain(trail, []) == []
