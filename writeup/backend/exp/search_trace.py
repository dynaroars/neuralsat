'''Record a BaB and a DPLL(T) run on a small instance for the website.

The searches run unchanged: their bound function, ``pick_neuron``,
``propagate``, ``explain`` and ``backjump`` are wrapped to log what they do.
The log goes to ``frontend/data/search_trace.json``.

Usage (from the project root):
    python -m writeup.backend.exp.search_trace
'''

import json
import os

import torch

import search.bab as bab_module
import search.dpll as dpll_module
from abstraction.lirpa import crown
from search.common import infeasible, proven
from writeup.backend.exp.config import SITE_DIR
from writeup.backend.exp.instance import make_net, verified_radius

OUT = f'{SITE_DIR}/data/search_trace.json'
SIZES = (2, 4, 4, 2)
SEED, FACTOR = 10, 1.3
DTYPE = torch.float64


def instance():
    # The model, the point, the radius and the label of the example.
    model = make_net(SEED, SIZES)
    torch.manual_seed(1000 + SEED)
    x0 = torch.randn(2, dtype=DTYPE) * 0.5
    label = int(model(x0).argmax())
    return model, x0, verified_radius(model, x0, label) * FACTOR, label


def name(neuron, sign=None):
    # ``+r1.2``: neuron 2 of the first ReLU layer is active.
    j, k = neuron
    mark = '' if sign is None else ('+' if sign > 0 else '-')
    return f'{mark}r{j // 2 + 1}.{k + 1}'


def assignment_of(splits):
    # The phase literals of a ``splits`` dict, as readable names.
    return [name((j, k), int(split[k]))
            for j, split in sorted(splits.items())
            for k in range(len(split)) if int(split[k]) != 0]


def margin_of(bounds):
    return round(float(bounds[-1][0].min()), 4)


def result_of(bounds):
    if infeasible(bounds):
        return 'proven (empty subdomain)'
    return 'proven' if proven(bounds) else 'not proven'


def run_bab(model, x0, eps, label):
    # BaB with a recording bound function; returns the log.
    log = []

    def bound_fn(layers, x, e, splits=None, **kwargs):
        bounds = crown.layer_bounds(layers, x, e, splits=splits, **kwargs)
        log.append({'node': len(log) + 1,
                    'assignment': assignment_of(splits or {}),
                    'margin': margin_of(bounds),
                    'result': result_of(bounds), 'split': None})
        return bounds

    real = bab_module.pick_neuron

    def pick(layers, bounds):
        neuron = real(layers, bounds)
        log[-1]['split'] = None if neuron is None else name(neuron)
        return neuron

    bab_module.pick_neuron = pick
    try:
        status, _ = bab_module.bab(model, x0, eps, label, bound_fn=bound_fn)
    finally:
        bab_module.pick_neuron = real
    return status, log


def run_dpll(model, x0, eps, label):
    # DPLL(T) with every step logged; returns the status and the log.
    log = []
    patched = {name_: getattr(dpll_module, name_)
               for name_ in ('propagate', 'explain', 'backjump',
                             'pick_neuron')}

    def bound_fn(layers, x, e, splits=None, **kwargs):
        out = crown.layer_bounds(layers, x, e, splits=splits, **kwargs)
        bounds = out[0] if kwargs.get('core') else out
        log.append({'event': 'theory check',
                    'detail': ' '.join(assignment_of(splits or {})) or '-',
                    'margin': margin_of(bounds),
                    'result': result_of(bounds)})
        return out

    def propagate(trail, clauses):
        new, clash = patched['propagate'](trail, clauses)
        for neuron, sign, reason in new[len(trail):]:
            log.append({'event': 'propagate',
                        'detail': name(neuron, sign), 'margin': None,
                        'result': 'forced by a learned clause'})
        if clash is not None:
            log.append({'event': 'clash', 'detail': 'a clause is false',
                        'margin': None, 'result': 'conflict'})
        return new, clash

    def pick(layers, bounds):
        neuron = patched['pick_neuron'](layers, bounds)
        log.append({'event': 'decide',
                    'detail': '-' if neuron is None else name(neuron, +1),
                    'margin': None, 'result': 'try active first'})
        return neuron

    def explain(trail, literals):
        out = patched['explain'](trail, literals)
        clause = ' or '.join(name(n, -s) for n, s in out) or 'empty'
        log.append({'event': 'learn', 'detail': clause, 'margin': None,
                    'result': 'the region where these hold is proven'})
        return out

    def backjump(trail, decision):
        log.append({'event': 'backjump',
                    'detail': f'undo {name(*decision)} and later',
                    'margin': None, 'result': 'the clause flips it'})
        return patched['backjump'](trail, decision)

    dpll_module.propagate, dpll_module.pick_neuron = propagate, pick
    dpll_module.explain, dpll_module.backjump = explain, backjump
    try:
        status, _ = dpll_module.dpll(model, x0, eps, label, bound_fn=bound_fn)
    finally:
        for key, value in patched.items():
            setattr(dpll_module, key, value)
    for step, entry in enumerate(log, 1):
        entry['step'] = step
    return status, log


def main():
    model, x0, eps, label = instance()
    layers = crown.margin_layers(model, x0.shape, label)
    root = crown.layer_bounds(layers, x0, eps)
    bab_status, bab_log = run_bab(model, x0, eps, label)
    dpll_status, dpll_log = run_dpll(model, x0, eps, label)
    data = {'instance': {'sizes': list(SIZES), 'eps': round(eps, 4),
                         'x0': [round(float(v), 4) for v in x0],
                         'label': label, 'root_margin': margin_of(root)},
            'bab': {'status': bab_status, 'nodes': bab_log},
            'dpll': {'status': dpll_status, 'steps': dpll_log}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as handle:
        json.dump(data, handle, indent=1)
    print(f'[search_trace] wrote {OUT}: bab {bab_status} in '
          f'{len(bab_log)} nodes, dpll {dpll_status} in {len(dpll_log)} '
          'steps')


if __name__ == '__main__':
    main()
