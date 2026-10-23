'''The toy network of the walkthrough and a trace of its CROWN bounds.

``trace`` repeats the loop of ``lirpa.crown.backsub`` with the real layer
functions and keeps every state, so the walkthrough shows what the code does.
It checks its final bounds against ``crown.layer_bounds``.
'''

import torch

from abstraction.lirpa import crown
from abstraction.lirpa import layer as ops
from abstraction.lirpa.layer import relu

DTYPE = torch.float64
X0 = (0.5, 0.0)
EPS = 1.0
SAMPLES = 200_000
# (weight, bias) of the three linear layers; ReLU follows the first two
LAYERS = (([[1, 1], [-1, 1]], [0, 0]),
          ([[1, -1], [-1, 1]], [0.5, 0.5]),
          ([[1, 1]], [-0.9]))
# the variables before layer j: x, z1, a1, z2, a2, y
VARIABLES = ('x', 'z1', 'a1', 'z2', 'a2', 'y')


def network():
    # The toy network as an nn.Sequential.
    modules = []
    for k, (weight, bias) in enumerate(LAYERS):
        linear = torch.nn.Linear(2, len(bias)).to(DTYPE)
        linear.weight.data = torch.tensor(weight, dtype=DTYPE)
        linear.bias.data = torch.tensor(bias, dtype=DTYPE)
        modules += [linear] + ([torch.nn.ReLU()] if k < 2 else [])
    return torch.nn.Sequential(*modules)


def point():
    return torch.tensor(X0, dtype=DTYPE)


def listed(tensor):
    return [float(v) for v in tensor.reshape(-1)]


def interval_bounds(layers):
    # Interval propagation through the toy network: bounds per layer output.
    lower, upper = point() - EPS, point() + EPS
    out = []
    for layer in layers:
        if layer['op'] == 'linear':
            center, radius = (lower + upper) / 2, (upper - lower) / 2
            center = layer['weight'] @ center + layer['bias']
            radius = layer['weight'].abs() @ radius
            lower, upper = center - radius, center + radius
        else:
            lower, upper = ops.interval(layer, lower, upper)
        out.append((listed(lower), listed(upper)))
    return out


def sampled_bounds(layers):
    # Smallest and largest value of every layer output on random inputs.
    generator = torch.Generator().manual_seed(0)
    noise = torch.rand(SAMPLES, 2, generator=generator, dtype=DTYPE)
    values = point() + EPS * (2 * noise - 1)
    out = []
    for layer in layers:
        values = ops.forward(layer, values)
        out.append((listed(values.min(0).values),
                    listed(values.max(0).values)))
    return out


def concretize(coef, off, sign):
    # Bound of every row over the input box (the picks per input).
    rows = []
    for row, shift in zip(coef, off):
        terms = []
        for k, c in enumerate(listed(row)):
            low, high = X0[k] - EPS, X0[k] + EPS
            end = low if (c >= 0) == (sign > 0) else high
            terms.append({'coef': c, 'end': end,
                          'pick': 'lower' if end == low else 'upper'})
        value = sum(t['coef'] * t['end'] for t in terms) + float(shift)
        rows.append({'terms': terms, 'const': float(shift), 'value': value})
    return rows


def walk(layers, bounds, i):
    # The back-substitution of layer ``i``: the state after every layer.
    size = int(layers[i]['out_shape'][0])
    eye = torch.eye(size, dtype=DTYPE)
    lo = hi = (eye, torch.zeros(size, dtype=DTYPE))
    states = [{'op': 'start', 'layer': i + 1, 'lo': lo, 'hi': hi}]
    for j in range(i, -1, -1):
        pre = bounds.get(j - 1)
        new_lo, new_hi = ops.backward(layers[j], lo, hi, pre)
        states.append({'op': layers[j]['op'], 'layer': j, 'before':
                       {'lo': lo, 'hi': hi}, 'lo': new_lo, 'hi': new_hi})
        lo, hi = new_lo, new_hi
    return states, lo, hi


def relaxations(pre):
    # The ReLU lines of a layer from the bounds of its input.
    lower, upper = pre
    (d_lo, e_lo), (d_hi, e_hi) = relu.relax(lower, upper)
    out = []
    for k in range(len(lower)):
        status = ('active' if lower[k] >= 0 else
                  'inactive' if upper[k] <= 0 else 'unstable')
        out.append({'lower': float(lower[k]), 'upper': float(upper[k]),
                    'status': status,
                    'lo': (float(d_lo[k]), float(e_lo[k])),
                    'hi': (float(d_hi[k]), float(e_hi[k]))})
    return out


def trace():
    '''Everything the walkthrough shows, as plain numbers.

    Returns ``{'layers', 'steps', 'ibp', 'sampled'}``; ``steps`` has one
    entry per bounded layer with the neuron walks and the final bounds.
    '''
    model = network()
    layers = crown.from_sequential(model, (2,))
    expected = crown.layer_bounds(layers, point(), EPS)
    bounds, steps = {}, []
    for position, i in enumerate(crown.bounded_layers(layers)):
        states, lo, hi = walk(layers, bounds, i)
        lower = concretize(*lo, +1)
        upper = concretize(*hi, -1)
        bounds[i] = (torch.tensor([r['value'] for r in lower], dtype=DTYPE),
                     torch.tensor([r['value'] for r in upper], dtype=DTYPE))
        assert torch.allclose(bounds[i][0], expected[position][0])
        assert torch.allclose(bounds[i][1], expected[position][1])
        steps.append({'layer': i, 'states': states, 'lower': lower,
                      'upper': upper})
    relax = {j: relaxations(bounds[j - 1]) for j in (1, 3)}
    return {'layers': layers, 'steps': steps, 'relax': relax,
            'ibp': interval_bounds(layers),
            'sampled': sampled_bounds(layers)}
