# Tests for abstraction/lirpa/layer. Run ``make test``.

import torch

from abstraction.lirpa import alphacrown, crown
from abstraction.lirpa import layer as ops
from abstraction.lirpa.layer import (
    leakyrelu,
    maxpool,
    relu,
    sigmoid,
    tanh,
)

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
    norm = model[1]
    with torch.no_grad():
        norm.running_mean.normal_(0, 0.3)
        norm.running_var.uniform_(0.5, 2.0)
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.normal_(0, 0.3)
    return model.eval()


def ball_samples(x0, eps, n=20_000, seed=1):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, *x0.shape, generator=generator, dtype=x0.dtype)
    return x0 + eps * (2 * noise - 1)


def test_layer_forward_matches_torch():
    model = make_cnn()
    layers = crown.from_sequential(model, SHAPE)
    x = torch.randn(5, *SHAPE, dtype=DTYPE)
    out = x
    for layer in layers:
        out = ops.forward(layer, out)
    assert torch.allclose(out, model(x), atol=1e-12)
    assert [layer['out_shape'] for layer in layers][-1] == (3,)


def test_affine_backward_matches_jacobian():
    model = make_cnn()
    layers = crown.from_sequential(model, SHAPE)
    torch.manual_seed(3)
    for layer in layers:
        if layer['op'] not in ('linear', 'conv', 'batchnorm', 'avgpool',
                               'flatten', 'identity'):
            continue
        size_in = int(torch.tensor(layer['in_shape']).prod())
        size_out = int(torch.tensor(layer['out_shape']).prod())

        def fn(flat):
            x = flat.reshape(1, *layer['in_shape'])
            return ops.forward(layer, x).reshape(-1)

        base = torch.randn(size_in, dtype=DTYPE)
        jacobian = torch.autograd.functional.jacobian(fn, base)
        const = fn(torch.zeros(size_in, dtype=DTYPE)) if size_in else None
        coef = torch.randn(4, size_out, dtype=DTYPE)
        off = torch.randn(4, dtype=DTYPE)
        (new, new_off), _ = ops.backward(layer, (coef, off), (coef, off),
                                         None)
        assert torch.allclose(new, coef @ jacobian, atol=1e-10)
        assert torch.allclose(new_off, coef @ const + off, atol=1e-10)


def check_relaxation(func, relax, lower, upper, tol=TOL):
    # lower line <= func <= upper line on a dense grid of every interval
    (d_lo, e_lo), (d_hi, e_hi) = relax(lower, upper)
    for t in torch.linspace(0, 1, 401, dtype=DTYPE):
        z = lower + t * (upper - lower)
        assert (d_lo * z + e_lo <= func(z) + tol).all()
        assert (d_hi * z + e_hi >= func(z) - tol).all()


def random_intervals(n=300, scale=4.0, seed=5):
    generator = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, generator=generator, dtype=DTYPE) - 0.5) * 2 * scale
    b = (torch.rand(n, generator=generator, dtype=DTYPE) - 0.5) * 2 * scale
    lower, upper = torch.minimum(a, b), torch.maximum(a, b)
    # also stable and degenerate ones
    extra_l = torch.tensor([-2.0, 0.0, 1.0, -3.0, 0.5, -0.5, 0.0],
                           dtype=DTYPE)
    extra_u = torch.tensor([-1.0, 2.0, 3.0, 0.2, 0.5, 0.0, 0.0], dtype=DTYPE)
    return torch.cat([lower, extra_l]), torch.cat([upper, extra_u])


def test_relu_relaxation_sound():
    lower, upper = random_intervals()
    check_relaxation(torch.relu, relu.relax, lower, upper)


def test_leakyrelu_relaxation_sound():
    lower, upper = random_intervals()
    for slope in (0.0, 0.1, 0.5, 1.0):
        check_relaxation(
            lambda z: torch.nn.functional.leaky_relu(z, slope),
            lambda lo, up: leakyrelu.relax(slope, lo, up), lower, upper)


def test_sigmoid_relaxation_sound():
    lower, upper = random_intervals(scale=6.0)
    check_relaxation(torch.sigmoid, sigmoid.relax, lower, upper)


def test_tanh_relaxation_sound():
    lower, upper = random_intervals(scale=6.0)
    check_relaxation(torch.tanh, tanh.relax, lower, upper)


def test_sshaped_exact_on_a_point():
    point = torch.tensor([-1.5, 0.0, 2.0], dtype=DTYPE)
    (d_lo, e_lo), (d_hi, e_hi) = tanh.relax(point, point)
    value = torch.tanh(point)
    assert torch.allclose(d_lo * point + e_lo, value, atol=1e-9)
    assert torch.allclose(d_hi * point + e_hi, value, atol=1e-9)


def test_maxpool_relaxation_sound():
    layer = {'op': 'maxpool', 'kernel': (2, 2), 'in_shape': (2, 4, 4)}
    generator = torch.Generator().manual_seed(7)
    for scale in (0.05, 0.5, 3.0):
        center = torch.randn(32, generator=generator, dtype=DTYPE)
        lower, upper = center - scale, center + scale
        index, (d_lo, e_lo), (d_hi, e_hi) = maxpool.relaxation(
            layer, lower, upper)
        for _ in range(200):
            x = lower + torch.rand(32, generator=generator,
                                   dtype=DTYPE) * (upper - lower)
            out = torch.nn.functional.max_pool2d(
                x.reshape(1, 2, 4, 4), 2).flatten()
            assert (d_lo * x[index] + e_lo <= out + TOL).all()
            assert (d_hi * x[index] + e_hi >= out - TOL).all()


def test_maxpool_rejects_overlap_and_padding():
    for bad in (torch.nn.MaxPool2d(3, stride=1),
                torch.nn.MaxPool2d(2, padding=1),
                torch.nn.MaxPool2d(2, ceil_mode=True)):
        try:
            maxpool.parse(bad)
        except TypeError:
            continue
        raise AssertionError(f'accepted {bad}')


def test_cnn_bounds_sound():
    model = make_cnn()
    x0 = torch.randn(*SHAPE, dtype=DTYPE) * 0.5
    layers = crown.from_sequential(model, SHAPE)
    samples = ball_samples(x0, 0.3)
    activations, a = {}, samples
    for i, layer in enumerate(layers):
        a = ops.forward(layer, a)
        activations[i] = a.reshape(a.shape[0], -1)
    for bound_fn in (crown.layer_bounds,
                     lambda *args: alphacrown.layer_bounds(*args, steps=5)):
        out = bound_fn(layers, x0, 0.3)
        for i, (lower, upper) in zip(crown.bounded_layers(layers), out):
            assert (activations[i] >= lower - TOL).all()
            assert (activations[i] <= upper + TOL).all()


def test_affine_cnn_bounds_exact():
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 3, padding=1), torch.nn.BatchNorm2d(2),
        torch.nn.AvgPool2d(2), torch.nn.Flatten(),
        torch.nn.Linear(18, 3)).to(DTYPE).eval()
    x0 = torch.randn(*SHAPE, dtype=DTYPE)
    (lower, upper), = crown.bounds(model, x0, 0.2)
    jacobian = torch.autograd.functional.jacobian(
        lambda x: model(x.unsqueeze(0))[0], x0).reshape(3, -1)
    center = model(x0.unsqueeze(0))[0]
    radius = 0.2 * jacobian.abs().sum(-1)
    assert torch.allclose(lower, center - radius, atol=1e-9)
    assert torch.allclose(upper, center + radius, atol=1e-9)


def test_alphacrown_not_looser_on_cnn():
    model = make_cnn(seed=4)
    x0 = torch.randn(*SHAPE, dtype=DTYPE) * 0.5
    base = crown.bounds(model, x0, 0.3)[-1]
    tuned = alphacrown.bounds(model, x0, 0.3, steps=10)[-1]
    assert (tuned[0] >= base[0] - TOL).all()
    assert (tuned[1] <= base[1] + TOL).all()


def cnn_without_final_relu(seed=0):
    return make_cnn(seed)[:-1]


def cnn_point(seed=0):
    torch.manual_seed(100 + seed)
    return torch.randn(*SHAPE, dtype=DTYPE) * 0.5


def test_split_leakyrelu_bounds_sound():
    from search.common import pick_neuron, split_tensors

    model = cnn_without_final_relu()
    x0 = cnn_point()
    layers = crown.from_sequential(model, SHAPE)
    root = crown.layer_bounds(layers, x0, 0.4)
    j, k = pick_neuron(layers, root)
    assert layers[j]['op'] in ops.SPLITTABLE
    samples = ball_samples(x0, 0.4)
    pre = samples
    for layer in layers[:j]:
        pre = ops.forward(layer, pre)
    pre = pre.reshape(pre.shape[0], -1)
    for sign in (+1, -1):
        splits = split_tensors(layers, {(j, k): sign})
        assert splits[j].numel() == pre.shape[1]
        out = crown.layer_bounds(layers, x0, 0.4, splits=splits)
        keep = pre[:, k] * sign >= 0
        a = samples[keep]
        for i, layer in enumerate(layers):
            a = ops.forward(layer, a)
            if i in crown.bounded_layers(layers):
                lower, upper = out[crown.bounded_layers(layers).index(i)]
                flat = a.reshape(a.shape[0], -1)
                assert (flat >= lower - TOL).all()
                assert (flat <= upper + TOL).all()


def test_search_on_cnn():
    from search.bab import bab
    from search.dpll import dpll

    model = cnn_without_final_relu()
    x0 = cnn_point()
    label = int(model(x0.unsqueeze(0)).argmax())
    for search in (bab, dpll):
        assert search(model, x0, 0.05, label)[0] == 'unsat'
        status, witness = search(model, x0, 4.0, label)
        assert status == 'sat'
        assert (witness - x0).abs().max() <= 4.0 + TOL
        assert int(model(witness.unsqueeze(0)).argmax()) != label
