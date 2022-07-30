import torch

__all__ = [
    'evaluate',
    'numpy_evaluate',
    'pos',
    'neg',
    'linear_eq_forward',
    'conv2d_eq_forward',
    'flatten_eq_forward',
    'relu_eq_forward',
]


@torch.no_grad()
def evaluate(eq_lower, eq_upper, input_lower, input_upper):
    output_shape = eq_lower.shape[1:]
    eq_lower = eq_lower.flatten(1)
    eq_upper = eq_upper.flatten(1)
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    output_lower = pos(eq_upper[:-1]) * input_lower + neg(eq_lower[:-1]) * input_upper
    output_upper = pos(eq_upper[:-1]) * input_upper + neg(eq_lower[:-1]) * input_lower
    output_lower = output_lower.sum(0) + eq_lower[-1]
    output_upper = output_upper.sum(0) + eq_upper[-1]
    return output_lower.view(output_shape), output_upper.view(output_shape)


def numpy_evaluate(eq_lower, eq_upper, input_lower, input_upper):
    output_shape = eq_lower.shape[1:]
    eq_lower = eq_lower.flatten(1)
    eq_upper = eq_upper.flatten(1)
    input_lower = input_lower.reshape(-1, 1)
    input_upper = input_upper.reshape(-1, 1)
    output_lower = pos(eq_upper[:-1]).numpy() * input_lower + neg(eq_lower[:-1]).numpy() * input_upper
    output_upper = pos(eq_upper[:-1]).numpy() * input_upper + neg(eq_lower[:-1]).numpy() * input_lower
    output_lower = output_lower.sum(0) + eq_lower[-1].numpy()
    output_upper = output_upper.sum(0) + eq_upper[-1].numpy()
    return output_lower.reshape(output_shape), output_upper.reshape(output_shape)


@torch.no_grad()
def pos(x):
    return torch.clamp(x, 0, torch.inf)


@torch.no_grad()
def neg(x):
    return torch.clamp(x, -torch.inf, 0)


@torch.no_grad()
def linear_eq_forward(eq_lower, eq_upper, weight, bias=None):
    pos_weight, neg_weight = pos(weight), neg(weight)
    out_eq_upper = eq_upper @ pos_weight.T + eq_lower @ neg_weight.T
    out_eq_lower = eq_lower @ pos_weight.T + eq_upper @ neg_weight.T
    if bias is not None:
        out_eq_lower[-1] += bias
        out_eq_upper[-1] += bias
    return out_eq_lower, out_eq_upper


@torch.no_grad()
def conv2d_eq_forward(eq_lower, eq_upper, weight, bias=None, stride=1, padding=0, dilation=1):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    weight_h, weight_w = weight.shape[-2:]

    n_inputs, n_in_channels, in_h, in_w = eq_lower.shape
    n_out_channels = weight.shape[0]

    out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

    n_weight_grps = n_in_channels // weight.shape[1]
    in_c_per_weight_grp = weight.shape[1]
    out_c_per_weight_grp = n_out_channels // n_weight_grps

    pos_weight, neg_weight = pos(weight), neg(weight)
    output_eq_lower = eq_lower.new_zeros(n_inputs, n_out_channels, out_h, out_w)
    output_eq_upper = torch.zeros_like(output_eq_lower)
    for b in range(n_inputs):
        for c_out in range(n_out_channels):
            weight_grp = c_out // out_c_per_weight_grp
            for i in range(out_h):
                for j in range(out_w):
                    for di in range(weight_h):
                        for dj in range(weight_w):
                            for c in range(in_c_per_weight_grp):
                                c_in = weight_grp * in_c_per_weight_grp + c

                                pi = stride_h * i - pad_h + dil_h * di
                                pj = stride_w * j - pad_w + dil_w * dj
                                if 0 <= pi < in_h and 0 <= pj < in_w:
                                    v_lower = eq_lower[b, c_in, pi, pj].item()
                                    v_upper = eq_upper[b, c_in, pi, pj].item()

                                    output_eq_upper[b, c_out, i, j] += \
                                        (v_upper * pos_weight[c_out, c, di, dj] +
                                         v_lower * neg_weight[c_out, c, di, dj])
                                    output_eq_lower[b, c_out, i, j] += \
                                        (v_lower * pos_weight[c_out, c, di, dj] +
                                         v_upper * neg_weight[c_out, c, di, dj])
    if bias is not None:
        output_eq_lower[-1] += bias.view(-1, 1, 1)
        output_eq_upper[-1] += bias.view(-1, 1, 1)
    return output_eq_lower, output_eq_upper


@torch.no_grad()
def flatten_eq_forward(eq_lower, eq_upper, start_dim=1, end_dim=-1):
    out_eq_lower = eq_lower.flatten(start_dim, end_dim)
    out_eq_upper = eq_upper.flatten(start_dim, end_dim)
    return out_eq_lower, out_eq_upper


@torch.no_grad()
def relu_eq_forward(eq_lower, eq_upper, signature):
    # evaluate output ranges
    n_inputs = eq_lower.shape[0]
    output_eq_lower = eq_lower.clone()
    output_eq_upper = eq_upper.clone()

    for i, sig in enumerate(signature.flatten()):
        if not sig:  # <= 0
            output_eq_lower.reshape(n_inputs, -1)[:, i] = 0
            output_eq_upper.reshape(n_inputs, -1)[:, i] = 0
    return output_eq_lower, output_eq_upper
