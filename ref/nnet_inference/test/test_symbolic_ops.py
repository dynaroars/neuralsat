import torch


@torch.no_grad()
def evaluate(eq_lower, eq_upper, input_lower, input_upper):
    output_shape = eq_lower.size()[1:]
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    eq_lower = eq_lower.flatten(1)
    eq_upper = eq_upper.flatten(1)
    output_lower = pos(eq_upper[:-1]) * input_lower + neg(eq_lower[:-1]) * input_upper
    output_upper = pos(eq_upper[:-1]) * input_upper + neg(eq_lower[:-1]) * input_lower
    output_lower = output_lower.sum(0) + eq_lower[-1]
    output_upper = output_upper.sum(0) + eq_upper[-1]
    return output_lower.view(output_shape), output_upper.view(output_shape)


@torch.no_grad()
def pos(x):
    return torch.clamp(x, 0, torch.inf)


@torch.no_grad()
def neg(x):
    return torch.clamp(x, -torch.inf, 0)


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


def main():
    x = torch.rand(1, 1, 4, 4)

    eq_lower = torch.zeros(x.numel() + 1, *x.shape[1:])
    eq_upper = torch.zeros(x.numel() + 1, *x.shape[1:])
    eq_lower.flatten(1).fill_diagonal_(1)
    eq_upper.flatten(1).fill_diagonal_(1)

    conv = torch.nn.Conv2d(1, 2, (2, 2))

    output_eq_lower, output_eq_upper = conv2d_eq_forward(
        eq_lower, eq_upper, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation
    )
    print(torch.allclose(output_eq_lower, output_eq_upper))

    output_lower, output_upper = evaluate(output_eq_lower, output_eq_upper, x[0], x[0])
    print(output_lower)

    y = conv(x)
    print(y == output_lower)


if __name__ == '__main__':
    main()
