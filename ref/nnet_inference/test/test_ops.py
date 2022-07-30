import torch


# noinspection DuplicatedCode
@torch.no_grad()
def conv2d_forward(x, weight, bias, stride=1, padding=0, dilation=1):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    weight_h, weight_w = weight.shape[-2:]

    n_batches, n_in_channels, in_h, in_w = x.shape
    n_out_channels = weight.shape[0]

    out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

    n_weight_grps = n_in_channels // weight.shape[1]
    in_c_per_weight_grp = weight.shape[1]
    out_c_per_weight_grp = n_out_channels // n_weight_grps

    out = torch.zeros(n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    for b in range(n_batches):
        for c_out in range(n_out_channels):
            weight_grp = c_out // out_c_per_weight_grp
            for i in range(out_h):
                for j in range(out_w):
                    for di in range(weight_h):
                        for dj in range(weight_w):
                            for c in range(in_c_per_weight_grp):
                                pi = stride_h * i - pad_h + dil_h * di
                                pj = stride_w * j - pad_w + dil_w * dj
                                if 0 <= pi < in_h and 0 <= pj < in_w:
                                    c_in = weight_grp * in_c_per_weight_grp + c
                                    out[b, c_out, i, j] += \
                                        weight[c_out, c, di, dj].item() * x[b, c_in, pi, pj].item()
    out += bias.view(1, n_out_channels, 1, 1)
    return out


# noinspection DuplicatedCode
@torch.no_grad()
def maxpool2d_forward(x, kernel_size, stride=1, padding=0, dilation=1):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    n_batches, n_in_channels, in_h, in_w = x.shape

    out_h = (in_h + 2 * pad_h - (dil_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dil_w * (kernel_w - 1) + 1)) // stride_w + 1

    out = torch.empty(n_batches, n_in_channels, out_h, out_w, device=x.device, dtype=x.dtype)
    for b in range(n_batches):
        for c in range(n_in_channels):
            for i in range(out_h):
                for j in range(out_w):
                    vals = []
                    for di in range(kernel_h):
                        for dj in range(kernel_w):
                            pi = stride_h * i - pad_h + dil_h * di
                            pj = stride_w * j - pad_w + dil_w * dj
                            if 0 <= pi < in_h and 0 <= pj < in_w:
                                vals.append(x[b, c, pi, pj].item())
                    out[b, c, i, j] = max(vals)
    return out


def test_conv2d():
    x = torch.randn(1, 4, 8, 8)
    conv = torch.nn.Conv2d(
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(2, 1),
        groups=2,
        bias=True,
    )

    print(conv.weight.shape)

    y = conv2d_forward(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation)
    print(y.shape)
    print(torch.allclose(y, conv(x), atol=1e-5))


def test_maxpool2d():
    x = torch.randn(1, 4, 8, 8)
    pool = torch.nn.MaxPool2d(
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dilation=(2, 1)
    )

    y = maxpool2d_forward(x, pool.kernel_size, pool.stride, pool.padding, pool.dilation)
    print(y.shape)
    print(torch.allclose(y, pool(x), atol=1e-5))


if __name__ == '__main__':
    test_conv2d()
    test_maxpool2d()
