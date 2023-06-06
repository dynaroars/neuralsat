import torch

import haioc


def main():
    data = torch.randperm(70 * 10).sub_(70 * 10 // 2).view(70, 10).float()
    xs = torch.arange(0, 50).float()

    output = haioc.ops.any_eq_any(data, xs)
    print(output)
    print(haioc.__version__)


if __name__ == '__main__':
    main()
