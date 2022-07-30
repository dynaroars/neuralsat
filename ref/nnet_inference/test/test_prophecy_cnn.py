import torch
import torch.nn as nn

from prophecy import ProphecyAnalyzer, NNInferAnalyzer
import argparse


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 2, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(2, 4, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, 2),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['prophecy', 'nninfer'], default='prophecy')
    parser.add_argument('--envelop', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = SimpleCNN()
    if args.algo == 'prophecy':
        analyzer = ProphecyAnalyzer(
            model,
            input_shape=(1, 6, 6),
            input_lower=torch.ones(1, 6, 6) * -1,
            input_upper=torch.ones(1, 6, 6) * 1,
            verbose=True,
        )
    elif args.algo == 'nninfer':
        analyzer = NNInferAnalyzer(
            model,
            input_shape=(1, 6, 6),
            input_lower=torch.ones(1, 6, 6) * -1,
            input_upper=torch.ones(1, 6, 6) * 1,
            verbose=True,
        )
    else:
        raise NotImplementedError

    x = torch.randn(analyzer.input_shape, dtype=torch.float64)
    print('\nInput:', x)
    input_low, input_high = analyzer.infer(x, envelop=args.envelop)
    print('\nOutput:')
    print(input_high)
    print(input_low)
    print(input_high - input_low)


def main_check_grid():
    model = SimpleCNN()
    analyzer = ProphecyAnalyzer(
        model,
        input_shape=(1, 6, 6),
        input_lower=torch.ones(1, 6, 6) * -1,
        input_upper=torch.ones(1, 6, 6) * 1,
    )

    x_grid = torch.stack(
        torch.meshgrid(*[torch.linspace(lb, ub, 11, dtype=torch.float64)
                         for lb, ub in zip(analyzer.input_lower, analyzer.input_upper)]),
        dim=-1).view(-1, *analyzer.input_shape)
    for x in x_grid:
        print('\nInput:', x)
        input_low, input_high = analyzer.infer(x)
        print('\nOutput:')
        print(input_high)
        print(input_low)


if __name__ == '__main__':
    main()
    # main_check_grid()
