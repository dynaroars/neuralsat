import torch
import torch.nn as nn

from prophecy import ProphecyAnalyzer, NNInferAnalyzer

import argparse


class CorinaNet(nn.Module):
    def __init__(self):
        super(CorinaNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
        ])
        self._init_weights()

    def _init_weights(self):
        self.layers[0].weight.data = torch.tensor([
            [1., -1.],
            [1., 1.]
        ])
        self.layers[0].bias.data.fill_(0)
        self.layers[2].weight.data = torch.tensor([
            [0.5, -0.2],
            [-0.5, 0.1]
        ])
        self.layers[2].bias.data.fill_(0)
        self.layers[4].weight.data = torch.tensor([
            [1., -1.],
            [-1., 1.]
        ])
        self.layers[4].bias.data.fill_(0)

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

    model = CorinaNet()
    if args.algo == 'prophecy':
        analyzer = ProphecyAnalyzer(
            model,
            input_shape=(2,),
            input_lower=torch.ones(2) * -10,
            input_upper=torch.ones(2) * 10,
            verbose=True,
        )
    elif args.algo == 'nninfer':
        analyzer = NNInferAnalyzer(
            model,
            input_shape=(2,),
            input_lower=torch.ones(2) * -10,
            input_upper=torch.ones(2) * 10,
            verbose=True,
        )
    else:
        raise NotImplementedError

    x = torch.tensor([2.5, 2.5], dtype=torch.float64)
    print('\nInput:', x)
    input_low, input_high = analyzer.infer(x, envelop=args.envelop)
    print('\nOutput:')
    print(input_high)
    print(input_low)


def main_check_grid():
    model = CorinaNet()
    analyzer = ProphecyAnalyzer(
        model,
        input_shape=(2,),
        input_lower=torch.ones(2) * -10,
        input_upper=torch.ones(2) * 10,
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
