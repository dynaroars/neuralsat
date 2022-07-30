import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from prophecy import ProphecyAnalyzer, NNInferAnalyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model', choices=['fnn', 'cnn'], default='cnn')
    parser.add_argument('--algo', choices=['prophecy', 'nninfer'], default='prophecy')
    parser.add_argument('--envelop', action='store_true')

    args = parser.parse_args()
    return args


class MNISTFNN(nn.Module):
    def __init__(self, n_classes):
        super(MNISTFNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MNISTCNN(nn.Module):
    def __init__(self, n_classes):
        super(MNISTCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 4, (5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(4, 8, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(800, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(
        root=args.data_root,
        train=True,
        transform=transform,
        download=True)
    test_set = torchvision.datasets.MNIST(
        root=args.data_root,
        train=False,
        transform=transform,
        download=True)

    if args.model == 'fnn':
        model = MNISTFNN(n_classes=len(train_set.classes))
    elif args.model == 'cnn':
        model = MNISTCNN(n_classes=len(train_set.classes))
    else:
        raise NotImplementedError
    model_save_file = os.path.join(args.save_dir, f'mnist_{args.model}.pt')
    model.load_state_dict(torch.load(model_save_file, map_location='cpu'))

    input_shape = (1, 28, 28)
    if args.algo == 'prophecy':
        analyzer = ProphecyAnalyzer(
            model,
            input_shape=input_shape,
            input_lower=torch.zeros(input_shape, dtype=torch.float64),
            input_upper=torch.ones(input_shape, dtype=torch.float64),
        )
    elif args.algo == 'nninfer':
        analyzer = NNInferAnalyzer(
            model,
            input_shape=input_shape,
            input_lower=torch.zeros(input_shape, dtype=torch.float64),
            input_upper=torch.ones(input_shape, dtype=torch.float64),
        )
    else:
        raise NotImplementedError

    output_dir = os.path.join(args.output_dir, f'mnist_{args.model}')
    os.makedirs(output_dir, exist_ok=True)
    for sample_id, (X, y) in enumerate(tqdm(train_set)):
        X = X.to(torch.float64)
        input_low, input_high = analyzer.infer(X, envelop=args.envelop)

        sample_output_file = os.path.join(output_dir, f'{sample_id:06d}.pt')
        torch.save({
            'X': X,
            'input_low': input_low,
            'input_high': input_high,
        }, sample_output_file)


if __name__ == '__main__':
    main()
