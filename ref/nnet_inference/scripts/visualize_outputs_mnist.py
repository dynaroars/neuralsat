import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model', choices=['fnn', 'cnn'], default='fnn')
    parser.add_argument('--algo', choices=['prophecy', 'nninfer'], default='nninfer')

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


def to_cv_img(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return tensor.permute(-2, -1, 0).mul(255.).round_().numpy().astype(np.uint8)


def main():
    args = parse_args()

    output_dir = os.path.join(args.output_dir, f'mnist_{args.model}')
    assert os.path.isdir(output_dir), f'Output dir {args.output_dir} not exists.'
    for sample_id in range(len(os.listdir(output_dir))):
        sample_output_file = os.path.join(output_dir, f'{sample_id:06d}.pt')
        sample_output = torch.load(sample_output_file)
        X = cv2.resize(to_cv_img(sample_output['X']),
                       None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        input_low = cv2.resize(to_cv_img(sample_output['input_low']),
                               None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        input_high = cv2.resize(to_cv_img(sample_output['input_high']),
                                None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        vis_img = np.ones((X.shape[0], X.shape[1] * 3 + 4), dtype=np.uint8) * 255
        vis_img[:, :X.shape[1]] = X
        vis_img[:, X.shape[1] + 2:2 * X.shape[1] + 2] = input_low
        vis_img[:, -X.shape[1]:] = input_high

        # if sample_id == 1:
        #     # gen so 0
        #     cv2.imwrite('outputs/sample.jpg', X)
        #     cv2.imwrite('outputs/lower_bound.jpg', input_low)
        #     cv2.imwrite('outputs/upper_bound.jpg', input_high)
        #     cv2.imwrite('outputs/all.jpg', vis_img)
        cv2.imshow('input', vis_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
