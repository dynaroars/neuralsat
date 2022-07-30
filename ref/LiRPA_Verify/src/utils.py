from modules import Flatten
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

########################################
# Defined the model architectures
########################################

def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(1024, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_wide():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def load_model_data(args):
    model_ori = eval(args.model)()
    print(model_ori)
    print("[no_LP]:", args.no_LP)
    # loaded_model = torch.load(args.load)
    model_ori.load_state_dict(torch.load(args.load)['state_dict'][0])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    test_data = datasets.CIFAR10("../data", train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor([0.485, 0.456, 0.406])
    test_data.std = torch.tensor([0.225, 0.225, 0.225])
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))

    if args.data == 'CIFAR-deep':
        gt_results = pd.read_pickle('../data/deep.pkl')
    elif args.data == 'CIFAR-wide':
        gt_results = pd.read_pickle('../data/wide.pkl')
    elif args.data == 'CIFAR-easy':
        gt_results = pd.read_pickle('../data/base_easy.pkl')
    elif args.data == 'CIFAR-med':
        gt_results = pd.read_pickle('../data/base_med.pkl')
    elif args.data == 'CIFAR-hard':
        gt_results = pd.read_pickle('../data/base_hard.pkl')

    return model_ori, gt_results, test_data, data_min, data_max
