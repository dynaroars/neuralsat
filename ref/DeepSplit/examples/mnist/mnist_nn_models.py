from deepsplit.admm import Flatten
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def mnist_loaders(batch_size, shuffle_test=False):
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def mnist_fc():
    # fully-connected nn classifier for MNIST
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,600),
        nn.ReLU(),
        nn.Linear(600,400),
        nn.ReLU(),
        nn.Linear(400,200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model