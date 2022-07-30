
"""
A simple MNIST model used for testing the verification.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as trans
import torch.nn.functional as functional

from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode


# noinspection PyPep8Naming,PyShadowingNames,PyTypeChecker
class MNISTNN(VeriNetNN):

    """
    A simple fully-connected network for the MNIST dataset.
    """

    def __init__(self, use_gpu: bool = False, act: str = "ReLU"):

        """
        Args:
            use_gpu:
                If true, and a GPU is available, the GPU is used, else the CPU is used.
            act:
                The activation function, "ReLU or "PReLU",
        """

        if act == "ReLU":
            layers = [
                VeriNetNNNode(0, nn.Identity(), [], [1]),
                VeriNetNNNode(1, nn.Linear(784, 256), [0], [2]),
                VeriNetNNNode(2, nn.ReLU(), [1], [3]),
                VeriNetNNNode(3, nn.Linear(256, 256), [2], [4]),
                VeriNetNNNode(4, nn.ReLU(), [3], [5]),
                VeriNetNNNode(5, nn.Linear(256, 10), [4], [])
            ]

        else:
            raise ValueError(f"Activation function '{act}' not recognised")

        super().__init__(layers, use_gpu=use_gpu)

        self.dset_train = None
        self.dset_val = None
        self.dset_test = None
        self.loader_train = None
        self.loader_val = None
        self.loader_test = None

    def init_data_loader(self, data_dir: str, num_train: int = 45000, normalise: int = False):

        """
        Initializes the data loaders.

        If the data isn't found, it will be downloaded.

        Args:
            data_dir:
                The directory of the data.
            num_train:
                The number of training examples used.
            normalise:
                If true, the dataset is normalised.
        """

        if normalise:
            mean = (0.1307,)
            std = (0.3081,)
            trns_norm = trans.Compose([trans.ToTensor(), trans.Normalize(mean, std)])
        else:
            trns_norm = trans.ToTensor()

        self.dset_train = dset.MNIST(data_dir, train=True, download=True, transform=trns_norm)
        self.loader_train = DataLoader(self.dset_train, batch_size=64,
                                       sampler=sampler.SubsetRandomSampler(range(num_train)))

        self.dset_val = dset.MNIST(data_dir, train=True, download=True, transform=trns_norm)
        self.loader_val = DataLoader(self.dset_val, batch_size=64,
                                     sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))

        self.dset_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)
        self.loader_test = DataLoader(self.dset_test, batch_size=100)

    def train_model(self, optimiser: optim, epochs: int = 1, l1_reg: float = 0, l2_reg: float = 0):

        """
        Trains the model.

        Args:
            optimiser:
                The torch optimiser
            epochs:
                The number of epochs to train the model.
            l1_reg:
                The l1 regularization multiplier.
            l2_reg:
                The l2 regularization multiplier.
        """

        msg = "Initialize data loaders before calling train_model"
        assert (self.loader_train is not None) and (self.loader_val is not None), msg

        for epoch in range(epochs):

            print(f"Training epoch: {epoch + 1}/{epochs}")
            iter_train = iter(self.loader_train)

            num_correct = 0
            num_samples = 0

            pbar = tqdm.tqdm(range(len(self.loader_train)))
            pbar.set_description(f"Training Accuracy: {0}")

            for _ in pbar:

                x, y = iter_train.next()
                x = x.reshape((x.shape[0], -1))

                self.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = functional.log_softmax(self(x)[0], dim=1)

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                train_acc = num_correct / num_samples
                pbar.set_description(f"Epoch Accuracy: {train_acc:.2f}")

                l1_regularization_loss = 0
                l2_regularization_loss = 0
                for param in self.parameters():
                    l1_regularization_loss += torch.sum(torch.abs(param))
                    l2_regularization_loss += torch.sum(param ** 2)

                loss = (functional.cross_entropy(scores, y) +
                        l1_reg * l1_regularization_loss +
                        l2_reg * l2_regularization_loss)

                for layer in self.layers:
                    if isinstance(layer, nn.PReLU):
                        loss -= 0.1*torch.mean(layer.weight)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                for layer in self.layers:
                    if isinstance(layer, nn.PReLU):
                        layer.weight.data = torch.clip(layer.weight, 0, 1)

            val_acc = self.check_accuracy(dataset=1)[2]

            if epoch % 5 == 0:
                print(f"Validation accuracy: {val_acc:.4f}")

        test_accuracy = self.check_accuracy(dataset=2)[2]
        print(f"Final test set accuracy: {test_accuracy:.4f}")

    def check_accuracy(self, dataset: int = 0) -> tuple:

        """
        Calculates and returns the accuracy of the current model.
        Args:
             dataset:
                0 for training data, 1 for validation data, 2 for test data.
        Returns:
            (num_correct, num_samples, accuracy). The number of correct
            classifications, the total number of samples and the accuracy in percent.
        """

        if dataset == 0:
            loader = self.loader_train
        elif dataset == 1:
            loader = self.loader_val
        elif dataset == 2:
            loader = self.loader_test
        else:
            raise ValueError("Para dataset should be 0 for training, 1 for validation and 2 for test")

        num_correct = 0
        num_samples = 0

        self.eval()

        with torch.no_grad():
            for x, y in loader:

                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x = x.reshape((x.shape[0], -1))

                # noinspection PyCallingNonCallable
                scores = self(x)[0]
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            acc = float(num_correct) / num_samples

        return num_correct.cpu().detach().numpy(), num_samples, acc


if __name__ == '__main__':

    model_path = "../../resources/models/onnx/mnist_nn.onnx"
    model = MNISTNN(use_gpu=True, act="PReLU")
    print(f"Device used: {model.device}")

    for layer in model.layers:
        if isinstance(layer, nn.PReLU):
            print(layer.weight)

    model.init_data_loader("../../resources/images/mnist_full/")

    optimiser = torch.optim.Adam(params=model.parameters(), weight_decay=1e-4, lr=1e-3)
    model.train_model(optimiser, epochs=5)

    for layer in model.layers:
        if isinstance(layer, nn.PReLU):
            print(layer.weight)

    model.save_sd(path=model_path)

    model = MNISTNN(use_gpu=True, act="PReLU")

    # noinspection PyTypeChecker
    model.load_sd(model_path)

    for layer in model.layers:
        if isinstance(layer, nn.PReLU):
            print(layer.weight)
