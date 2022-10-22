import torch.nn as nn
import torch

class FC(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        self.input_size = input_size
        layers = []
        prev_size = input_size
        for idx, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            if idx < len(hidden_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CorinaNet(nn.Module):

    def __init__(self):
        super().__init__()

        fc1 = nn.Linear(2, 2)
        fc1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, 1]]))
        fc1.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        fc2 = nn.Linear(2, 2)
        fc2.weight = torch.nn.Parameter(torch.Tensor([[0.5, -0.2], [-0.5, 0.1]]))
        fc2.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        fc3 = nn.Linear(2, 2)
        fc3.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [-1, 1]]))
        fc3.bias = torch.nn.Parameter(torch.Tensor([0, 0]))

        self.layers = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)


    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = CorinaNet()
    x = torch.Tensor([1, -0.5])
    print(net(x))