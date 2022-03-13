import torch.nn as nn
import torch

class FC(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        layers = [nn.Flatten()]
        prev_size = input_size
        for idx, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            if idx < len(hidden_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

