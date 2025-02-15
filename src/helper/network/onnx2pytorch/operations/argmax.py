from torch import nn
import torch


class Argmax(nn.Module):
    
    def __init__(self, axis=None):
        self.axis = axis
        super().__init__()

    def forward(self, data):
        return torch.argmax(data, dim=self.axis)
