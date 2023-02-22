from torch.nn import functional as F
import torch.nn as nn
import torch


class MnistFc(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        self.input_shape = (1, 784)
        self.output_shape = (1, 10)
        
    def forward(self, x):
        return self.layers(x)