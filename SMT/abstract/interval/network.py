import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNetwork(nn.Module):

    def __init__(self):
        super(TestNetwork, self).__init__()
        
        self.layers = []
        linear1 = nn.Linear(2, 2)
        linear1.weight = torch.nn.Parameter(torch.tensor([[2, 3], [-3, -1]], dtype=torch.float64), requires_grad=True)
        linear1.bias = None
        self.layers.append(linear1)
        self.layers.append(nn.ReLU())
        linear2 = nn.Linear(2, 1)
        linear2.weight = torch.nn.Parameter(torch.tensor([1,-1], dtype=torch.float64), requires_grad=True)
        linear2.bias = None
        self.layers.append(linear2)
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, state):
        output = self.sequential(state)
        return output