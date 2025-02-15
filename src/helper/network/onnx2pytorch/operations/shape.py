from torch import nn
import torch


class Shape(nn.Module):
    
    def forward(self, input: torch.Tensor):
        return torch.tensor(input.shape, device=input.device)
