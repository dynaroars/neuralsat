from torch import nn
import torch

class Tile(nn.Module):
    
    def forward(self, input: torch.Tensor, repeats: torch.Tensor):
        return torch.tile(input, tuple(repeats))
