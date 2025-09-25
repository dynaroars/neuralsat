from torch import nn
import torch


class Shape(nn.Module):
    
    def forward(self, input: torch.Tensor):
        return torch._shape_as_tensor(input).to(input.device)
