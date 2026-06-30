import torch
from torch import nn


class Shape(nn.Module):
    def forward(self, input: torch.Tensor):
        return torch._shape_as_tensor(input).to(input.device)
