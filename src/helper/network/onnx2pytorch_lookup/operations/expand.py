import torch
from torch import nn


class Expand(nn.Module):
    def forward(self, input: torch.Tensor, shape: torch.Tensor):
        if isinstance(shape, torch.Tensor) and (shape == 1).all():
            return input * torch.ones(
                torch.Size(shape), dtype=input.dtype, device=input.device)
        else:
            return input.expand(torch.Size(shape))
