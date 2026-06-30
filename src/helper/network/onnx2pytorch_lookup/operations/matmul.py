import torch
from torch import nn


class MatMul(nn.Module):
    def forward(self, A, V):
        if V.ndim == 1:
            return (A * V).sum(dim=-1)
        else:
            return torch.matmul(A, V)
