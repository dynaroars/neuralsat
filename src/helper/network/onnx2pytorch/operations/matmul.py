from torch import nn
import torch


class MatMul(nn.Module):
    
    def forward(self, A, V):
        if V.ndim == 1:
            return (A * V).sum(dim=-1)
        return torch.matmul(A, V)
