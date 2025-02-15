from torch import nn
import torch


class MatMul(nn.Module):
    
    def forward(self, A, V):
        return torch.matmul(A, V)
