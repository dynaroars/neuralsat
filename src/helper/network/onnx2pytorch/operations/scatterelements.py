from torch import nn
import torch


# class ScatterElements(nn.Module):
    
#     def __init__(self, dim=0):
#         self.dim = dim
#         super().__init__()

#     def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
#         indices[indices < 0] = indices[indices < 0] + data.size(self.dim)
#         return torch.scatter(data, self.dim, indices, updates)

class ScatterElements(nn.Module):

    def __init__(self, dim=0, reduction='add'):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        indices = torch.where(indices < 0, indices + data.size(self.dim), indices)
        return torch.scatter(data, self.dim, indices, updates, reduce=self.reduction)