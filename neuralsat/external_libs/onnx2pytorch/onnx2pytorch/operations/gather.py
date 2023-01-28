import torch
from torch import nn


class Gather(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        selection = self.selection + [indices.to(torch.int64)]
        # return data.__getitem__(selection)
        if indices == -1:
            indices = torch.tensor(data.shape[self.dim] - 1)

        if indices.ndim == 0:
            return torch.index_select(data, dim=self.dim, index=indices).squeeze(self.dim)

        return torch.index_select(data, dim=self.dim, index=indices)

