import torch
from torch import nn


class Transpose(nn.Module):
    def __init__(self, dims=None, quirks=None):
        self.dims = dims
        self.quirks = {} if quirks is None else quirks
        assert isinstance(self.quirks, dict)
        super().__init__()

    def forward(self, data: torch.Tensor):
        if not self.dims:
            dims = tuple(reversed(range(data.dim())))
        else:
            dims = self.dims
        # if the first dim is batch size, manually add the batch size to the shape
        if len(data.shape) == len(dims) + 1 and self.quirks.get("merge_batch_size_with_channel"):
            dims = tuple([0] + [tmp + 1 for tmp in dims])
        transposed = data.permute(dims)
        return transposed
