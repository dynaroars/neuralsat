import torch.nn as nn

__all__ = ['CachedReLU']


class CachedReLU(nn.Module):
    def __init__(self):
        super(CachedReLU, self).__init__()
        self.decision = None

    def forward(self, x):
        self.decision = x.gt(0).type_as(x)
        return x * self.decision
