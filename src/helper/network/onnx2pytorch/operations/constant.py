from torch import nn
import numpy as np
import torch


class Constant(nn.Module):
    
    def __init__(self, constant):
        super().__init__()
        self.register_buffer("constant", torch.from_numpy(np.copy(constant)))

    def forward(self):
        return self.constant

    def extra_repr(self) -> str:
        return "constant={}".format(self.constant)
