import torch.nn as nn
import torch

class ZonotopeAbstraction(nn.Module):

    def __init__(self, net):
        super(ZonotopeAbstraction, self).__init__()