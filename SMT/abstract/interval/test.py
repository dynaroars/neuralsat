import numpy as np
import torch
import os

from symbolic_network import IntervalNetwork
from interval import SymbolicInterval
from network import TestNetwork

def main():
    net = TestNetwork().sequential

    inet = IntervalNetwork(net)
    ix = SymbolicInterval(
        lower=torch.tensor([[-1, 1]], dtype=torch.float64),
        upper=torch.tensor([[6, 5]], dtype=torch.float64)
    )

    ic = inet(ix)
    print(ic.l.item(), ic.u.item())
    print(ic)

if __name__ == '__main__':
    main()