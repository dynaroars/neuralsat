import torch
import numpy as np
import pypolycontain as pp
import matplotlib.pyplot as plt
import network
import deepz


def plot_z():
    x=np.array([4,2,0]).reshape(3,1) # offset
    G=np.array([[1,0.5, 2],[2,0, 3], [2,0.5, 4], ]).reshape(3,3)
    C=pp.zonotope(x=x,G=G)
    pp.visualize([C],title=r'$C$')
    plt.show()

def test():

    torch.manual_seed(0)

    net = network.FC(input_size=4, hidden_sizes=[22, 21, 40, 5])

    x = torch.rand([1, 2, 2])
    x = x / x.abs().max()

    # print(x)

    eps = 0.01

    upper = torch.clamp(x.data + eps, max=1)
    lower = torch.clamp(x.data - eps, min=0)    # 

    # print(upper.shape)

    # print(net(x))

    res = deepz.forward(net, lower, upper)

    print(res)


if __name__ == '__main__':
    plot_z()