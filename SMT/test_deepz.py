import torch
import numpy as np
# import pypolycontain as pp
import matplotlib.pyplot as plt
from abstract.deepz import deepz, network
from abstract.reluval import reluval


def plot_z():
    x=np.array([1,2]).reshape(2,1) # offset
    G=np.array([[1,0.5],[2,3], [-5,-0.5]]).reshape(2,3)
    C=pp.zonotope(x=x,G=G)
    pp.visualize([C],title=r'$C$')
    plt.show()

def test():

    torch.manual_seed(0)
    net = network.FC(input_size=2, hidden_sizes=[3, 4, 5, 6, 5])
    # net = network.CorinaNet()

    # x = torch.rand([1, 3])
    # print(net(x))
    # x = x / x.abs().max()

    # torch.onnx.export(net, x, 'test.onnx')

    # print(x)

    # eps = 0.01

    lower = torch.Tensor([-5, -4,])
    upper = torch.Tensor([-1, -2,])
    center, error = deepz.forward(net, lower, upper)
    deepz.print_bound('Deepz', center, error)

    
    # upper = torch.Tensor([-1, -2, -3])
    # lower = torch.Tensor([-3, -4, -10])
    # center, error = deepz.forward(net, lower, upper)
    # deepz.print_bound('Random', center, error)


    # upper = torch.Tensor([-3, -2, -3])
    # lower = torch.Tensor([-5, -4, -10])
    # center, error = deepz.forward(net, lower, upper)
    # deepz.print_bound('Random', center, error)

    lower = torch.Tensor([-5, -4,])
    upper = torch.Tensor([-1, -2,])
    lb, ub = reluval.forward(net, lower, upper)
    print('Reluval')
    print('\t- lb:', lb.data)
    print('\t- ub:', ub.data)



if __name__ == '__main__':
    test()