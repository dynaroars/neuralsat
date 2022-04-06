import torch
import numpy as np
# import pypolycontain as pp
import matplotlib.pyplot as plt
from abstract.deepz import deepz, network
from abstract.reluval import reluval
# from abstract.eran import eran
import time

from utils.read_nnet import NetworkDeepZono


def plot_z():
    x=np.array([1,2]).reshape(2,1) # offset
    G=np.array([[1,0.5],[2,3], [-5,-0.5]]).reshape(2,3)
    C=pp.zonotope(x=x,G=G)
    pp.visualize([C],title=r'$C$')
    plt.show()

def test():

    torch.manual_seed(0)
    # net = network.CorinaNet()
    net = network.FC(input_size=2, hidden_sizes=[3, 4, 5, 6, 2])

    lower = torch.Tensor([-5, -4])
    upper = torch.Tensor([-1, -2])

    lbs, ubs = deepz.forward(net, lower, upper)

    # x = torch.rand([1, net.input_size])
    # print(net(x))

    # x = x / x.abs().max()

    # torch.onnx.export(net, x, 'example/test.onnx')

    # print(x)

    # eps = 0.01
    # lower = torch.Tensor([-5, -4, -1, -0.2, -0.3])
    # upper = torch.Tensor([-1, -2, 1, 0.5, 1.5])

    # path = 'benchmark/acasxu/nnet/ACASXU_run2a_1_1_batch_2000'
    # net = NetworkDeepZono(path + '.nnet')

    # tic = time.time()
    # lbs, ubs = deepz.forward(net, lower, upper)
    # print('DeepZ', time.time() - tic)
    # print('lbs:', lbs)
    # print('ubs:', ubs)


    # cac = eran.ERAN('example/test.onnx', 'deeppoly')
    # cac = eran.ERAN(path + '.onnx', 'deeppoly')

    # tic = time.time()
    # lbs, ubs = cac(lower, upper)
    # print('DeepPoly', time.time() - tic)
    # print('lbs:', lbs)
    # print('ubs:', ubs)


if __name__ == '__main__':
    test()