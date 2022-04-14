import torch
import numpy as np
# import pypolycontain as pp
import matplotlib.pyplot as plt
from abstract.deepz import deepz, network, deeppoly
from abstract.reluval import reluval
# from abstract.eran import eran
import time
from abstract.neurify import neurify

from utils.read_nnet import NetworkDeepZono


def plot_z():
    x=np.array([1,2]).reshape(2,1) # offset
    G=np.array([[1,0.5],[2,3], [-5,-0.5]]).reshape(2,3)
    C=pp.zonotope(x=x,G=G)
    pp.visualize([C],title=r'$C$')
    plt.show()

def test():

    torch.manual_seed(1)
    net = network.CorinaNet()

    lower = torch.Tensor([-5, -4])
    upper = torch.Tensor([-1, -2])

    net = network.FC(input_size=5, hidden_sizes=[50, 50, 50, 50, 50,  5])
    lower = torch.Tensor([-5, -4, -1, -0.2, -0.3])
    upper = torch.Tensor([-1, -2, 1, 0.5, 1.5])


    tic = time.time()
    (lbs, ubs), _ = deepz.forward(net, lower, upper)
    print('DeepZ', time.time() - tic)
    print('lbs:', lbs)
    print('ubs:', ubs)
    print()

    tic = time.time()
    lbs, ubs = neurify.forward(net, lower, upper)
    print('Neurify', time.time() - tic)
    print('lbs:', lbs.data)
    print('ubs:', ubs.data)
    print()



    # dpf2 = deeppoly.build_verifier_network(net, lower.shape[0])
    # inp = deeppoly.DeepPoly(lower.shape[0], lower, upper)
    # out = dpf2(inp)
    # print(inp.lb.data)
    # print(inp.ub.data)



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