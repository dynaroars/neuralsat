import torch

from utils.read_nnet import Network, NetworkTorch
from dnn_solver.utils import InputParser
from abstract.deepz import deepz


    
if __name__ == '__main__':

    dnn = Network('example/random.nnet')

    upper = torch.Tensor([-1, -2, -3, 1, 2])
    lower = torch.Tensor([-5, -4, -10, 3, 4])

    (lbs, ubs), _ = deepz.forward(dnn, lower, upper)

    print('lbs:', lbs)
    print('ubs:', ubs)

    print(dnn(lower).data)
    print(dnn(upper).data)


    dnn = NetworkTorch('example/random.nnet')
    print(dnn(lower).data)
    print(dnn(upper).data)

    (lbs, ubs), _ = deepz.forward(dnn, lower, upper)

    print('lbs:', lbs)
    print('ubs:', ubs)

    # dnn = Network('example/random.nnet')

    vars_mapping, layers_mapping = InputParser.parse(dnn)

    print(layers_mapping)
