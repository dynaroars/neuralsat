import torch

from utils.read_nnet import Network, NetworkDeepZono
from dnn_solver.utils import InputParser
from utils.read_nnet import read_nnet
from abstract.deepz import deepz


if __name__ == '__main__':
    # w, b, lb, ub, means, ranges = read_nnet('example/random.nnet', with_norm=True)

    # print(lb)

    # for i in w:
    #     print(i.transpose())
    #     print(i.shape)
    #     print()

    # for i in b:
    #     print(i.shape)

    # dnn = Network('benchmark/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet')

    # print(dnn.input_lower_bounds)
    # print(dnn.input_upper_bounds)

    # print(dnn.input_means)
    # print(dnn.input_ranges)
    
    # print(dnn.output_mean)
    # print(dnn.output_range)


    # for layer_id, layer in enumerate(dnn.layers):
    #     weight, bias = layer.get_weights()
    #     print(weight.shape, bias.shape)


    # vars_mapping, layers_mapping = InputParser.parse(dnn)

    # print(vars_mapping)
    # print(layers_mapping)


    dnn = NetworkDeepZono('benchmark/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet')
    for layer in dnn.layers:
        print(layer)


    upper = torch.Tensor([-1, -2, -3, 1, 2])
    lower = torch.Tensor([-5, -4, -10, 3, 4])

    center, error = deepz.forward_nnet(dnn, lower, upper)

    deepz.print_bound('test_nnet', center, error)
