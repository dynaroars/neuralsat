from utils.read_nnet import read_nnet
from utils.read_nnet import Network
from dnn_solver.utils import InputParser

if __name__ == '__main__':
    w, b = read_nnet('example/random.nnet')

    # for i in w:
    #     print(i.transpose())
    #     print(i.shape)
    #     print()

    # for i in b:
    #     print(i.shape)

    dnn = Network('benchmark/acasxu/nnet/ACASXU_run2a_1_1_batch_2000.nnet')


    for layer_id, layer in enumerate(dnn.layers):
        weight, bias = layer.get_weights()
        print(weight.shape, bias.shape)


    # vars_mapping, layers_mapping = InputParser.parse(dnn)

    # print(vars_mapping)
    # print(layers_mapping)
