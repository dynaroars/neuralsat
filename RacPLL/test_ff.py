from heuristic.falsification import randomized_falsification
from utils.read_vnnlib import read_vnnlib_simple
from dnn_solver.spec import SpecificationVNNLIB
from utils.dnn_parser import DNNParser
import time
import torch
if __name__ == '__main__':
    nnet_name = 'benchmark/acasxu/nnet/ACASXU_run2a_2_9_batch_2000.nnet'
    spec_name = 'benchmark/acasxu/spec/prop_7.vnnlib'


    nnet_name = 'benchmark/mnistfc/nnet/mnist-net_256x2.onnx'
    spec_name = 'benchmark/mnistfc/spec/prop_7_0.03.vnnlib'


    nnet_name = 'benchmark/mnistfc/nnet/mnist-net_256x2.onnx'
    spec_name = 'benchmark/mnistfc/spec/prop_0_0.05.vnnlib'


    # nnet_name = 'example/random.nnet'
    device = torch.device('cpu')

    net = DNNParser.parse(nnet_name, 'mnistfc', device)

    spec_list = read_vnnlib_simple(spec_name, net.n_input, net.n_output)

    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        ff = randomized_falsification.RandomizedFalsification(net, spec)

        tic = time.time()
        stat, adv = ff.eval(timeout=10)

        print(stat, time.time() - tic)
        if stat == 'violated':
            print('input :', adv[0])
            print('output:', adv[1].numpy().tolist())
            break