from utils.read_vnnlib import read_vnnlib_simple
from heuristic.fast_falsify import fast_falsify
from utils.read_nnet import NetworkTorch
import time

if __name__ == '__main__':
    spec_name = 'benchmark/acasxu/spec/prop_7.vnnlib'
    nnet_name = 'benchmark/acasxu/nnet/ACASXU_run2a_1_9_batch_2000.nnet'

    net = NetworkTorch(nnet_name)

    spec_list = read_vnnlib_simple(spec_name, net.input_shape[1], net.output_shape[1])
    # for i, s in enumerate(spec_list):
    #     print(i, s[0])
    #     print(i, s[1])

    ff = fast_falsify.FastFalsify(net, spec_list)

    tic = time.time()
    stat, adv = ff.eval(timeout=500)

    print(stat, time.time() - tic)
    if stat == 'violated':
        print('input :', adv[0])
        print('output:', adv[1])
        print('output:', adv[1].numpy().tolist())