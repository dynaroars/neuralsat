from utils.read_vnnlib import read_vnnlib_simple
from heuristic.fast_falsify import fast_falsify
from utils.read_nnet import Network


if __name__ == '__main__':
    nnet_name = 'benchmark/acasxu/nnet/ACASXU_run2a_1_7_batch_2000.nnet'
    spec_name = 'benchmark/acasxu/spec/prop_3.vnnlib'

    net = Network(nnet_name)

    spec_list = read_vnnlib_simple(spec_name, net.input_shape[1], net.output_shape[1])
    # print(spec_list)

    ff = fast_falsify.FastFalsify(net, spec_list)

    stat, adv = ff.eval()

    print(stat)
    if stat == 'violated':
        print('input :', adv[0])
        print('output:', adv[1])