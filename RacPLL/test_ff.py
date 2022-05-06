from utils.read_vnnlib import read_vnnlib_simple
from heuristic.fast_falsify import fast_falsify
from utils.read_nnet import NetworkTorch


if __name__ == '__main__':
    spec_name = 'benchmark/acasxu/spec/prop_6.vnnlib'
    nnet_name = 'benchmark/acasxu/nnet/ACASXU_run2a_4_5_batch_2000.nnet'

    net = NetworkTorch(nnet_name)

    spec_list = read_vnnlib_simple(spec_name, net.input_shape[1], net.output_shape[1])

    for i, s in enumerate(spec_list):
        print(i, len(s))
        bounds = s[0]
        print('bounds:', bounds)
        spec = s[1]
        print('spec:', len(spec))
        for prop_mat, prop_rhs in spec:
            print('\t+ mat:', prop_mat)
            print('\t+ rhs:', prop_rhs)
    exit()

    ff = fast_falsify.FastFalsify(net, spec_list)

    stat, adv = ff.eval(timeout=60)

    print(stat)
    if stat == 'violated':
        print('input :', adv[0])
        print('output:', adv[1])
        print('output:', adv[1].numpy().tolist())