from dnn_solver.spec import SpecificationVNNLIB
from utils.read_vnnlib import read_vnnlib_simple


if __name__ == '__main__':
    spec_name = 'benchmark/acasxu/spec/prop_7.vnnlib'

    spec_list = read_vnnlib_simple(spec_name, 5, 5)

    for i, s in enumerate(spec_list):
        # print(s)
        specification = SpecificationVNNLIB(s)

        print(specification.get_input_property())
        print(specification.get_output_property(None))