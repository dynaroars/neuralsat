from dnn_solver.spec import SpecificationVNNLIB, Specification
from utils.read_vnnlib import read_vnnlib_simple

import gurobipy as grb

if __name__ == '__main__':
    spec_name = 'benchmark/acasxu/spec/prop_5.vnnlib'

    spec_list = read_vnnlib_simple(spec_name, 5, 5)
    model = grb.Model()

    # old_spec = Specification(7, {'ubs': [], 'lbs': []})

    x1 = model.addVar(name='x1', lb=-1, ub=1)
    x2 = model.addVar(name='x2', lb=-2, ub=2)
    x3 = model.addVar(name='x3', lb=-1, ub=1)
    x4 = model.addVar(name='x4', lb=-2, ub=2)
    x5 = model.addVar(name='x5', lb=-2, ub=2)

    model.update()

    output = [x1, x2, x3, x4, x5]

    for i, s in enumerate(spec_list):
        # print(s)
        specification = SpecificationVNNLIB(s)

        print(specification.get_input_property())
        dnf = specification.get_output_property(output)

        for cnf in dnf:
            for c in cnf:
                print('c', c)
            print()

        # dnf = old_spec.get_output_property(output)

        # for cnf in dnf:
        #     for c in cnf:
        #         print('c', c)
        #     print()
        lbs = [-1, -2, -3, -4, -5]
        ubs = [1, 2, 3, 4, -3.999999]
        dnf = specification.check_output_reachability(lbs, ubs)
        print(dnf)
