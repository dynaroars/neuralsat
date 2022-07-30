'''
Fenerate vnnlib property files and acasxu_instances.csv for acasxu for vnncomp 2021

Stanley Bak, April 2021
'''

from typing import Tuple, List, Any

from math import pi

import numpy as np

def get_init_box(property_str):
    'get lb, ub lists for the given property'

    if property_str in ("1", "2"):
        init_lb = [55947.691, -pi, -pi, 1145, 0]
        init_ub = [60760, pi, pi, 1200, 60]
    elif property_str == "3":
        init_lb = [1500, -0.06, 3.1, 980, 960]
        init_ub = [1800, 0.06, pi, 1200, 1200]
    elif property_str == "4":
        init_lb = [1500, -0.06, 0, 1000, 700]
        init_ub = [1800, 0.06, 0, 1200, 800]
    elif property_str == "5":
        init_lb = [250, 0.2, -3.141592, 100, 0]
        init_ub = [400, 0.4, -3.141592 + 0.005, 400, 400]
    elif property_str == "6.1":
        init_lb = [12000, 0.7, -3.141592, 100, 0]
        init_ub = [62000, 3.141592, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "6.2":
        init_lb = [12000, -3.141592, -3.141592, 100, 0]
        init_ub = [62000, -0.7, -3.141592 + 0.005, 1200, 1200]
    elif property_str == "7":
        init_lb = [0, -3.141592, -3.141592, 100, 0]
        init_ub = [60760, 3.141592, 3.141592, 1200, 1200]
    elif property_str == "8":
        init_lb = [0, -3.141592, -0.1, 600, 600]
        init_ub = [60760, -0.75*3.141592, 0.1, 1200, 1200]
    elif property_str == "9":
        init_lb = [2000, -0.4, -3.141592, 100, 0]
        init_ub = [7000, -0.14, -3.141592 + 0.01, 150, 150]
    else:
        assert property_str == "10", f"unsupported property string: {property_str}"
        
        init_lb = [36000, 0.7, -3.141592, 900, 600]
        init_ub = [60760, 3.141592, -3.141592 + 0.01, 1200, 1200]
        
    return init_lb, init_ub

def is_minimal_spec(indices):
    '''create a (disjunctive) spec that a specific set of outputs is minimal

    returns a list of 2-tuples, where each 2-tuple is 
    mat, rhs
    '''

    if isinstance(indices, int):
        indices = [indices]

    spec_list = []

    for i in range(5):
        if i in indices: # index 3 is strong left
            continue

        mat = []
        rhs = []

        for index in indices:
            l = [0, 0, 0, 0, 0]
            l[index] = -1
            l[i] = 1
            
            mat.append(l)
            rhs.append(0)

        spec_list.append((mat, rhs))

    return spec_list

def get_spec(property_str) -> Tuple[str, List[Tuple[Any, Any]]]:
    'get the specificaion string description and a list of specification mat and rhs'

    #labels = ['Clear of Conflict (COC)', 'Weak Left', 'Weak Right', 'Strong Left', 'Strong Right']

    if property_str == "1":
        desc = 'Unsafe if COC >= 1500. Output scaling is 373.94992 with a bias of 7.518884: (1500 - 7.518884) ' + \
               '/ 373.94992 = 3.991125'
        output_scaling_mean = 7.5188840201005975
        output_scaling_range = 373.94992
        
        # (1500 - 7.518884) / 373.94992 = 3.991125
        threshold = (1500 - output_scaling_mean) / output_scaling_range
        mat = [[-1.0, 0, 0, 0, 0]]
        rhs = [-threshold]
        spec = (desc, [(mat, rhs)])
        
    elif property_str == "2":
        desc = 'Unsafe if COC is maximal'
        # y0 > y1 and y0 > y1 and y0 > y2 and y0 > y3 and y0 > y4
        mat = [[-1, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0],
               [-1, 0, 0, 1, 0],
               [-1, 0, 0, 0, 1]]
        rhs = [0, 0, 0, 0]

        spec = (desc, [(mat, rhs)])
        
    elif property_str in ("3", "4"):
        desc = 'Unsafe if COC is minimal'
        mat = [[1, -1, 0, 0, 0],
               [1, 0, -1, 0, 0],
               [1, 0, 0, -1, 0],
               [1, 0, 0, 0, -1]]
        rhs = [0, 0, 0, 0]

        spec = (desc, [(mat, rhs)])
    elif property_str == '5':
        desc = "unsafe if strong right is not minimal"
        spec = (desc, is_minimal_spec(4))
    elif property_str in ["6", "10"]:
        desc = "unsafe if coc is not minimal"
        spec = (desc, is_minimal_spec(0))
    elif property_str == "7":
        desc = "unsafe if strong left is minimial or strong right is minimal"

        mat1 = [[-1, 0, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, 0, -1, 1, 0]]
        rhs1 = [0.0, 0, 0]

        mat2 = [[-1, 0, 0, 0, 1],
                [0, -1, 0, 0, 1],
                [0, 0, -1, 0, 1]]
        rhs2 = [0, 0, 0]
        
        spec = (desc, [(mat1, rhs1), (mat2, rhs2)])
    elif property_str == "8":
        desc = "weak left is minimal or COC is minimal"
        spec = (desc, is_minimal_spec([0, 1]))
    else:
        assert property_str == "9"
        desc = "strong left should be minimal"
        spec = (desc, is_minimal_spec(3))
        
    return spec

def print_prop(num, f):
    'print prop to file f'

    f.write(f'; ACAS Xu property {num}\n\n')

    # declare constants
    for x in range(5):
        f.write(f"(declare-const X_{x} Real)\n")

    f.write("\n")

    for x in range(5):
        f.write(f"(declare-const Y_{x} Real)\n")

    means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]
    range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]

    # input bounds
    if num != "6":
        init_lb, init_ub = get_init_box(num)

        for i in range(len(init_lb)):
            f.write(f"\n; Unscaled Input {i}: {init_lb[i], init_ub[i]}\n")
            lb = (init_lb[i] - means_for_scaling[i]) / range_for_scaling[i]
            ub = (init_ub[i] - means_for_scaling[i]) / range_for_scaling[i]
            f.write(f"(assert (<= X_{i} {round(ub, 9)}))\n")
            f.write(f"(assert (>= X_{i} {round(lb, 9)}))\n")

        f.write("\n")
    else:
        # spec 6, special handling

        init_lb1, init_ub1 = get_init_box("6.1")
        init_lb2, init_ub2 = get_init_box("6.2")
        
        desc, mat_rhs_list = get_spec(num)

        f.write('\n; Spec 6\n')
        f.write(f'; {desc}\n')

        for i in range(len(init_lb1)):
            f.write(f"; Unscaled Input {i}: {init_lb1[i], init_ub1[i]}\n")

        f.write(';;;; or ;;;;\n')
        for i in range(len(init_lb1)):
            f.write(f"; Unscaled Input {i}: {init_lb2[i], init_ub2[i]}\n")

        f.write("(assert (or\n")

        for init_lb, init_ub in zip([init_lb1, init_lb2], [init_ub1, init_ub2]):
            f.write("    (and")

            for i in range(len(init_lb)):
                lb = (init_lb[i] - means_for_scaling[i]) / range_for_scaling[i]
                ub = (init_ub[i] - means_for_scaling[i]) / range_for_scaling[i]
                f.write(f" (<= X_{i} {round(ub, 9)})")
                f.write(f" (>= X_{i} {round(lb, 9)})")

            f.write(")\n") # close 'and'

        f.write("))\n\n") # close 'or' and 'assert'

    # print spec
    desc, mat_rhs_list = get_spec(num)

    f.write(f'; {desc}\n')

    if len(mat_rhs_list) == 1:
        mat, rhs_vec = mat_rhs_list[0]

        for row, rhs in zip(mat, rhs_vec):
            row = np.array(row, dtype=float)

            if rhs == 0:
                assert sum(row != 0) == 2
                i1, i2 = np.where(row)[0]

                if row[i1] == 1.0 and row[i2] == -1.0:
                    f.write(f"(assert (<= Y_{i1} Y_{i2}))\n")
                else:
                    assert row[i1] == -1.0 and row[i2] == 1.0
                    f.write(f"(assert (<= Y_{i2} Y_{i1}))\n")
            else:
                assert sum(row != 0) == 1

                i = np.argmax(np.abs(row))

                if row[i] > 0:
                    f.write(f"(assert (<= Y_{i} {rhs}))\n")
                else:
                    f.write(f"(assert (>= Y_{i} {-rhs}))\n")
    else:
        # disjunctive property
        f.write("(assert (or\n")

        for mat, rhs_vec in mat_rhs_list:
            f.write("    (and")

            for row, rhs in zip(mat, rhs_vec):
                row = np.array(row, dtype=float)
                assert rhs == 0, "only spec 1 has threshold"
                assert sum(row != 0) == 2
                i1, i2 = np.where(row)[0]

                if row[i1] == 1.0 and row[i2] == -1.0:
                    f.write(f" (<= Y_{i1} Y_{i2})")
                else:
                    assert row[i1] == -1.0 and row[i2] == 1.0
                    f.write(f" (<= Y_{i2} Y_{i1})")

            f.write(")\n")


        f.write("))\n")

def main():
    'main entry point'

    # generate spec files
    for spec in range(1, 11):
        filename = f"prop_{spec}.vnnlib"
        print(f"Generating {filename}...")
        
        with open(filename, 'w') as f:
            print_prop(str(spec), f)

    # generate csv
    total_timeout = 6 * 60 * 60 # 6 hours
    num_benchmarks = (4 * 5 * 9) + 6
    benchmark_timeout = int(round(total_timeout / num_benchmarks))
    filename = "acasxu_instances.csv"
    
    with open(filename, 'w') as f:
        print(f"Generating {filename} with benchmark timeout {benchmark_timeout}...")
        
        for prop_num in range(1, 5):
            p = f'prop_{prop_num}.vnnlib'
            for a in range(1, 6):
                for b in range(1, 10):
                    o = f'ACASXU_run2a_{a}_{b}_batch_2000.onnx'

                    f.write(f'{o},{p},{benchmark_timeout}\n')

        prop_net_list = [(5, 1, 1), (6, 1, 1), (7, 1, 9), (8, 2, 9), (9, 3, 3), (10, 4, 5)]

        for prop_num, a, b in prop_net_list:
            p = f'prop_{prop_num}.vnnlib'
            o = f'ACASXU_run2a_{a}_{b}_batch_2000.onnx'

            f.write(f'{o},{p},{benchmark_timeout}\n')


if __name__ == "__main__":
    main()
