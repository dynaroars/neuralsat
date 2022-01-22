from simplex.simplex import Simplex
from simplex.parser import Parser
# from real_solver.bak import RealSolver
from real_solver.real_solver import RealSolver
import numpy as np
from pprint import pprint

from dnn_solver.helpers import Utils

if __name__ == '__main__':
    f_str = '(and (and (x0 < 0) (x1 > 1)) (and (a00 = 1x0 - 1x1) (a01 = 1x0 + 1x1)))'
    f_str = '(or (not (and (x0 > 0) (x1 > 0))) (and (a = 1x0 + 1x1) (a < 0)))'
    f_str = '(and (and (x >= 2) (a = 2x)) (not (a >= 4.0)))'
    f_str = '(and (and (and (and (and (and (and (and (x0 < 0) (x1 > 1)) (n00 = 0)) (and (n01 >= 0) (n01 = 1x0 + 1x1))) (n10 = 0)) (and (n11 >= 0) (n11 = -0.5n00 + 0.1n01))) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (not (not (y0 > y1))))'

    conditions = {
        'in': '(and (x >= 2) (a = 2.1*x))',
        'out': '(a >= 4.2)'
    }

    # parsed_input = Parser.parse(f_str)
    # pprint(parsed_input.row_dict)
    # pprint(parsed_input.col_dict)
    # print('col_dict', parsed_input.col_dict)
    # print('vars_dict', parsed_input.vars_dict)
    # print('reversed_vars_dict', parsed_input.reversed_vars_dict)

    # print(parsed_input.A)
    # print(parsed_input.b)
    # exit()

    # solver = Simplex(parsed_input, rows=[1, 2])
    # feasible = solver.solve()
    # print(feasible)
    # if feasible:
    #     print(solver.get_assignment())
    
    f_str = '(and (and (and (and (and (and (and (and (and (x0 < 0) (x1 > 1)) (0.3428747653*x0 + 0.1191892027*x1 + -0.3442705273*x2 > 0)) (0.4022843837*x0 + 0.3385578393*x1 + -0.6574112176*x2 > 0)) (-0.0505836606*x0 + 0.2380395531*x1 + 0.3830572366*x2 > 0)) (0.5419781208*x0 + 0.2346444129*x1 + -0.1017882227*x2 > 0)) (0.3075127601*x0 + 0.2864566445*x1 + 0.2507228851*x2 > 0)) (0.1208061575*x0 + 0.1779784560*x1 + 0.2350426912*x2 > 0)) (0.0671699643*x0 + -0.2187577188*x1 + -0.5578975677*x2 > 0)) (not (-0.2631770948*x0 + -0.3188667095*x1 + -0.2357685142*x2 <= 0)))'
    solver = RealSolver(f_str)
    print(solver.solve())
    # print(solver.get_assignment())