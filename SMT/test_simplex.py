from simplex.parser import Parser
from linear_solver.linear_solver import LinearSolver
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
    f_str = '(and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (x0 < 0) (x1 > 1)) (-0.5254483222*x0 + -0.0247446894*x1 + 0.6538374423*x2 > 0)) (-0.2076188027*x0 + 0.6304305791*x1 + 0.2272567749*x2 > 0)) (0.1363617777*x0 + 0.1437975168*x1 + 0.2977073788*x2 > 0)) (-0.5705137252*x0 + -0.2265316545*x1 + 0.5072546005*x2 > 0)) (-0.1618499159*x0 + 0.0783964991*x1 + -0.1737857460*x2 > 0)) (-0.4461473524*x0 + 0.5485363006*x1 + -0.1291967034*x2 > 0)) (-0.4291794896*x0 + 0.6391158103*x1 + -0.0208246111*x2 > 0)) (-0.2847822010*x0 + -0.0138310790*x1 + -0.1750609874*x2 > 0)) (-0.2116975971*x0 + 0.3503022095*x1 + 0.0334987041*x2 > 0)) (-0.2472857214*x0 + 0.0910085493*x1 + 0.5772947333*x2 > 0)) (0.0327355273*x0 + -0.2696579051*x1 + 0.4167513060*x2 > 0)) (-0.4387794635*x0 + -0.1791660548*x1 + 0.5035365020*x2 > 0)) (-0.1244718770*x0 + 0.2913324430*x1 + -0.0519786044*x2 > 0)) (-0.1068522427*x0 + 0.0962806748*x1 + 0.0640542749*x2 > 0)) (-0.1844041635*x0 + -0.1802565269*x1 + 0.5119393831*x2 > 0)) (-0.1550510270*x0 + -0.2138309656*x1 + 0.2236257548*x2 > 0)) (0.0049923433*x0 + 0.2303416156*x1 + 0.1987598444*x2 > 0)) (-0.3359068019*x0 + 0.1368511350*x1 + 0.2967057436*x2 > 0)) (0.0160504730*x0 + 0.2182754062*x1 + -0.0416319720*x2 > 0)) (-0.2651961889*x0 + 0.1439227036*x1 + 0.3690338053*x2 > 0)) (-0.0708347807*x0 + -0.0851026752*x1 + 0.1174278477*x2 > 0)) (-0.1032977019*x0 + 0.0593960886*x1 + 0.2768955596*x2 > 0)) (-0.0619864409*x0 + -0.0982409333*x1 + 0.2528981789*x2 > 0)) (0.0202174192*x0 + -0.1856294923*x1 + 0.0730808819*x2 > 0)) (-0.0963643226*x0 + -0.1674137551*x1 + 0.0932552049*x2 > 0)) (-0.1651497119*x0 + 0.0317936332*x1 + 0.2186574578*x2 > 0)) (-0.0730159735*x0 + 0.1461228152*x1 + 0.1455443376*x2 > 0)) (-0.1965492290*x0 + 0.0303470935*x1 + 0.2960153319*x2 > 0)) (-0.0753867599*x0 + 0.0615153767*x1 + 0.1217910941*x2 > -0.0398708760*x0 + -0.0656358180*x1 + 0.0351406983*x2))'

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
    
    solver = LinearSolver(f_str)
    print(solver.solve()[0] != False)
    # print(solver.get_assignment())