from dnn_solver.utils import InputParser, model_random
from dnn_solver.dnn_solver import DNNSolver
from pprint import pprint
import time

conditions = {
    'in': '(and (x0 < 0) (x1 > 1))',
    'out': '(y0 > y1)'
}

# dnn = {
#     'a0_0': [(1.0, 'x0'), (-1.0, 'x1')],
#     'a0_1': [(1.0, 'x0'), (1.0, 'x1')],
#     'a1_0': [(0.5, 'n0_0'), (-0.2, 'n0_1')],
#     'a1_1': [(-0.5, 'n0_0'), (0.1, 'n0_1')],
#     'y0' : [(1.0, 'n1_0'), (-1.0, 'n1_1')],
#     'y1' : [(-1.0, 'n1_0'), (1.0, 'n1_1')],
# }

# vars_mapping = {
#     'a0_0': 1,
#     'a0_1': 2,
#     'a1_0': 3,
#     'a1_1': 4,
# }

model = model_random(3, [10]*2, 4)
dnn, vars_mapping, layers_mapping = InputParser.parse(model)

pprint(dnn)

pprint(vars_mapping)

solver = DNNSolver(dnn, vars_mapping, layers_mapping, conditions)
tic = time.time()
print(solver.solve())
print(time.time() - tic)