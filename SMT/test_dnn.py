from dnn_solver.utils import InputParser, model_random, model_pa4
from dnn_solver.dnn_solver_gurobi import DNNSolver
# from dnn_solver.dnn_solver import DNNSolver
from tensorflow import keras
from pprint import pprint
import time

conditions = {
    'in': '(and (x0 - x1 > 0) (x0 + x1 <= 0))',
    'out': '(y0 < y1)'
}


model = model_random(10, [200]*9, 10)
# model = model_pa4()
# model.save('example/model_4x2_debug.keras')
# model = keras.models.load_model('example/model_4x2_debug.keras')

vars_mapping, layers_mapping = InputParser.parse(model)
# print(layers_mapping)

solver = DNNSolver(model, vars_mapping, layers_mapping, conditions)
# tic = time.time()
print(solver.solve())
# print({k: v['value'] for k, v in solver._solver._assignment.items()})
# print(time.time() - tic)