from dnn_solver.utils import InputParser, model_random, model_pa4
from dnn_solver.dnn_solver_gurobi import DNNSolver
# from dnn_solver.dnn_solver import DNNSolver
from utils.read_nnet import Network
from tensorflow import keras
from pprint import pprint
import time



# model = model_random(2, [3, 4, 5], 3)
# model = model_pa4()
# model.save('example/model_4x2_debug.keras')
# model = keras.models.load_model('example/model_4x2_debug.keras')
# model = Network('example/corina.nnet')
# model = Network('example/random.nnet')

tic = time.time()

for i in range(5):
    for j in range(9):
        name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
        # if '1_1' in name or '1_2' in name or '1_3' in name or '1_4' in name:
        #     continue

        model = Network(name)
        vars_mapping, layers_mapping = InputParser.parse(model)
        tic_2 = time.time()
        solver = DNNSolver(model, vars_mapping, layers_mapping)
        print(i*9+j+1, name, solver.solve(), time.time() - tic_2)
        # print({k: v['value'] for k, v in solver._solver._assignment.items()})
print(time.time() - tic)