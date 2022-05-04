from dnn_solver.spec import Specification, get_acasxu_bounds
from dnn_solver.dnn_solver import DNNSolver
from example import random_nnet
import time

if __name__ == '__main__':
    name = 'example/paper.nnet'
    name = 'example/random.nnet'

    # model = random_nnet.Model([2, 2, 2, 1])
    # model.save(name)

    p = 0
    print('\nRunning:', name)
    print('Property:', p)

    bounds = get_acasxu_bounds(p)
    spec = Specification(p=p, bounds=bounds)
    
    tic = time.time()
    
    solver = DNNSolver(name, spec)
    status = solver.solve()
    print(name, status, time.time() - tic)

    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('lower:', solver.dnn_theorem_prover.lbs_init.data)
        print('upper:', solver.dnn_theorem_prover.ubs_init.data)
        print('solution:', solution.numpy().tolist())
        print('output:', output.numpy().tolist())
