from dnn_solver.dnn_solver_gurobi import DNNSolver
import time

if __name__ == '__main__':
    
    i = 1
    j = 2
    p = 7

    name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i}_{j}_batch_2000.nnet'
    # name = 'benchmark/acasxu/nnet/ACASXU_run2a_3_3_batch_2000.nnet'
    # name = 'benchmark/acasxu/nnet/ACASXU_run2a_5_3_batch_2000.nnet'
    # name = f'example/random.nnet'
    tic = time.time()
    solver = DNNSolver(name, p)
    status = solver.solve()
    print(name, status , time.time() - tic)

    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('solution:', solution)
        print('output:', output)
