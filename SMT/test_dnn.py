from dnn_solver.dnn_solver_gurobi import DNNSolver
import time


def test_property_1():
    print('Testing property 1')
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            solver = DNNSolver(name, p=1)
            tic_2 = time.time()
            print(i*9+j+1, name, solver.solve(), time.time() - tic_2)
    print(time.time() - tic)


def test_property_2():
    print('Testing property 2')
    tic = time.time()
    for i in range(5):
        for j in range(9):
            if i < 1:
                continue
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            solver = DNNSolver(name, p=2)
            tic_2 = time.time()
            print(i*9+j+1, name, solver.solve(), time.time() - tic_2)
    print(time.time() - tic)




def test_property_3():
    print('Testing property 3')
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            if '1_7' in name or '1_8' in name or '1_9' in name:
                continue

            solver = DNNSolver(name, p=3)
            tic_2 = time.time()
            print(i*9+j+1, name, solver.solve(), time.time() - tic_2)
    print(time.time() - tic)



if __name__ == '__main__':
    test_property_2()