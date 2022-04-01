import argparse
import time

from dnn_solver.dnn_solver_gurobi import DNNSolver


def test_property_1():
    print(' ----------- Testing property 1 ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            solver = DNNSolver(name, p=1)
            tic_2 = time.time()
            status = solver.solve()
            count += 1
            print(count, i*9+j+1, name, status, time.time() - tic_2)

            if status:
                solution = solver.get_solution()
                output = solver.dnn(solution)
                print('\t- solution:', solution)
                print('\t- output:', output)

    print(time.time() - tic)


def test_property_2():
    print(' ----------- Testing property 2 ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            if i < 1:
                continue
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            solver = DNNSolver(name, p=2)
            tic_2 = time.time()
            status = solver.solve()
            count += 1
            print(count, i*9+j+1, name, status, time.time() - tic_2)

            if status:
                solution = solver.get_solution()
                output = solver.dnn(solution)
                print('\t- solution:', solution)
                print('\t- output:', output)
    print(time.time() - tic)



def test_property_3():
    print(' ----------- Testing property 3 ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            if '1_7' in name or '1_8' in name or '1_9' in name:
                continue

            solver = DNNSolver(name, p=3)
            tic_2 = time.time()
            status = solver.solve()
            count += 1
            print(count, i*9+j+1, name, status, time.time() - tic_2)

            if status:
                solution = solver.get_solution()
                output = solver.dnn(solution)
                print('\t- solution:', solution)
                print('\t- output:', output)
    print(time.time() - tic)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        type=int, 
                        required=True, 
                        choices=[1, 2, 3],
                        help='Property to verify.')

    args = parser.parse_args()

    if args.p == 1:
        test_property_1()
    elif args.p == 2:
        test_property_2()
    elif args.p == 3:
        test_property_3()

