import argparse
import time

from dnn_solver.dnn_solver import DNNSolver
from test_torch import *

def test_property_1(args):
    print(f' ----------- Testing property {args.p} ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            test_multiprocess(name, args.p)
    print(time.time() - tic)


def test_property_2(args):
    print(f' ----------- Testing property {args.p} ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            if i < 1:
                continue
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            solver = DNNSolver(name, p=args.p)
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



def test_property_3(args):
    print(f' ----------- Testing property {args.p} ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            if '1_7' in name or '1_8' in name or '1_9' in name:
                continue

            solver = DNNSolver(name, p=args.p)
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


def test_property_4(args):
    print(f' ----------- Testing property {args.p} ----------- ')
    count = 0
    tic = time.time()
    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
            if '1_7' in name or '1_8' in name or '1_9' in name:
                continue

            solver = DNNSolver(name, p=args.p)
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


def test_property(args):
    print(f' ----------- Testing property {args.p} ----------- ')
    count = 0
    tic = time.time()
    bounds = get_acasxu_bounds(args.p)

    for i in range(5):
        for j in range(9):
            name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'

            solver = DNNSolver(name, p=args.p)
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
                        help='Property to verify.')

    args = parser.parse_args()

    if args.p == 1:
        test_property_1(args)
    elif args.p == 2:
        test_property_2(args)
    elif args.p == 3:
        test_property_3(args)
    elif args.p == 4:
        test_property_4(args)
    elif args.p == 5:
        test_property_5(args)
    elif args.p == 6:
        raise NotImplementedError
    elif args.p in [7, 8, 9, 10]:
        test_property_7(args)

