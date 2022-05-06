import argparse
import time

from dnn_solver.dnn_solver import DNNSolver
from test_torch import *

# def test_property_1(args):
#     print(f' ----------- Testing property {args.p} ----------- ')
#     count = 0
#     tic = time.time()
#     for i in range(5):
#         for j in range(9):
#             name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
#             test_multiprocess(name, args.p)
#     print(time.time() - tic)


# def test_property_2(args):
#     print(f' ----------- Testing property {args.p} ----------- ')
#     count = 0
#     tic = time.time()
#     for i in range(5):
#         for j in range(9):
#             if i < 1:
#                 continue
#             name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
#             solver = DNNSolver(name, p=args.p)
#             tic_2 = time.time()
#             status = solver.solve()
#             count += 1
#             print(count, i*9+j+1, name, status, time.time() - tic_2)

#             if status:
#                 solution = solver.get_solution()
#                 output = solver.dnn(solution)
#                 print('\t- solution:', solution)
#                 print('\t- output:', output)
#     print(time.time() - tic)



# def test_property_3(args):
#     print(f' ----------- Testing property {args.p} ----------- ')
#     count = 0
#     tic = time.time()
#     for i in range(5):
#         for j in range(9):
#             name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
#             if '1_7' in name or '1_8' in name or '1_9' in name:
#                 continue

#             solver = DNNSolver(name, p=args.p)
#             tic_2 = time.time()
#             status = solver.solve()
#             count += 1
#             print(count, i*9+j+1, name, status, time.time() - tic_2)

#             if status:
#                 solution = solver.get_solution()
#                 output = solver.dnn(solution)
#                 print('\t- solution:', solution)
#                 print('\t- output:', output)
#     print(time.time() - tic)


# def test_property_4(args):
#     print(f' ----------- Testing property {args.p} ----------- ')
#     count = 0
#     tic = time.time()
#     for i in range(5):
#         for j in range(9):
#             name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'
#             if '1_7' in name or '1_8' in name or '1_9' in name:
#                 continue

#             solver = DNNSolver(name, p=args.p)
#             tic_2 = time.time()
#             status = solver.solve()
#             count += 1
#             print(count, i*9+j+1, name, status, time.time() - tic_2)

#             if status:
#                 solution = solver.get_solution()
#                 output = solver.dnn(solution)
#                 print('\t- solution:', solution)
#                 print('\t- output:', output)
#     print(time.time() - tic)


# def test_property(args):
#     print(f' ----------- Testing property {args.p} ----------- ')
#     count = 0
#     tic = time.time()
#     bounds = get_acasxu_bounds(args.p)

#     for i in range(5):
#         for j in range(9):
#             name = f'benchmark/acasxu/nnet/ACASXU_run2a_{i+1}_{j+1}_batch_2000.nnet'

#             solver = DNNSolver(name, p=args.p)
#             tic_2 = time.time()
#             status = solver.solve()
#             count += 1
#             print(count, i*9+j+1, name, status, time.time() - tic_2)

#             if status:
#                 solution = solver.get_solution()
#                 output = solver.dnn(solution)
#                 print('\t- solution:', solution)
#                 print('\t- output:', output)
#     print(time.time() - tic)


from utils.read_vnnlib import read_vnnlib_simple
from dnn_solver.spec import SpecificationVNNLIB
from utils.read_nnet import NetworkTorch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='NNET file path.')
    parser.add_argument('--spec', type=str, required=True, help='VNNLIB file path.')

    args = parser.parse_args()

    net = NetworkTorch(args.net)
    spec_list = read_vnnlib_simple(args.spec, net.input_shape[1], net.output_shape[1])

    tic = time.time()
    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        solver = DNNSolver(net, spec)
        status = solver.solve()
        if status:
            break

    print(args.net, args.spec, status, time.time() - tic)
    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('\t- solution:', solution)
        print('\t- output:', output.numpy().tolist())