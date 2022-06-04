import argparse
import time

from dnn_solver.dnn_solver import DNNSolver
from dnn_solver.spec import SpecificationVNNLIB
from utils.read_vnnlib import read_vnnlib_simple
from utils.dnn_parser import DNNParser
from utils.timer import Timers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='NNET/ONNX file path.')
    parser.add_argument('--spec', type=str, required=True, help='VNNLIB file path.')
    parser.add_argument('--solution', action='store_true', help='Get counter-example.')
    parser.add_argument('--timer', action='store_true', help='Timer report.')
    args = parser.parse_args()

    net = DNNParser.parse(args.net)
    spec_list = read_vnnlib_simple(args.spec, net.n_input, net.n_output)

    if args.timer:
        Timers.reset()
        Timers.tic('dnn_solver')
        
    tic = time.time()
    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        solver = DNNSolver(net, spec)
        status = solver.solve()
        if status:
            break

    print(args.net, args.spec, status, time.time() - tic)
    if status and args.solution:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('\t- lower:', spec.get_input_property()['lbs'])
        print('\t- upper:', spec.get_input_property()['ubs'])
        print('\t- solution:', solution.numpy().tolist())
        print('\t- output:', output.numpy().tolist())

    if args.timer:
        Timers.toc('dnn_solver')
        Timers.print_stats()