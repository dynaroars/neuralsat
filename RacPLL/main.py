import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import time
import os

from heuristic.falsification import gradient_falsification
from utils.read_vnnlib import read_vnnlib_simple
from dnn_solver.spec import SpecificationVNNLIB
from dnn_solver.dnn_solver import DNNSolver
from utils.dnn_parser import DNNParser
from utils.timer import Timers
from abstract.crown import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--spec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acasxu', 'mnist', 'cifar', 'test'])
    parser.add_argument('--solution', action='store_true')
    parser.add_argument('--attack', action='store_true')
    parser.add_argument('--timer', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--file', type=str, default='res.txt')
    args = parser.parse_args()

    device = torch.device(args.device)

    tic = time.time()
    net = DNNParser.parse(args.net, args.dataset, args.device)
    spec_list = read_vnnlib_simple(args.spec, net.n_input, net.n_output)

    if args.timer:
        Timers.reset()
        Timers.tic('dnn_solver')
        
    arguments.Config["general"]["batch"] = args.batch
    new_spec_list = []
    attacked = False
    status = 'UNKNOWN'
    if args.dataset != 'acasxu':
        if args.attack:
            Timers.tic('PGD attack')
            pgd = gradient_falsification.GradientFalsification(net, SpecificationVNNLIB(spec_list[0]))
            attacked, adv = pgd.evaluate()
            if attacked:
                status = 'SAT'
                print(args.net, args.spec, status, time.time() - tic)
            Timers.toc('PGD attack')

        bounds = spec_list[0][0]
        for i in spec_list[0][1]:
            new_spec_list.append((bounds, [i]))
        spec_list = new_spec_list

    if not attacked:
        for i, s in enumerate(spec_list):
            spec = SpecificationVNNLIB(s)
            solver = DNNSolver(net, spec, args.dataset)
            status = solver.solve(timeout=args.timeout)
            if status in ['SAT', 'TIMEOUT']:
                break

        print(args.net, args.spec, status, time.time() - tic)
        if status=='SAT' and args.solution:
            solution = solver.get_solution()
            output = solver.net(solution)
            print('\t- lower:', spec.get_input_property()['lbs'])
            print('\t- upper:', spec.get_input_property()['ubs'])
            print('\t- solution:', solution.detach().numpy().tolist())
            print('\t- output:', output.detach().numpy().tolist())

    if args.timer:
        Timers.toc('dnn_solver')
        Timers.print_stats()

    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    with open(os.path.join(f'results/{args.dataset}', args.file), 'w') as fp:
        print(f'{status},{time.time()-tic:.02f}', file=fp)