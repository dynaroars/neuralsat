import argparse
import torch
import time
import os

from heuristic.falsification import gradient_falsification, randomized_falsification
from utils.read_vnnlib import read_vnnlib_simple
from dnn_solver.spec import SpecificationVNNLIB
from dnn_solver.dnn_solver_multi import DNNSolverMulti
from dnn_solver.dnn_solver import DNNSolver
from utils.dnn_parser import DNNParser
from utils.timer import Timers
from abstract.crown import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--spec', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acasxu', 'mnist', 'cifar', 'test'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--attack', action='store_true')
    parser.add_argument('--timer', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--file', type=str, default='res.txt')
    args = parser.parse_args()

    args.device = torch.device(args.device) if args.dataset in ['mnist', 'cifar'] else torch.device('cpu')

    net = DNNParser.parse(args.net, args.dataset, args.device)
    spec_list = read_vnnlib_simple(args.spec, net.n_input, net.n_output)
    tic = time.time()

    if args.timer:
        Timers.reset()
        Timers.tic('Main')
        
    arguments.Config["general"]["verbose"] = args.verbose

    attacked = False
    status = 'UNKNOWN'

    if args.attack:
        if net.n_input > 50:
            Timers.tic('PGD attack')
            pgd = gradient_falsification.GradientFalsification(net, SpecificationVNNLIB(spec_list[0]))
            attacked, adv = pgd.evaluate()
            if attacked:
                status = 'SAT'
                # print(args.net, args.spec, status, time.time() - tic)
            Timers.toc('PGD attack')
        else:
            Timers.tic('Random attack')
            for i, s in enumerate(spec_list):
                spec = SpecificationVNNLIB(s)
                rf = randomized_falsification.RandomizedFalsification(net, spec)
                stat, adv = rf.eval(timeout=0.5)
                if stat == 'violated':
                    attacked = True
                    status = 'SAT'
                    # print(args.net, args.spec, status, time.time() - tic)
                    break
            Timers.toc('Random attack')

    if net.n_input > 50:
        new_spec_list = []
        bounds = spec_list[0][0]
        for i in spec_list[0][1]:
            new_spec_list.append((bounds, [i]))
        spec_list = new_spec_list

    if not attacked:
        if net.n_input > 50:
            for i, s in enumerate(spec_list):
                spec = SpecificationVNNLIB(s)
                solver = DNNSolver(net, spec, args.dataset)
                try:
                    status = solver.solve(timeout=args.timeout)
                    if status in ['SAT', 'TIMEOUT']:
                        break
                except KeyboardInterrupt:
                    exit()
                except:
                    raise
                    status = 'UNKNOWN'
                    break
        else:
            arguments.Config["general"]["n_procs"] = 16
            for i, s in enumerate(spec_list):
                spec = SpecificationVNNLIB(s)
                solver = DNNSolverMulti(net, spec, args.dataset)
                status = solver.solve()
                if status in ['SAT', 'UNKNOWN']:
                    break


    print(f'\n\n[{args.dataset}]', args.net, args.spec, status, f'{time.time()-tic:.02f}')
    print(f'\tExport to: results/{args.dataset}/{args.file}\n')
    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    with open(f'results/{args.dataset}/{args.file}', 'w') as fp:
        print(f'{status},{time.time()-tic:.02f}', file=fp)

    if args.timer:
        Timers.toc('Main')
        Timers.print_stats()
