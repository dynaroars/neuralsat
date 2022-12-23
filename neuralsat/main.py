import argparse
import torch
import time
import os

from util.network.network_parser import NetworkParser
from util.spec.read_vnnlib import read_vnnlib
from util.misc.logger import logger
from neuralsat import NeuralSAT
import arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--spec', type=str, required=True)
    parser.add_argument('--solution', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--summary', type=str, default='res.txt')
    args = parser.parse_args()
    args.device = torch.device(args.device)

    net = NetworkParser.parse(args.net, args.device)
    print(net)
    specs = read_vnnlib(args.spec)
    # print(specs)
    
    start_time = time.perf_counter()
    solver = NeuralSAT(net, specs)

    stat = solver.solve(timeout=args.timeout)


    msg = f'{stat:<50} time={time.perf_counter() - start_time:.03f}'
    logger.info(msg)

    if stat == arguments.ReturnStatus.SAT and args.solution:
        print('adv:', solver.get_assignment().flatten()[:5])