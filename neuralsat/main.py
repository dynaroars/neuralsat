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
    parser.add_argument('--net', type=str, required=True, help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True, help="path to VNNLIB specification file.")
    parser.add_argument('--solution', action='store_true', help='get a solution (counterexample) if verifier returns SAT.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='select device to run verifier, cpu or cuda (GPU).')
    parser.add_argument('--timeout', type=int, default=1000, help='timeout (in second) for verifying one instance.')
    parser.add_argument('--batch', type=int, help='the maximum number of parallel splits in bound abstraction.')
    parser.add_argument('--summary', type=str, help='path to result file.')
    args = parser.parse_args()
    
    # global configifurations
    arguments.Config['device'] = args.device
    if args.batch:
        arguments.Config['batch'] = args.batch
        
    net = NetworkParser.parse(args.net, args.device)
    specs = read_vnnlib(args.spec)
    print(net)
    # print(specs)

    start_time = time.perf_counter()

    solver = NeuralSAT(net, specs)
    stat = solver.solve(timeout=args.timeout)

    runtime = time.perf_counter() - start_time
    print(f'{stat},{runtime:.03f}')

    if stat == arguments.ReturnStatus.SAT and args.solution:
        print('adv (first 5):', solver.get_assignment().flatten()[:5])

    if args.summary:
        with open(args.summary, 'w') as fp:
            print(f'{stat},{runtime:.03f}', file=fp)