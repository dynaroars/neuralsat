import argparse
import time
from pathlib import Path


from util.network.network_parser import NetworkParser
from util.spec.read_vnnlib import read_vnnlib
from util.misc.args import update_arguments
from util.misc.logger import logger
from neuralsat import NeuralSAT
import arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--solution', action='store_true',
                        help='get a solution (counterexample) if verifier returns SAT.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='select device to run verifier, cpu or cuda (GPU).')
    parser.add_argument('--timeout', type=int, default=1000,
                        help='timeout (in second) for verifying one instance.')
    parser.add_argument('--batch', type=int, default=1000,
                        help='the maximum number of parallel splits in bound abstraction.')
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--summary', type=str,
                        help='path to result file.')
    parser.add_argument('--attack', action='store_true',
                        help='enable adversarial attacks.')
    parser.add_argument('--refine', action='store_true',
                        help='enable pre-verifying bound refinement (not support ResNet yet!).')
    args = parser.parse_args()   
    
    # update global configifurations
    update_arguments(args)

    # load network
    net = NetworkParser.parse(args.net, args.device)
    print(net.layers)

    # load spec
    specs = read_vnnlib(Path(args.spec))
    # print(specs)
    start_time = time.perf_counter()
    
    solver = NeuralSAT(net, specs)
    stat = solver.solve(timeout=args.timeout)

    runtime = time.perf_counter() - start_time
    print(f'{stat},{runtime:.03f}')

    if stat == arguments.ReturnStatus.SAT and args.solution:
        print('adv (first 5):', solver.get_assignment().flatten().detach().cpu()[:5])

    if args.summary:
        with open(args.summary, 'w') as fp:
            print(f'{stat},{runtime:.03f}', file=fp)
