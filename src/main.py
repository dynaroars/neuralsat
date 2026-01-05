import argparse
import warnings
import torch
import time
import os

from helper.network.read_onnx import parse_onnx
from helper.network.read_pth import parse_pth
from helper.spec.objective import parse_vnnlib


from helper.misc.logger import logger, LOGGER_LEVEL
from helper.misc.export import get_adv_string
from helper.misc.result import ReturnStatus

from verifier.verifier import Verifier 

from setting import Settings

 
if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--input_shape', type=int, nargs='+', default=None,
                        help="Input shape of network, e.g., --input_shape 1 3 32 32")
    parser.add_argument('--output_shape', type=int, nargs='+', default=None,
                        help="Output shape of network, e.g., --output_shape 1 10")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda',
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--export_cex', action='store_true',
                        help="enable exporting counter-example to result file.")
    parser.add_argument('--disable_attack', action='store_false',
                        help="disable attack.")
    parser.add_argument('--disable_restart', action='store_false',
                        help="disable RESTART heuristic.")
    parser.add_argument('--disable_stabilize', action='store_false',
                        help="disable STABILIZE heuristic.")
    parser.add_argument('--force_split', type=str, choices=['input', 'hidden'],
                        help="select SPLITTING strategy.")
    parser.add_argument('--reasoning_output', type=str, required=False,
                        help="file to save reasoning steps.")
    parser.add_argument('--setting_file', type=str, required=False,
                        help="file to load specific settings.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")
    parser.add_argument('--export_runtime', action='store_true', required=False,
                        help="output runtime.")
    
    args = parser.parse_args()   
    Settings.setup(args)
    print(Settings)
    
    if os.environ.get('NEURALSAT_SYNTHETIC_BUG_DROP_PROBABILITY'):
        assert Settings.use_save_reasoning_step, 'Reasoning step is required for synthetic bug'
        
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # network
    if args.net.endswith('.onnx'):
        model, input_shape, output_shape = parse_onnx(args.net, args.input_shape, args.output_shape)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape = parse_pth(args.net, args.input_shape, args.output_shape)
    else:
        raise NotImplementedError('Unsupported network type')
    
    logger.debug(f'net path: {args.net}')
    logger.debug(f'spec path: {args.spec}')
    logger.debug(f'timeout: {args.timeout}')
    logger.debug(f'device: {args.device}')
    
    model.to(args.device)

    if args.verbosity:
        print(model)
    logger.info(f'[!] Input shape: {input_shape}')
    logger.info(f'[!] Output shape: {output_shape}')
    
    # specification
    objectives = parse_vnnlib(args.spec, input_shape)
    
    # verifier
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    
    # verify
    timeout = args.timeout - (time.time() - START_TIME)
    status = verifier.verify(objectives, timeout=timeout, force_split=args.force_split)
    runtime = time.time() - START_TIME
    
    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    if verifier.adv is not None:
        logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
        logger.debug(f'output: {verifier.net(verifier.adv).flatten().detach().cpu()}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            if args.export_runtime:
                print(f'{status},{runtime:.04f}', file=fp)
            else:
                print(status, file=fp)
            if (verifier.adv is not None) and args.export_cex:
                print(get_adv_string(inputs=verifier.adv, net_path=args.net), file=fp)

    if args.reasoning_output and Settings.use_save_reasoning_step and status == ReturnStatus.UNSAT:
        if hasattr(verifier, 'domains_list') and not isinstance(verifier.domains_list, list):
            verifier.domains_list.reasoning_domains.export_aptp(args.reasoning_output)
        else:
            print(f'[!] Does not have any reasoning step')

    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')
    
    print(f'{status},{runtime:.04f}')

    if os.environ.get('NEURALSAT_SYNTHETIC_BUG_DROP_PROBABILITY'):
        print('[!] Synthetic bug is enabled for demonstatration purpose. Do not enable for benchmarking.')
        