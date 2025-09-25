import traceback
import argparse
import torch
import time
import copy
import os

from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.network.read_onnx import _parse_onnx as parse_onnx
from helper.misc.logger import logger, LOGGER_LEVEL
from helper.spec.objective import parse_vnnlib
from helper.network.read_pth import parse_pth
from helper.misc.result import ReturnStatus

from verifier.dec_verifier import DecompositionalVerifier
from attacker.attacker import Attacker
from setting import Settings



def main():
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--batch', type=int, default=500,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--category', type=str, required=True,
                        help="running category.")

    args = parser.parse_args()   
    
    
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    Settings.setup_decompose(args)
    
    print(Settings)
        
    print('\n****** Running with NeuralSAT verifier ******\n\n')
        
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # network
    if args.net.endswith('.onnx'):
        pth_path = args.net[:-5] + '.pth'
        if os.path.exists(pth_path):
            model, input_shape, output_shape = parse_pth(pth_path)
        else:
            model, input_shape, output_shape = parse_onnx(args.net)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape = parse_pth(args.net)
    else:
        raise NotImplementedError('Unsupported network type')
    
    model.eval()
    model.to(args.device)
    
    if args.verbosity:
        print(model)
    
    logger.info(f'[!] Input shape: {input_shape}')
    logger.info(f'[!] Output shape: {output_shape}')
    
    # specification
    dnf_objectives = parse_vnnlib(args.spec, input_shape)

    # attacker
    if Settings.use_attack:
        attacker = Attacker(
            net=model, 
            objective=dnf_objectives, 
            input_shape=input_shape, 
            device=args.device,
        )
        
        _, adv = attacker.run(timeout=10.0)
        if adv is not None:
            runtime = time.time() - START_TIME
            # export
            if args.result_file:
                os.remove(args.result_file) if os.path.exists(args.result_file) else None
                with open(args.result_file, 'w') as fp:
                    print(f'sat,{runtime:.06f}', file=fp)
            print(f'sat,{runtime:.04f}')
            return
        
        print('[!] Attacks failed')
    
    
    # verifier
    verifier = DecompositionalVerifier(
        net=model,
        input_shape=input_shape,
        device=args.device,
    )    
    
    timeout = args.timeout - (time.time() - START_TIME)
    status = ReturnStatus.UNKNOWN
    
    run_decompose = True
    try:
        print('[+] Original Verification')
        status = verifier.original_verify(
            objectives=copy.deepcopy(dnf_objectives), 
            timeout=timeout, 
            batch=args.batch,
        )
    except RuntimeError as exception:
        if is_cuda_out_of_memory(exception):
            gc_cuda()
            print('[+] Decomposition Verification')
            Settings.use_decompose = 1 
            status = verifier.decompositional_verify(
                objectives=copy.deepcopy(dnf_objectives), 
                timeout=timeout, 
                batch=args.batch,
                interm_batch=Settings.sequential_batch,
            )
            run_decompose = False
        else:
            print(traceback.format_exc())
            status = ReturnStatus.UNKNOWN
    except:
        print(traceback.format_exc())
        status = ReturnStatus.UNKNOWN
        
    if (status == ReturnStatus.UNKNOWN) and Settings.use_decompose and run_decompose:
        status = verifier.decompositional_verify(
            objectives=copy.deepcopy(dnf_objectives), 
            timeout=timeout, 
            batch=args.batch,
            interm_batch=Settings.sequential_batch,
        )
    
    runtime = time.time() - START_TIME
    
    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            print(f'{status},{runtime:.06f}', file=fp)

    print(f'{status},{runtime:.04f}')
        
if __name__ == '__main__':
    main()