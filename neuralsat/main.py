from pathlib import Path
import argparse
import time
import os

from verifier.objective import Objective, DnfObjectives
from util.misc.logger import logger, LOGGER_LEVEL
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from util.misc.export import get_adv_string
from verifier.verifier import Verifier 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    args = parser.parse_args()   
    
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # parse instance
    vnnlibs = read_vnnlib(Path(args.spec))
    model, input_shape, output_shape = parse_onnx(args.net)
    model.to(args.device)
    if args.verbosity:
        print(model)
        # pass
        # exit()
        
    logger.info(f'[!] Input shape: {input_shape}')
    logger.info(f'[!] Output shape: {output_shape}')
    
    
    START_TIME = time.time()
    # verifier
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives)
    
    # verify
    timeout = args.timeout - (time.time() - START_TIME)
    status = verifier.verify(objectives, timeout=timeout)
    
    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    print(f'{status},{time.time()-START_TIME:.04f}')
    if verifier.adv is not None:
        logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            print(status, file=fp)
            if verifier.adv is not None:
                print(get_adv_string(inputs=verifier.adv, outputs=verifier.net(verifier.adv)), file=fp)