from pathlib import Path
import argparse
import time

from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx

from verifier.objective import Objective, DnfObjectives
from verifier.verifier import Verifier 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--batch', type=int, default=1000,
                        help="number of branches verified each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    
    args = parser.parse_args()   
    
    START_TIME = time.time()
    vnnlibs = read_vnnlib(Path(args.spec))
    model, input_shape, output_shape = parse_onnx(args.net)
    model.to(args.device)
    print(model)
    
    # print(vnnlib[0][1])
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    objectives = DnfObjectives(objectives)
   
    timeout = args.timeout - (time.time() - START_TIME)
    status = verifier.verify(objectives, timeout=timeout)
    print('\n[!] Iterations:', verifier.iteration)
    
    print(f'\n{status},{time.time()-START_TIME:.04f}')
    if verifier.adv is not None:
        print('adv (first 5):', verifier.adv.flatten()[:5].detach().cpu())