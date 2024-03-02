import argparse
import logging
import proton # type: ignore
import torch
import time
import os

from verifier.objective import Objective, DnfObjectives
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.verifier import Verifier 
from util.misc.logger import logger
from setting import Settings



def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives


@proton.profile(name='cac', context='python')
def run(verifier, objectives):
    START_TIME = time.time()
    with proton.scope("verify"):
        status = verifier.verify(objectives)
    runtime = time.time() - START_TIME

    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')


if __name__ == '__main__':
     # argument
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
    args = parser.parse_args()   
    
    Settings.setup(args)
    logger.setLevel(logging.INFO)
    
    model, input_shape, objectives = extract_instance(args.net, args.spec)
    model.to(args.device)
    print(model)
    print(Settings)
    
    # verifier
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    run(verifier, objectives)
    logger.info('Finalizing Proton ...')
    proton.finalize()
    
    # func = proton.profile(run(verifier, objectives), name="dynamic_net", context=args.context)
    
    