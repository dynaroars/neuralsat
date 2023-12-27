
from pathlib import Path
import torch.nn as nn
import random
import torch
import time
import os

from verifier.verifier import Verifier 
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.objective import Objective, DnfObjectives
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


    
def test_1():
    net_path = 'example/backup/motivation_example_161.onnx'
    vnnlib_path = 'example/motivation_example.vnnlib'
    
    net_path = 'example/mnist-net_256x2.onnx'
    vnnlib_path = 'example/prop_1_0.05.vnnlib'
    
    device = 'cuda'
    logger.setLevel(1)
    Settings.setup(args=None)
    
    print('Running test with', net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    print(model)

    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    stable, unstable, lbs, ubs = verifier.compute_stability(objectives)
    print('stable:', stable)
    print('unstable:', unstable)
    
    verifier.verify(objectives)
    print(verifier.get_stats())
    # print('lbs:', lbs)
    # print('ubs:', ubs)


if __name__ == "__main__":
    net_path = 'example/cifar10_2_255_simplified.onnx'
    vnnlib_path = Path('example/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib')
    device = 'cuda'

    print('\n\nRunning test with', net_path, vnnlib_path)
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    Settings.setup(args=None)
    logger.setLevel(1)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=200,
        device=device,
    )
    
    status = verifier.verify(objectives)
    print(status)
