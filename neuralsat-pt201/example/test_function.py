
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
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    
    # net_path = 'example/onnx/mnist-net_256x2.onnx'
    # vnnlib_path = 'example/vnnlib/prop_1_0.05.vnnlib'
    
    device = 'cpu'
    logger.setLevel(1)
    # Settings.setup(args=None)
    Settings.setup_test()
    
    print(Settings)
    
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
    
    # stable, unstable, lbs, ubs = verifier.compute_stability(objectives)
    # print('stable:', stable)
    # print('unstable:', unstable)
    
    preconditions = [
        # {'/input': (torch.tensor([0]), torch.tensor([-1.]), torch.tensor([0.])), '/input.3': ([], [], [])},
        {'/input': (torch.tensor([0, 1]), torch.tensor([1., 1.]), torch.tensor([0., 0.])), '/input.3': (torch.tensor([0]), torch.tensor([1.]), torch.tensor([0.]))},
        {'/input': (torch.tensor([0, 1]), torch.tensor([ 1., -1.]), torch.tensor([0., 0.])), '/input.3': (torch.tensor([0, 1]), torch.tensor([-1.,  1.]), torch.tensor([0., 0.]))},
    ]
    # preconditions = []
    
    print(preconditions)
    
    verifier.verify(objectives, preconditions=preconditions)
    print('status:', verifier.status)
    print('unsat core:', verifier.get_unsat_core())
    
    # for c in verifier._get_learned_conflict_clauses():
    #     print(c)
    # print('lbs:', lbs)
    # print('ubs:', ubs)
    
    print(verifier.get_stats())


def test_2():
    net_path = 'example/mnist-net_256x2.onnx'
    vnnlib_path = Path('example/prop_1_0.03.vnnlib')
    device = 'cuda'

    print('\n\nRunning test with', net_path, vnnlib_path)
    
    # preconditions = [eval(l.replace('tensor', 'torch.tensor')) for l in open('log.txt').read().strip().split('\n')][:400]
    # print(preconditions)
    preconditions = []
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    Settings.setup(args=None)
    print(Settings)
    logger.setLevel(1)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=200,
        device=device,
    )
    
    
    status = verifier.verify(objectives, preconditions=preconditions)
    print('status:', status, verifier.status)
    print('unsat core')
    from pprint import pprint
    print(verifier.get_unsat_core())
    
    # for c in verifier._get_learned_conflict_clauses():
    #     print(c)

if __name__ == "__main__":
    test_1()