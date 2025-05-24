from pathlib import Path
import numpy as np
import time
import os

import warnings
warnings.filterwarnings(action='ignore')

from helper.spec.objective import Objective, DnfObjectives
from helper.spec.read_vnnlib import read_vnnlib
from helper.network.read_onnx import parse_onnx
from verifier.verifier import Verifier 
from helper.misc.result import ReturnStatus
from helper.misc.logger import logger
from auto_LiRPA.abstractor.bound_ops import BoundRelu
from test import extract_instance


def refine_layer(node):
    from auto_LiRPA.abstractor.bound_ops import BoundLinear
    
    for n in node.inputs:
        refine_layer(n)
        
    if isinstance(node, BoundLinear):
        print('[+] Refine layer:', node)
        # print('\t- lower:', node.lower)
        # print('\t- upper:', node.upper)
        candidates = []
        candidate_neuron_ids = []
        
        for neuron_idx, v in enumerate(node.solver_vars):
            # print(v.VarName, v.lb==node.lower[0, neuron_idx], v.ub==node.upper[0, neuron_idx] if node.upper is not None else None)
            candidates.append(v.VarName)
            candidate_neuron_ids.append(neuron_idx)
            v.lb = -np.inf
            v.ub = np.inf
        
        # exit()
    

if __name__ == "__main__":
    
    net_path = 'example/test_mnistfc.onnx'
    # net_path = 'example/mnistfc-medium-net-151.onnx'
    vnnlib_path = Path('example/prop_2_0.03.vnnlib')
    
    net_path = '../benchmark/mnistfc_hard/onnx/mnistfc-hard-net-177.onnx'
    vnnlib_path = Path('../benchmark/mnistfc_hard/spec/prop_8_0.05.vnnlib')
    
    print('\nRunning test with', net_path, vnnlib_path)
    device = 'cpu'
    batch = 1
    logger.setLevel(1)
    
    preconditions = [eval(line) for line in open('clause.txt').read().strip().split('\n') if not line.startswith('#')]
    # preconditions = []
    print(preconditions)
    

    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    print(model)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=batch,
        device=device,
    )

    
    print(verifier.verify(objectives, preconditions=preconditions))
    exit()
    