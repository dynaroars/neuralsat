from pathlib import Path
import numpy as np
import time
import os

import warnings
warnings.filterwarnings(action='ignore')

from verifier.objective import Objective, DnfObjectives
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.verifier import Verifier 
from util.misc.result import ReturnStatus
from util.misc.logger import logger


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


def refine_layer(node):
    from auto_LiRPA.bound_ops import BoundLinear
    
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
    
    # net_path = 'example/mnist-net_256x4.onnx'
    net_path = 'example/test_mnistfc.onnx'
    vnnlib_path = Path('example/prop_0_0.05.vnnlib')
    
    device = 'cpu'
    logger.setLevel(1)
    
    print('\nRunning test with', net_path, vnnlib_path)

    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    print(model)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )

    
    obj = objectives.pop(1)
    input_lowers = obj.lower_bounds.view(input_shape).to(device)
    input_uppers = obj.upper_bounds.view(input_shape).to(device)
    c = obj.cs.to(device)
    print(input_uppers.shape)
    print(obj.cs.shape)
    
    
    verifier._setup_restart(0, obj)
    verifier._initialize(obj, [])
    
    tic = time.time()
    verifier.abstractor.build_lp_solver('mip', input_lowers, input_uppers, c=None)
    # print(verifier.abstractor.net.model)
    print('MIP refine:', time.time() - tic)
    
    print(verifier.verify(objectives))
    exit()
    
    # verifier.abstractor.net.clear_solver_module(verifier.abstractor.net.final_node())
    # del verifier.abstractor.net.model
    
    # verifier.abstractor.build_lp_solver('lp', input_lowers, input_uppers, c)
    # print(verifier.abstractor.net.model)
    
    
    # for v in verifier.abstractor.net.model.getVars():
    #     print(v.VarName, v.lb, v.ub)
    
    # refine_layer(verifier.abstractor.net.final_node())
    # verifier.abstractor.net.model.update()
    
    # for v in verifier.abstractor.net.model.getVars():
    #     print(v.VarName, v.lb, v.ub)
        