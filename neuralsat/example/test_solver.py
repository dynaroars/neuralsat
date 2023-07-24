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
from auto_LiRPA.bound_ops import BoundRelu

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
    
    net_path = 'example/test_mnistfc.onnx'
    # net_path = 'example/mnistfc-medium-net-151.onnx'
    vnnlib_path = Path('example/prop_2_0.03.vnnlib')
    
    device = 'cpu'
    logger.setLevel(1)
    
    print('\nRunning test with', net_path, vnnlib_path)

    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    print(model)
    # exit()
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )

    
    # print(verifier.verify(objectives))
    # exit()
    
    obj = objectives.pop(1)
    input_lowers = obj.lower_bounds.view(input_shape).to(device)
    input_uppers = obj.upper_bounds.view(input_shape).to(device)
    c = obj.cs.to(device)
    print(input_uppers.shape)
    print(obj.cs.shape)
    
    
    verifier._setup_restart(0, obj)
    verifier.abstractor.initialize(obj, None)
    
    tic = time.time()
    verifier.abstractor.build_lp_solver('mip', input_lowers, input_uppers, c=None)
    print('MIP refine:', time.time() - tic)
    
    intermediate_layer_bounds = verifier.abstractor.net.get_refined_intermediate_bounds()
    
    for k, v in intermediate_layer_bounds.items():
        print(k, [_.shape for _ in v])
    exit()
    
    verifier.abstractor.initialize(obj, None, intermediate_layer_bounds)
    
    # name_dict = {i: layer.inputs[0].name for (i, layer) in enumerate(verifier.abstractor.net.perturbed_optimizable_activations)}
    # pre_relu_indices = [i for (i, layer) in enumerate(verifier.abstractor.net.perturbed_optimizable_activations) if isinstance(layer, BoundRelu)]
    # print(pre_relu_indices)
    # print(name_dict)
    for _ in range(0):
        tic = time.time()
        intermediate_layer_bounds = verifier.abstractor.net.get_refined_intermediate_bounds()
        verifier.abstractor.build_lp_solver('mip', input_lowers, input_uppers, c=None, intermediate_layer_bounds=intermediate_layer_bounds)
        print(_, 'MIP refine:', time.time() - tic)
        
        intermediate_layer_bounds = verifier.abstractor.net.get_refined_intermediate_bounds()
        verifier.abstractor.initialize(obj, None, intermediate_layer_bounds)
        print()
    
    
    # tic = time.time()
    # intermediate_layer_bounds[name_dict[pre_relu_indices[1]]][0][0][83] = -0.02
    # intermediate_layer_bounds[name_dict[pre_relu_indices[2]]][1][0][26] = -0.05
    # intermediate_layer_bounds[name_dict[4]][1][0][105] = 2.05
    # verifier.abstractor.build_lp_solver('mip', input_lowers, input_uppers, c=None, intermediate_layer_bounds=intermediate_layer_bounds)
    # print('MIP refine:', time.time() - tic)
    
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
        