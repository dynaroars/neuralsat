
from pathlib import Path
import torch.nn as nn
import random
import torch
import time
import os

from verifier.verifier import Verifier 
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx, custom_quirks
from verifier.objective import Objective, DnfObjectives
from util.misc.logger import logger
from setting import Settings

from auto_LiRPA import BoundedModule
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor
from abstractor.params import *
from auto_LiRPA.utils import stop_criterion_batch_any, multi_spec_keep_func_all, stop_criterion_all

custom_quirks['Reshape']['fix_batch_size'] = False

def new_input(x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
    return BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(x_L.device)

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


if __name__ == "__main__":
    net_path = 'example/onnx/pgd_2_3_16.onnx'
    vnnlib_path = 'example/vnnlib/pgd_2_3_16_4021.vnnlib'
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    print(model)
    print(custom_quirks)
    device = 'cpu'
    device = 'cuda'
    
    bound_opts = {
        'verbosity': 1, 
        'conv_mode': 'matrix', 
        'deterministic': False, 
        'sparse_features_alpha': True, # no
        'sparse_spec_alpha': True, # no
        'sparse_intermediate_bounds': False, # yes (backward)
        'crown_batch_size': 1000000000, 
        'max_crown_size': 1000000000, 
        'forward_refinement': False, 
        'dynamic_forward': False, 
        'forward_max_dim': 10000, 
        'use_full_conv_alpha': True, 
        'disable_optimization': [], 
        'fixed_reducemax_index': True, 
        'matmul': {'share_alphas': False}, # yes
        'tanh': {'loose_threshold': None}, 
        'buffers': {'no_batchdim': False}, # yes (optimized backward)
        'optimize_graph': {'optimizer': None}
    }

    net = BoundedModule(
        model=model, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts=bound_opts,
        device=device,
        verbose=0,
    )
    
    x_L = objectives.lower_bounds[0].view(input_shape).to(device)
    x_U = objectives.upper_bounds[0].view(input_shape).to(device)
    x = new_input(x_L=x_L, x_U=x_U)
    cs = objectives.cs.transpose(0, 1).to(device)
    rhs = objectives.rhs.transpose(0, 1).to(device)
    
    print(net(x))
    net.get_split_nodes(input_split=False)
    print(net.split_nodes)
    print(net.split_activations)
    print(cs.shape)
    print(rhs.shape)
    
    net.set_bound_opts({
        'optimize_bound_args': {'stop_criterion_func': stop_criterion_all(rhs)}, 'verbosity': 1,
    })
    net.set_bound_opts({
        'optimize_bound_args': {
            'enable_alpha_crown': True, 
            'enable_beta_crown': False, 
            'init_alpha': False, 
            'fix_interm_bounds': True, 
            'use_shared_alpha': False, 
            'early_stop_patience': 10, 
        }
    })

    lb, _, aux_reference_bounds = net.init_alpha(
        (x,), share_alphas=False, c=cs, bound_upper=False)
    print('initial CROWN bounds:', lb,)
    
    lb, _ = net.compute_bounds(
        x=(x,), C=cs, method='CROWN-Optimized',
        bound_upper=False, 
        aux_reference_bounds=aux_reference_bounds,
    )    
    
    print('initial CROWN-Optimized bounds:', lb,)
    