from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch
import copy
import time


from .auto_LiRPA import BoundedModule, BoundedTensor
from .auto_LiRPA.perturbations import *
from .auto_LiRPA.utils import *


def reduction_str2func(reduction_func):
    if type(reduction_func) == str:
        if reduction_func == 'min':
            return reduction_min
        elif reduction_func == 'max':
            return reduction_max
        elif reduction_func == 'sum':
            return reduction_sum
        elif reduction_func == 'mean':
            return reduction_mean
        else:
            raise NotImplementedError(f'Unknown reduction_func {reduction_func}')
    else:
        return reduction_func


class LiRPANaive:

    def __init__(self, model_ori, input_shape, c=None, rhs=None, device='cpu', conv_mode='patches'):

        net = copy.deepcopy(model_ori)
        layers = list(net.children())
        self.c = c
        self.rhs = rhs
        self.model_ori = model_ori
        self.layers = layers
        self.input_shape = input_shape
        self.device = device
        
        self.net = BoundedModule(
            net, torch.zeros(input_shape, device=self.device),
            bound_opts={
                'relu': 'adaptive',
                'deterministic': False,
                'conv_mode': conv_mode,
                'sparse_features_alpha': True,
                'sparse_spec_alpha': True,
                'crown_batch_size': 1e9,
                'max_crown_size': 1e9,
                'forward_refinement': False,
                'dynamic_forward': False,
                'forward_max_dim': 1e9,
                'use_full_conv_alpha': True,
            },
            device=self.device
        )
        self.net.eval()

        # check conversion correctness
        dummy = torch.randn(input_shape, device=self.device)
        try:
            assert torch.allclose(net(dummy), self.net(dummy))
        except AssertionError:
            print(f'torch allclose failed: norm {torch.norm(net(dummy) - self.net(dummy))}')


    @torch.no_grad()
    def __call__(self, dm_l, dm_u, stop_criterion_func=None, reset_param=True):
        batch = len(dm_l)
        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=dm_l, x_U=dm_u)
        new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
        C = self.c.repeat(batch, *[1] * len(self.c.shape[1:]))

        with torch.no_grad():
            lb, _ = self.net.compute_bounds(x=(new_x,), C=C, method="backward", bound_upper=False)
            lb = lb.cpu()
        ub = [None] * (batch)

        return (lb, ub), None

    