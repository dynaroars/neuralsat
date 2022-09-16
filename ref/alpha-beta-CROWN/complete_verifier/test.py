import numpy as np
import random
import torch
import time
import os

from batch_branch_and_bound_input_split import relu_bab_parallel
from beta_CROWN_solver_input_split import LiRPAConvNet
from auto_LiRPA import BoundedModule, BoundedTensor
from read_vnnlib import read_vnnlib_simple
from auto_LiRPA.perturbations import *
from utils import *


seed = 5
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def mnist_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    return model


def bab(model_ori, data, target, norm, eps, decision_thresh, y, prop_mat, prop_rhs, data_ub=None, data_lb=None, c=None, shape=None, all_prop=None):

    # LiRPA wrapper
    model = LiRPAConvNet(model_ori, y, target, solve_slope=False, device='cpu', in_size=data.shape, simplify=True if c is None else False, c=c)
    if list(model.net.parameters())[0].is_cuda:
        data = data.cuda()
        data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

    ptb = PerturbationLpNorm(norm=norm, eps=None, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.cat([data_lb, data_ub])
    min_lb, min_input, glb_record, nb_states = relu_bab_parallel(model, domain, x, batch=1,
                                                                 decision_thresh=decision_thresh,
                                                                 iteration=2, 
                                                                 shape=shape,
                                                                 timeout=100, 
                                                                 lr_alpha=1e-2,
                                                                 lr_init_alpha=1e-1,
                                                                 branching_candidates=1,
                                                                 share_slopes=False,
                                                                 branching_method='kfsb',
                                                                 prop_mat=prop_mat, 
                                                                 prop_rhs=prop_rhs, 
                                                                 model_ori=model_ori,
                                                                 all_prop=all_prop)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    return min_lb, nb_states, min_input

def main():
    shape = (1, 1)
    onnx_path = 'tests/test_tiny.onnx'
    vnnlib_path = 'tests/test_tiny.vnnlib'

    # model_ori = load_model_onnx(onnx_path, input_shape=(1,))
    model_ori = mnist_model()
    vnnlib = read_vnnlib_simple(vnnlib_path, 1, 2)
    timeout = 100
    vnnlib_shape = shape
    verified_ret = []
    min_input = None

    for vnn in vnnlib:
        x_range = torch.tensor(vnn[0])
        data_min = x_range[:, 0].reshape(vnnlib_shape)
        data_max = x_range[:, 1].reshape(vnnlib_shape)

        x = x_range.mean(1).reshape(vnnlib_shape)  # only the shape of x is important.
        eps_temp = 0.5 * (data_max - data_min).flatten(-2).mean(-1).reshape(1, -1, 1, 1)

        print(x, eps_temp)
        c = None
        for prop_mat, prop_rhs in vnn[1]:
            if len(prop_rhs) > 1:
                c = torch.tensor(prop_mat).unsqueeze(0).type_as(data_max)
                y = np.where(prop_mat == 1)[1]  # true label
                pidx = np.where(prop_mat == -1)[1]  # target label
                decision_thresh = prop_rhs[0]
            else:
                print(prop_mat)
                y = np.where(prop_mat[0] == 1)[0]
                if len(y) != 0:
                    y = int(y)
                else:
                    y = None  # no true label
                pidx = int(np.where(prop_mat[0] == -1)[0])  # target label
                decision_thresh = prop_rhs[0]  # already flipped in read_vnnlib_simple()

            print(pidx, y)
            # Main function to run verification
            l, nodes, min_input = bab(model_ori, x, pidx, np.inf, eps_temp, decision_thresh, y, prop_mat, prop_rhs, data_ub=data_max, data_lb=data_min, c=c, shape=shape, all_prop=vnn[1])

            if min_input is not None:
                # adv example found
                verified_ret.append(['SAT'])
            else:
                # all props verified
                verified_ret.append(['UNSAT'])
            print(verified_ret)

if __name__ == "__main__":
    main()