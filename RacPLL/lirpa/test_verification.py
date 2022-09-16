import torch.nn as nn
import numpy as np
import random
import torch
import time
import os


from auto_lirpa.perturbations import PerturbationLpNorm
from auto_lirpa.bound_general import BoundedModule
from auto_lirpa.bound_tensor import BoundedTensor
from read_vnnlib import read_vnnlib_simple
from verifier.bab import relu_bab_parallel
from verifier.crown import LiRPAConvNet
import config


def mnist_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1,32),
        nn.ReLU(),
        nn.Linear(32,32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model



def bab(unwrapped_model, data, target, y, eps=None, data_ub=None, data_lb=None, lower_bounds=None, upper_bounds=None, reference_slopes=None, attack_images=None):
    norm = config.Config["specification"]["norm"]
    num_outputs = config.Config["data"]["num_outputs"]
    device = config.Config["general"]["device"]

    if y is not None:
        if num_outputs > 1:
            c = torch.zeros((1, 1, num_outputs), device=device)  # we only support c with shape of (1, 1, n)
            c[0, 0, y] = 1
            c[0, 0, target] = -1
        else:
            # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
            c = (float(y) - 0.5) * 2 * torch.ones(size=(1, 1, 1))
    else:
        # if there is no ture label, we only verify the target output
        c = torch.zeros((1, 1, num_outputs), device=device)  # we only support c with shape of (1, 1, n)
        c[0, 0, target] = -1

    model = LiRPAConvNet(unwrapped_model, y, target, device=device, in_size=data.shape, deterministic=config.Config["general"]["deterministic"], conv_mode=config.Config["general"]["conv_mode"], c=c)

    # if list(model.net.parameters())[0].is_cuda:
    data = data.to(device)
    data_lb, data_ub = data_lb.to(device), data_ub.to(device)

    print('Model prediction is:', model.net(data))

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    print(config.Config["general"]["seed"], x)

    min_lb, min_ub, glb_record, nb_states = relu_bab_parallel(model, domain, x, refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds, reference_slopes=reference_slopes, attack_images=attack_images)

    exit()


if __name__ == '__main__':
    config.Config.parse_config()

    torch.manual_seed(config.Config["general"]["seed"])
    random.seed(config.Config["general"]["seed"])
    np.random.seed(config.Config["general"]["seed"])

    config.Config["data"]["num_outputs"] = 1
    config.Config["data"]["shape"] = (1, 1)
    config.Config["general"]["vnnlib_path"] = 'data/test_tiny.vnnlib'
    config.Config["solver"]["beta-crown"]["beta"] = False

    # hardcorded model
    model_ori = mnist_model()

    # Main function to run verification
    vnnlib = read_vnnlib_simple(config.Config["general"]["vnnlib_path"], 1, 1)
    vnnlib_shape = config.Config["data"]["shape"]

    for vnn in vnnlib:
        x_range = torch.tensor(vnn[0])
        data_min = x_range[:, 0].reshape(vnnlib_shape)
        data_max = x_range[:, 1].reshape(vnnlib_shape)

        x = x_range.mean(1).reshape(vnnlib_shape)  # only the shape of x is important.
        # eps_temp = 0.5 * (data_max - data_min).flatten(-2).mean(-1).reshape(1, -1, 1, 1)
        # c = None
        for prop_mat, prop_rhs in vnn[1]:
            if len(prop_rhs) > 1:
                raise
            else:
                print(prop_mat)
                y = np.where(prop_mat[0] == 1)[0]
                if len(y) != 0:
                    y = int(y)
                else:
                    y = None  # no true label
                pidx = np.where(prop_mat[0] == -1)[0]  # target label
                pidx = int(pidx) if len(pidx) != 0 else None  # Fix constant specification with no target label.
                if y is not None and pidx is None:
                    y, pidx = pidx, y  # Fix vnnlib with >= const property.
                config.Config["bab"]["decision_thresh"] = prop_rhs[0]

            # verify
            l, u, nodes, _ = bab(model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min)
