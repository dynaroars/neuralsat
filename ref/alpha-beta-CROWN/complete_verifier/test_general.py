import numpy as np
import random
import torch
import time
import os

from auto_LiRPA import BoundedModule, BoundedTensor
from beta_CROWN_solver import LiRPAConvNet
from read_vnnlib import read_vnnlib_simple
from auto_LiRPA.perturbations import *
from batch_branch_and_bound import relu_bab_parallel
from utils import *
import arguments


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


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    arguments.Config.add_argument("--csv_name", type=str, default=None, help='Name of .csv file containing a list of properties to verify (VNN-COMP specific).', hierarchy=h + ["csv_name"])
    arguments.Config.add_argument("--onnx_path", type=str, default=None, help='Path to .onnx model file.', hierarchy=h + ["onnx_path"])
    arguments.Config.add_argument("--vnnlib_path", type=str, default=None, help='Path to .vnnlib specification file.', hierarchy=h + ["vnnlib_path"])
    arguments.Config.add_argument("--results_file", type=str, default=None, help='Path to results file.', hierarchy=h + ["results_file"])
    arguments.Config.add_argument("--root_path", type=str, default=None, help='Root path of VNN-COMP benchmarks (VNN-COMP specific).', hierarchy=h + ["root_path"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="mnist_9_200", help='Model name.', hierarchy=h + ["name"])

    h = ["data"]
    arguments.Config.add_argument("--dataset", type=str, default="CIFAR", choices=["MNIST", "CIFAR", "CIFAR_SDP_FULL", "CIFAR_RESNET", "CIFAR_SAMPLE", "MNIST_SAMPLE", "CIFAR_ERAN", "MNIST_ERAN",
                                 "MNIST_ERAN_UN", "MNIST_SDP", "MNIST_MADRY_UN", "CIFAR_SDP", "CIFAR_UN", "NN4SYS", "TEST"], help='Dataset name. Dataset must be defined in utils.py.', hierarchy=h + ["dataset"])
    arguments.Config.add_argument("--filter_path", type=str, default=None, help='A filter in pkl format contains examples that will be skipped (not used).', hierarchy=h + ["data_filter_path"])

    h = ["attack"]
    arguments.Config.add_argument("--mip_attack", action='store_true', help='Use MIP (Gurobi) based attack if PGD cannot find a successful adversarial example.', hierarchy=h + ["enable_mip_attack"])
    arguments.Config.add_argument('--pgd_steps', type=int, default=100, help="Steps of PGD attack.", hierarchy=h + ["pgd_steps"])
    arguments.Config.add_argument('--pgd_restarts', type=int, default=30, help="Number of random PGD restarts.", hierarchy= h + ["pgd_restarts"])
    arguments.Config.add_argument('--no_pgd_early_stop', action='store_false', dest='pgd_early_stop', help="Early stop PGD when an adversarial example is found.", hierarchy=h + ["pgd_early_stop"])
    arguments.Config.add_argument('--pgd_lr_decay', type=float, default=0.99, help='Learning rate decay factor used in PGD attack.', hierarchy= h + ["pgd_lr_decay"])
    arguments.Config.add_argument('--pgd_alpha', type=str, default="auto", help='Step size of PGD attack. Default (auto) is epsilon/4.', hierarchy=h + ["pgd_alpha"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option. Do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()

def bab(unwrapped_model, data, target, y, eps=None, data_ub=None, data_lb=None, lower_bounds=None, upper_bounds=None, reference_slopes=None, attack_images=None):
    num_outputs = 1
    device = 'cpu'
    norm = np.inf
    # assert num_outputs > 1
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

    # This will use the refined bounds if the complete verifier is "bab-refine".
    model = LiRPAConvNet(unwrapped_model, y, target, device=device, in_size=data.shape, deterministic=False, conv_mode='patches', c=c)
   
    print(f'Model prediction of {data} is:', model.net(data))

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb).to(data_lb.device)
    domain = torch.stack([data_lb.squeeze(0), data_ub.squeeze(0)], dim=-1)

    print('Range:', domain)

    min_lb, min_ub, glb_record, nb_states = relu_bab_parallel(model, domain, x, refined_lower_bounds=lower_bounds, refined_upper_bounds=upper_bounds, reference_slopes=reference_slopes, attack_images=attack_images)

    if isinstance(min_lb, torch.Tensor):
        min_lb = min_lb.item()
    if isinstance(min_ub, torch.Tensor):
        min_ub = min_ub.item()
    if min_ub is None:
        min_ub = float('inf')

    return min_lb, min_ub, nb_states, glb_record

def main():
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    shape = (1, 1)
    onnx_path = 'tests/test_tiny.onnx'
    vnnlib_path = 'tests/test_tiny.vnnlib'

    # model_ori = load_model_onnx(onnx_path, input_shape=(1,))
    model_ori = mnist_model()
    vnnlib = read_vnnlib_simple(vnnlib_path, 1, 1)
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
                arguments.Config["bab"]["decision_thresh"] = prop_rhs[0]
            # Main function to run verification

            print(y)
            l, u, nodes, _ = bab(model_ori, x, pidx, y, data_ub=data_max, data_lb=data_min)

            if min_input is not None:
                # adv example found
                verified_ret.append(['SAT'])
            else:
                # all props verified
                verified_ret.append(['UNSAT'])
            print(verified_ret)

if __name__ == "__main__":
    config_args()
    arguments.Config["general"]["seed"] = 0
    arguments.Config["general"]["vnnlib_path"] = 'tests/test_tiny.vnnlib'
    arguments.Config["data"]["dataset"] = 'TEST'
    arguments.Config["attack"]["pgd_order"] = 'skip'
    arguments.Config["general"]["enable_incomplete_verification"] = False
    arguments.Config["data"]["num_outputs"] = 1
    arguments.Config["solver"]["beta-crown"]["beta"] = False
    print(arguments.Config["bab"]["timeout"])
    main()