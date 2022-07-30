
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp
import numpy as np
import copy

from tqdm import tqdm
import time

from convex_adversarial import DualNetBounds, DualNetwork, ParallelDualNetwork
from convex_adversarial.dual_network import RobustBounds

def pre_act_bds_divide_and_conquer(nn_model, x0, eps, mini_batch_size=1):
    N = x0.size(0)
    num_iter = N // mini_batch_size
    res = N % mini_batch_size

    lb_list = []
    ub_list = []
    count = 0

    for i in range(num_iter):
        x_crop = x0[i * mini_batch_size:(i + 1) * mini_batch_size, :]
        pre_min, pre_max = pre_activation_bounds_dual(nn_model, x_crop, eps)

        if count == 0:
            lb_list = pre_min
            ub_list = pre_max
        else:
            lb_list = [torch.cat((lb_list[j], pre_min[j])) for j in range(len(lb_list))]
            ub_list = [torch.cat((ub_list[j], pre_max[j])) for j in range(len(ub_list))]

        torch.cuda.empty_cache()
        count += 1

    if res > 0:
        x_crop = x0[num_iter * mini_batch_size:, :]
        pre_min, pre_max = pre_activation_bounds_dual(nn_model, x_crop, eps)
        if count == 0:
            lb_list = pre_min
            ub_list = pre_max
        else:
            lb_list = [torch.cat((lb_list[j], pre_min[j])) for j in range(len(lb_list))]
            ub_list = [torch.cat((ub_list[j], pre_max[j])) for j in range(len(ub_list))]
        count += 1

    return lb_list, ub_list



def pre_activation_bounds_dual(model, X, epsilon):
    '''
    This function compute the bounds on the activation values of the neural
    network model assuming that the input to the neural net is L_inf ball with
    center X and radius epsilon
    Parameters
    ----------
    model : pytorch nn sequential
    a fully connected neural network.
    X : tensor
    center of the input region.
    epsilon : tensor
    radius of the input region (in l_inf norm).
    Returns
    -------
    l : tensor
    lower bound on the activation values.
    u : tensor
    upper bounds on the activation values.
    '''

    dual = DualNetwork(model, X, epsilon, proj = None, norm_type = 'l1', bounded_input = False)

    l, u = [], []
    for dual_layer in dual.dual_net:
        if 'DualReLU' in str(dual_layer.__class__):
            l.append(dual_layer.zl)
            u.append(dual_layer.zu)
            '''double check bounds'''
            assert torch.all(dual_layer.zl < dual_layer.zu)

    return l, u

