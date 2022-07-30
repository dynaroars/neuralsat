
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut
from nn_reachability.system_models import CartPoleSystem
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel, SystemDataSet, train_nn_torch
from nn_reachability.ADMM import init_sequential_admm_session, run_ADMM, intermediate_bounds_from_ADMM, InitModule, ADMM_Session
from nn_reachability.nn_models import iterative_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA
from nn_reachability.nn_models import Gurobi_reachable_set, ADMM_reachable_set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nn_dynamics(n=2, m = 100):
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(n, m),
        nn.ReLU(),
        nn.Linear(m, m),
        nn.ReLU(),
        nn.Linear(m, n)
    )
    return model


if __name__ == '__main__':

    nx = 4
    dt = 0.05

    # train a neural network
    is_train = False
    nn_width = 100

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)

    # original test bed
    # x0_lb = torch.tensor([[2.0, 1.0, -10*np.pi/180, -1.0]]).to(device)
    # x0_ub = torch.tensor([[2.2, 1.2, -6*np.pi/180, -0.8]]).to(device)

    x0_lb = torch.tensor([[2.0, 1.0, -10*np.pi/180, -1.0]])
    x0_ub = torch.tensor([[2.2, 1.2, -6*np.pi/180, -0.8]])

    domain = Polyhedron.from_bounds(x0_lb[0].numpy(), x0_ub[0].numpy())

    horizon = 2
    # reachability analysis through sequential nn
    print('Sequential reachability analysis \n')
    # method = 'CROWN-Optimized'
    method = 'forward'
    box = {'lb': x0_lb[0].numpy(), 'ub': x0_ub[0].numpy()}
    seq_bounds_list = [box]
    for i in range(horizon):
        nn_model = SequentialModel(nn_system, i+1)
        output_lb, output_ub = output_Lp_bounds_LiRPA(nn_model, x0_lb, x0_ub, method= method)
        # fixme: we only extract the bounds for the first batch
        box = {'lb': output_lb[0].detach().numpy(), 'ub': output_ub[0].detach().numpy()}
        seq_bounds_list.append(box)

    # reachability analysis through iterative methods
    print('Iterative reachability analysis \n')
    base_nn_system = SequentialModel(nn_system, 1)
    iter_bounds_list = iterative_output_Lp_bounds_LiRPA(base_nn_system, x0_lb, x0_ub, horizon, method= method)
    iter_bounds_list = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in iter_bounds_list]

    seq_poly_list = ut.bounds_list_to_polyhedron_list(seq_bounds_list)
    iter_poly_list = ut.bounds_list_to_polyhedron_list(iter_bounds_list)

    # plot
    hor = horizon

    bounds_list = [[x0_lb[0][k].item(), x0_ub[0][k].item()] for k in range(nx)]
    init_states = ut.random_unif_sample_from_box(bounds_list, 30)
    traj_list = ut.simulate_NN_system(nn_system, init_states, step=hor - 1)


    plt.figure(figsize=(9, 6), dpi=80)
    plot_dim = [2,3]
    for i in range(1,hor):
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5)
        # iter_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='b', linestyle='-.', linewidth=2)
    seq_poly_list[hor].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5,
                                label='one-shot')

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim, color='grey', linewidth=0.5)

    # plot iter method
    for i in range(1,hor):
        iter_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5)
    iter_poly_list[hor].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5,
                           label='recursive')
    domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)

    plt.xlabel(r'$x_3$', fontsize=18)
    plt.ylabel(r'$x_4$', fontsize=18)
    plt.grid()
    plt.legend()

