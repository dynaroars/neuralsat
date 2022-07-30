
import torch
import torch.nn as nn

import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

import matplotlib.pyplot as plt
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
    '''One-shot method with LP propagator solved by ADMM.'''

    nx = 4
    dt = 0.05

    # train a neural network
    is_train = False
    nn_width = 100

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)

    # initial set
    x0_lb = torch.tensor([[2.0, 1.0, -10*np.pi/180, -1.0]]).to(device)
    x0_ub = torch.tensor([[2.2, 1.2, -6*np.pi/180, -0.8]]).to(device)

    horizon = 7
    # ADMM agorithm parameters
    alg_options = {'rho': 0.1, 'eps_abs': 1e-5, 'eps_rel': 1e-4, 'residual_balancing': False, 'max_iter': 20000,
                   'record': False, 'verbose': True, 'alpha': 1.6}

    file_name = 'ADMM_result/ADMM_result_width_' + str(nn_width) + '_horizon_' + str(horizon) + '_eps_abs_' + str(alg_options['eps_abs'])
    output_bds_ADMM, ADMM_result = ADMM_reachable_set(nn_system, x0_lb, x0_ub, horizon, alg_options= alg_options, load_file= 0, file_name=file_name)
    seq_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_ADMM)

    print('run time:{}'.format(sum(ADMM_result['runtime_pre_act_bds'])+sum(ADMM_result['runtime_output_bds']) ))

    # plot
    x0_lb_cpu = x0_lb.to(torch.device('cpu'))
    x0_ub_cpu = x0_ub.to(torch.device('cpu'))

    domain = Polyhedron.from_bounds(x0_lb_cpu[0].numpy(), x0_ub_cpu[0].numpy())

    bounds_list = [[x0_lb_cpu[0][k].item(), x0_ub[0][k].item()] for k in range(nx)]
    init_states = ut.random_unif_sample_from_box(bounds_list, 100)

    # plot

    plot_dim = [2, 3]
    for i in range(horizon):
        plt.figure()
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='r', linestyle='-', linewidth=2)

        traj_list = ut.simulate_NN_system(nn_system.to(x0_lb_cpu.device), init_states, step=i - 1)

        ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim)
        domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)
        plt.title('Reachable set at step {}'.format(i + 1))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    plt.show()

    # figure 2
    plt.figure()
    plot_dim = [2,3]
    hor = horizon
    for i in range(hor):
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='r', linestyle='-', linewidth=2)

    traj_list = ut.simulate_NN_system(nn_system.to(x0_lb_cpu.device), init_states, step=hor-1)

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim)
    domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)
    plt.title('Reachable set at step {}'.format(i + 1))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


