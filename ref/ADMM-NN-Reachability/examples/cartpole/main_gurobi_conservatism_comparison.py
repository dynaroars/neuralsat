
import torch
import torch.nn as nn

import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel

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
    '''Compare the one-shot (sequential) and recursive (iterative) methods with LP propagator solved by Gurobi.
       Generates FIg.2 in the paper.'''

    nx = 4

    # train a neural network
    nn_width = 100

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)

    x0_lb = torch.tensor([[2.0, 1.0, -10*np.pi/180, -1.0]]).to(torch.device('cpu'))
    x0_ub = torch.tensor([[2.2, 1.2, -6*np.pi/180, -0.8]]).to(torch.device('cpu'))

    A_input = np.vstack((np.eye(nx), -np.eye(nx)))

    input_lb = x0_lb.to(torch.device('cpu')).numpy()
    input_ub = x0_ub.to(torch.device('cpu')).numpy()

    b_input = np.concatenate((input_ub, -input_lb)).flatten()

    horizon = 20
    # reachability analysis through one-shot/sequential method
    print('Sequential reachability analysis \n')
    output_bds_list_seq = []
    solver_time_seq_list = []
    seq_nn_system = SequentialModel(nn_system, 1)
    for i in range(horizon):
        print('step {} \n'.format(i+1))
        seq_nn_system.reset_horizon(i+1)
        bounds_list, solver_time_seq = seq_nn_system.output_inf_bounds_LP(A_input, b_input, None, file_name=None)
        output_bds = bounds_list[-1]
        output_bds_list_seq.append(output_bds)
        solver_time_seq_list.append(solver_time_seq)

    pre_act_bds_list_seq = bounds_list[:-1]

    # reachability analysis through recursive/iterative methods
    print('Iterative reachability analysis \n')
    output_bds_list_iter = []
    solver_time_iter_list = []
    A_input_iter, b_input_iter = A_input, b_input
    pre_act_bds_list_iter = []
    for i in range(horizon):
        base_nn_system = SequentialModel(nn_system, 1)
        bounds_list, solver_time_iter = base_nn_system.output_inf_bounds_LP(A_input_iter, b_input_iter, None, file_name=None)
        pre_act_bds_list_iter = pre_act_bds_list_iter + bounds_list[:-1]
        output_bds = bounds_list[-1]
        output_bds_list_iter.append(output_bds)
        output_lb, output_ub = output_bds['lb'], output_bds['ub']
        output_box = Polyhedron.from_bounds(output_lb, output_ub)
        A_input_iter, b_input_iter = output_box.A, output_box.b

        solver_time_iter_list.append(solver_time_iter)

    print('sequential:', output_bds_list_seq, 'solver time:', solver_time_seq_list)
    print('iterative:', output_bds_list_iter, 'solver_time:', solver_time_iter_list)
    # print('solver time comparison: sequentail:', str(sum(solver_time_seq_list)), ' iterative:', str(sum(solver_time_iter_list)))

    result = {'seq_bds': output_bds_list_seq, 'seq_pre_act_bds_list': pre_act_bds_list_seq, 'seq_solver_time': solver_time_seq_list,
              'iter_bds': output_bds_list_iter, 'iter_pre_act_bds_list': pre_act_bds_list_iter, 'iter_solver_time': solver_time_iter_list}
    ut.pickle_file(result, 'gurobi_comparison_horizon_' + str(horizon))

    result = ut.load_pickle_file('gurobi_comparison_horizon_' + str(horizon))
    iter_time = result['iter_solver_time']
    seq_time = result['seq_solver_time']
    iter_time_summary = [sum(item) for item in iter_time]
    seq_time_summary = [sum(item) for item in seq_time]

    # plt.figure()
    # solver_time_seq_accumulated_iter = [sum(solver_time_iter_list[:i+1]) for i in range(horizon)]
    # plt.semilogy(solver_time_seq_list, 'ro-', label = 'sequential')
    # plt.semilogy(solver_time_seq_accumulated_iter,'bs-.', label = 'iterative')
    # plt.title('solver time comparison')
    # plt.xlabel('step')
    # plt.ylabel('solver time [sec]')

    # construct polyhedra from box lower and upper bounds
    seq_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_seq)
    iter_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_iter)

    ## post processing
    result_seq = ut.load_pickle_file('gurobi_comparison_horizon_30')
    output_bds_list_seq = result_seq['seq_bds']
    seq_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_seq)

    result_iter = ut.load_pickle_file('gurobi_comparison_horizon_20')
    output_bds_list_iter = result_iter['iter_bds']
    iter_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_iter)

    seq_vol_list = []
    iter_vol_list = []
    seq_vol_list = [np.prod(bd['ub']-bd['lb']) for bd in output_bds_list_seq]
    iter_vol_list = [np.prod(bd['ub']-bd['lb']) for bd in output_bds_list_iter]
    ratio_list = [iter_vol_list[i]/seq_vol_list[i] for i in range(8)]

    ## plot on [x_1, x_2] plane
    # plot sequential method
    hor = 30
    domain = Polyhedron(A_input, b_input)
    bounds_list = [[x0_lb[0][k].item(), x0_ub[0][k].item()] for k in range(nx)]
    init_states = ut.random_unif_sample_from_box(bounds_list, 100)
    traj_list = ut.simulate_NN_system(nn_system, init_states, step=hor-1)

    plt.figure(figsize=(9, 6), dpi=80)
    plot_dim = [0,1]

    for i in range(hor-1):
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5)
    seq_poly_list[hor-1].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5, label ='one-shot')

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim, color = 'grey', linewidth = 0.5)
    domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)

    # plot iter method
    # only show 7 steps of recursive/iterative method since the over-approximation explodes
    for i in range(7):
        iter_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5)
    iter_poly_list[7].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5, label = 'recursive')

    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.grid()
    plt.legend()


    ## plot on [x_3, x_4] plane
    plt.figure(figsize=(9, 6), dpi=80)
    plot_dim = [2,3]

    for i in range(hor - 1):
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5)
        # iter_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='b', linestyle='-.', linewidth=2)
    seq_poly_list[hor - 1].plot(fill=False, residual_dimensions=plot_dim, ec='tab:red', linestyle='-', linewidth=1.5,
                                label='one-shot')

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim, color='grey', linewidth=0.5)
    domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)

    # plot iter method
    for i in range(7):
        iter_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5)
    iter_poly_list[7].plot(fill=False, residual_dimensions=plot_dim, ec='tab:blue', linestyle='-.', linewidth=1.5,
                           label='recursive')

    plt.xlabel(r'$x_3$', fontsize=18)
    plt.ylabel(r'$x_4$', fontsize=18)
    plt.grid()
    plt.legend()

    plt.show()
