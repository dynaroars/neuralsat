

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
    '''One-shot method with LP propagator solved by Gurobi.'''

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

    horizon = 7

    save_file_name = 'LP_result/gurobi_one_shot' + '_width_' + str(nn_width) + '_horizon_' + str(horizon)

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

        result = {'seq_bds': output_bds_list_seq, 'seq_pre_act_bds_list': pre_act_bds_list_seq,
                  'seq_solver_time': solver_time_seq_list}
        ut.pickle_file(result, save_file_name)


    pre_act_bds_list_seq = bounds_list[:-1]

    print('sequential:', output_bds_list_seq, 'solver time:', solver_time_seq_list)

    result = {'seq_bds': output_bds_list_seq, 'seq_pre_act_bds_list': pre_act_bds_list_seq, 'seq_solver_time': solver_time_seq_list}
    ut.pickle_file(result, save_file_name)


    seq_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_seq)

    # plot on a fixed horizon
    plt.figure()
    x0_lb_cpu = x0_lb.to(torch.device('cpu'))
    x0_ub_cpu = x0_ub.to(torch.device('cpu'))

    domain = Polyhedron.from_bounds(x0_lb_cpu[0].numpy(), x0_ub_cpu[0].numpy())

    bounds_list = [[x0_lb_cpu[0][k].item(), x0_ub[0][k].item()] for k in range(nx)]
    init_states = ut.random_unif_sample_from_box(bounds_list, 300)
    # init_states = ut.unif_sample_from_Polyhedron(domain, 3)

    plot_dim = [2, 3]
    hor = horizon
    for i in range(hor):
        seq_poly_list[i].plot(fill=False, residual_dimensions=plot_dim, ec='r', linestyle='-.', linewidth=1)

    traj_list = ut.simulate_NN_system(nn_system, init_states, step=hor-1)

    ut.plot_multiple_traj_tensor_to_numpy(traj_list, dim=plot_dim)
    domain.plot(fill=False, residual_dimensions=plot_dim, ec='k', linestyle='--', linewidth=2)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')



    ##############################################################################################################
    # post processing: compare the runtime of Gurobi and ADMM in one-shot method for horizon = 7; generate Fig.4
    ##############################################################################################################
    result = ut.load_pickle_file('LP_result/gurobi_one_shot_width_100_horizon_7')
    output_bds_list_seq = result['seq_bds']
    seq_solver_time = result['seq_solver_time']
    LP_solver_time_list = [sum(item) for item in seq_solver_time]

    # import ADM results
    ADMM_result_file = 'ADMM_result/ADMM_result_width_100_horizon_7_eps_abs_1e-05'
    ADMM_result = torch.load(ADMM_result_file)
    output_bds_ADMM = ADMM_result['output_bds']
    ADMM_solver_time = sum(ADMM_result['runtime_pre_act_bds'])+sum(ADMM_result['runtime_output_bds'])
    ADMM_solver_time_pre_act = ADMM_result['runtime_pre_act_bds']
    ADMM_solver_time_output = ADMM_result['runtime_output_bds']
    ADMM_solver_time_list = [ADMM_solver_time_pre_act[i] + ADMM_solver_time_output[i] for i in range(len(ADMM_solver_time_output))]

    N = len(ADMM_solver_time_list)
    ADMM_solver_time_cummulative_100 = [sum(ADMM_solver_time_list[:i+1]) for i in range(N)]
    LP_solver_time_cummulative_100 = [sum(LP_solver_time_list[:i+1]) for i in range(N)]

    # layer = 400
    result = ut.load_pickle_file('LP_result/gurobi_one_shot_width_400_horizon_7')
    output_bds_list_seq = result['seq_bds']
    output_bds_LP = result['seq_bds']
    seq_solver_time = result['seq_solver_time']
    LP_solver_time_list = [sum(item) for item in seq_solver_time]

    # import ADM results
    ADMM_result_file = 'ADMM_result/ADMM_result_width_400_horizon_7_eps_abs_1e-05'
    ADMM_result = torch.load(ADMM_result_file)
    output_bds_ADMM = ADMM_result['output_bds']
    ADMM_solver_time = sum(ADMM_result['runtime_pre_act_bds'])+sum(ADMM_result['runtime_output_bds'])
    ADMM_solver_time_pre_act = ADMM_result['runtime_pre_act_bds']
    ADMM_solver_time_output = ADMM_result['runtime_output_bds']
    ADMM_solver_time_list = [ADMM_solver_time_pre_act[i] + ADMM_solver_time_output[i] for i in range(len(ADMM_solver_time_output))]

    N = len(ADMM_solver_time_list)
    ADMM_solver_time_cummulative_400 = [sum(ADMM_solver_time_list[:i+1]) for i in range(N)]
    LP_solver_time_cummulative_400 = [sum(LP_solver_time_list[:i+1]) for i in range(N)]


    # runtime plot
    plt.figure(figsize=(9, 6), dpi=80)
    plt.semilogy(list(range(1,len(LP_solver_time_list)+1)), LP_solver_time_cummulative_100, 's-.', markersize = 10, linewidth = 2, label = 'Gurobi, width 100')
    plt.semilogy(list(range(1,len(ADMM_solver_time_list)+1)), ADMM_solver_time_cummulative_100, 'o-.', markersize = 10,  linewidth = 2, label = 'ADMM, width 100')
    plt.semilogy(list(range(1, len(LP_solver_time_list) + 1)), LP_solver_time_cummulative_400, 's-', markersize=10,
                 linewidth=2, label='Gurobi, width 400')
    plt.semilogy(list(range(1, len(ADMM_solver_time_list) + 1)), ADMM_solver_time_cummulative_400, 'o-', markersize=10,
                 linewidth=2, label='ADMM, width 400')
    plt.xlabel(r'time step k', fontsize = 18)
    plt.ylabel(r'cumulative solver time [s]', fontsize = 18)
    plt.grid()
    plt.legend()


    plt.show()
