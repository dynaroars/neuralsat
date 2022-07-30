
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")
import cvxpy as cp
import numpy as np

from pympc.geometry.polyhedron import Polyhedron
import nn_reachability.utilities as ut
import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel, SystemDataSet, train_nn_torch
from nn_reachability.ADMM import init_sequential_admm_session, run_ADMM, intermediate_bounds_from_ADMM, InitModule, ADMM_Session
from nn_reachability.nn_models import iterative_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA
from nn_reachability.nn_models import preactivation_bounds_of_sequential_nn_model_LiRPA, pre_act_bds_tensor_to_numpy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' FitzHugh-Nagumo Neuron Model from 
Semidefinite Approximations of Reachable Sets for Discrete-time Polynomial Systems, Morgan et al., 2019'''


def system_dynamics(x):
    y_1 = x[0] + 0.2*(x[0] - x[0]**3/3 - x[1] + 0.875)
    y_2 = x[1] + 0.2*(0.08*(x[0] + 0.7 - 0.8*x[1]))
    y = np.array([y_1, y_2])
    return y


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
    '''Use fast linear bounds (LiRPA) to initialize all the pre-activation bounds and call ADMM or Gurobi to solve the 
        LP to find the output bounds. '''


    is_train = False
    nn_width = 400

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    if is_train:
        '''train a neural network to approximate the given dynamics'''

        # domain of the system
        x_min = np.array([-5.0, -5.0])
        x_max = np.array([5.0, 5.0])
        domain = Polyhedron.from_bounds(x_min, x_max)

        # uniformly sample from the domain to generate training data
        resol = 50
        x_train_samples, y_train_samples = ut.generate_training_data(system_dynamics, domain, resol, None)

        sample_data = {'train_data': x_train_samples, 'label_data': y_train_samples}
        ut.pickle_file(sample_data, 'training_data_set')

        train_data_set = SystemDataSet(x_train_samples, y_train_samples)

        train_batch_size = 10
        train_loader = DataLoader(train_data_set, batch_size=train_batch_size, shuffle=True)

        nn_structure = nn_dynamics(2, nn_width)
        nn_model = train_nn_torch(train_loader, nn_structure, num_epochs=100, l1=None,
                                  pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path='torch_nn_model')

        torch.save(nn_model, nn_file_name)


    '''
    finite step reachability analysis
    '''
    torch.set_grad_enabled(False)

    nn_system = torch.load(nn_file_name)

    nx = 2
    x0 = torch.tensor([[0.2,0.2]])

    epsilon = 0.02
    x0_lb = x0 - epsilon
    x0_ub = x0 + epsilon

    horizon = 50
    seq_model = SequentialModel(nn_system, horizon)

    print('nn model verification with horiozn {}'.format(horizon))

    load_pre_act_bounds = 0
    if load_pre_act_bounds == 1:
        pre_act_bds = torch.load('pre_act_bds_from_LiRPA.pt')
    else:
        method = 'backward'
        print('generating intermediate bounds from LiRPA with method {}'.format(method))
        start_time = time.time()
        pre_act_bds, num_act_layers, _ = preactivation_bounds_of_sequential_nn_model_LiRPA(nn_system, horizon, x0_lb, x0_ub, method = method)
        print('LiRPA find bounds in {}s'.format(time.time() - start_time))
        torch.save(pre_act_bds, 'pre_act_bds_from_LiRPA.pt')

    # LP
    print('find output bounds by Gurobi')
    nx = 2

    A_input = np.vstack((np.eye(nx), -np.eye(nx)))

    input_lb = x0_lb.to(torch.device('cpu')).numpy()
    input_ub = x0_ub.to(torch.device('cpu')).numpy()

    b_input = np.concatenate((input_ub, -input_lb)).flatten()
    c_output = ut.unif_normal_vecs(nx, n=4)

    pre_act_bds_numpy = pre_act_bds_tensor_to_numpy(pre_act_bds)
    output_bds, diags = seq_model.pre_activation_bounds_LP(pre_act_bds_numpy,seq_model.num_activation_layers+1, A_input, b_input, c_output)
    LP_output_set = Polyhedron(output_bds['A'], output_bds['b'])

    print('LP solver Gurobi finished. Runtime: {}. Output bounds: {}'.format(diags['total_solver_time'], output_bds['b']))

    # LiRPA bounds
    print('Get LiRPA output bounds')
    lirpa_output_lb, lirpa_output_ub = output_Lp_bounds_LiRPA(seq_model, x0_lb, x0_ub, method='backward')
    lirpa_output_set = Polyhedron.from_bounds(lirpa_output_lb[0].numpy(), lirpa_output_ub[0].numpy())

    # ADMM
    print('ADMM start')
    alg_options = {'rho': 0.1, 'eps_abs': 1e-5, 'eps_rel': 1e-4, 'residual_balancing': False, 'max_iter': 20000,
                   'record': False, 'verbose': True, 'alpha': 1.6}

    nn_system.to(device)
    base_nn_model_list = list(nn_system)
    nn_layers_list = base_nn_model_list * horizon

    c_output = ut.unif_normal_vecs(nx, n=4)
    c_output = torch.from_numpy(c_output).type(torch.float32)
    c_output = -c_output.to(device)

    rho = alg_options['rho']

    num_batches = c_output.size(0)
    x_input = x0.repeat(num_batches, 1).to(device)
    lb_input = x0_lb.repeat(num_batches, 1).to(device)
    ub_input = x0_ub.repeat(num_batches, 1).to(device)

    init_module = InitModule(nn_layers_list, x_input, lb_input, ub_input, pre_act_bds_list=None)
    admm_module = init_module.init_ADMM_module()
    admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c_output, rho)

    pre_act_bds = [{'lb': item['lb'].repeat(num_batches, 1), 'ub': item['ub'].repeat(num_batches, 1)} for item in
                   pre_act_bds]
    admm_sess.assign_pre_activation_bounds(pre_act_bds)

    objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)
    print('ADMM finished. Runtime:{}. Output bounds: {}'.format(running_time, objective))
    print('LP solver Gurobi finished. Runtime: {}. Output bounds: {}'.format(diags['total_solver_time'], output_bds['b']))

    # full LP result
    # LP_result = torch.load('LP_result_width_100_horizon_5_radius_1.0.pt')
    # full_LP_output_set = Polyhedron(LP_result['output_bds']['A'], LP_result['output_bds']['b'])

    plt.figure()
    domain = Polyhedron.from_bounds(x0_lb[0].detach().numpy(), x0_ub[0].detach().numpy())

    init_states = ut.unif_sample_from_Polyhedron(domain, 8)
    nn_system.to(torch.device('cpu'))
    traj_list = ut.simulate_NN_system(nn_system, init_states, step=10)
    ut.plot_multiple_traj_tensor_to_numpy(traj_list)

    plt.figure()
    LP_output_set.plot(fill=False, ec='r', linestyle='-', linewidth=2)
    lirpa_output_set.plot(fill=False, ec='b', linestyle='-.', linewidth=2)
    plt.show()
    # full_LP_output_set.plot(fill = False, ec = 'k', linestyle = '-.', linewidth = 2)