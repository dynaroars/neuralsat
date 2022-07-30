
import torch
import torch.nn as nn
import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut
import numpy as np
from pympc.geometry.polyhedron import Polyhedron

import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel
from nn_reachability.nn_models import iterative_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA

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
    '''Compare the one-shot (sequential) and the recursive (iterative) methods with fast linear bounds (LiRPA) as propagators. Generates Fig.3.'''

    nx = 4

    x_min = np.array([-5.0, -5.0, -30*np.pi/180, -1.0])
    x_max = np.array([5.0, 5.0, 30*np.pi/180, 1.0])
    domain = Polyhedron.from_bounds(x_min, x_max)

    # train a neural network
    is_train = False
    nn_width = 100

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)

    x0_lb = torch.tensor([[2.0, 1.0, -20*np.pi/180, -1.0]]).to(torch.device('cpu'))
    x0_ub = torch.tensor([[2.2, 1.2, -10*np.pi/180, -0.6]]).to(torch.device('cpu'))

    horizon = 5

    # reachability analysis through sequential nn (one-shot method)
    print('Sequential reachability analysis \n')

    # different modes of LiRPA can be used: forward, backward, IBP, forward+backward, IBP+backward
    method = 'backward'
    box = {'lb': x0_lb[0].numpy(), 'ub': x0_ub[0].numpy()}
    seq_bounds_list = [box]
    for i in range(horizon):
        nn_model = SequentialModel(nn_system, i + 1)
        output_lb, output_ub = output_Lp_bounds_LiRPA(nn_model, x0_lb, x0_ub, method=method)
        box = {'lb': output_lb[0].detach().numpy(), 'ub': output_ub[0].detach().numpy()}
        seq_bounds_list.append(box)

    # reachability analysis through iterative methods
    print('Iterative reachability analysis \n')
    base_nn_system = SequentialModel(nn_system, 1)
    iter_bounds_list = iterative_output_Lp_bounds_LiRPA(base_nn_system, x0_lb, x0_ub, horizon, method=method)
    iter_bounds_list = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in
                        iter_bounds_list]

    # add IBP baseline
    # base_nn_system = SequentialModel(nn_system, 1)
    # iter_bounds_IBP = iterative_output_Lp_bounds_LiRPA(base_nn_system, x0_lb, x0_ub, horizon, method='IBP')
    # iter_bounds_IBP = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in
    #                    iter_bounds_IBP]
    # iter_poly_IBP = ut.bounds_list_to_polyhedron_list(iter_bounds_IBP)

    # construct polyhedra from box lower and upper bounds
    seq_poly_list = ut.bounds_list_to_polyhedron_list(seq_bounds_list)
    iter_poly_list = ut.bounds_list_to_polyhedron_list(iter_bounds_list)

    for i in range(horizon):
        plt.figure()
        seq_poly_list[i + 1].plot(fill=False, ec='r', linestyle='-', linewidth=2)
        iter_poly_list[i + 1].plot(fill=False, ec='b', linestyle='-.', linewidth=2)

        # add IBP baseline
        # iter_poly_IBP[i + 1].plot(fill=False, ec='g', linestyle='-.', linewidth=2)

        input_set = Polyhedron.from_bounds(x0_lb[0].numpy(), x0_ub[0].numpy())
        init_states = ut.unif_sample_from_Polyhedron(input_set, 3)
        traj_list = ut.simulate_NN_system(nn_system, init_states, step=i)

        ut.plot_multiple_traj_tensor_to_numpy(traj_list)
        input_set.plot(fill=False, ec='k', linestyle='--', linewidth=2)
        plt.title('Reachable set at step {}'.format(i + 1))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    plt.show()

