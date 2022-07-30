import numpy as np

from pympc.geometry.polyhedron import Polyhedron
from pympc.plot import plot_state_space_trajectory

from pympc.optimization.programs import linear_program
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pickle

'''
generate training data 
'''

def generate_training_data(dyn_fcn, X, N_dim, file_name = None):
    # if filename is not None, load the existing file
    if file_name is not None:
        data = load_pickle_file(file_name)
        input_samples = data['train_data']
        labels = data['label_data']
    else:
        input_samples = unif_sample_from_Polyhedron(X, N_dim)
        labels = sample_vector_field(dyn_fcn, input_samples)
    return input_samples, labels


def sample_vector_field(dyn_fcn, samples):
    num_samples, nx = samples.shape

    sample = samples[0]
    output = dyn_fcn(sample)
    ny = output.shape[0]

    labels = np.zeros((1,ny))
    for i in tqdm(range(num_samples), desc = 'sample_vector_filed'):
    # for i in range(num_samples):
        x_input = samples[i]
        y = dyn_fcn(x_input)
        labels = np.vstack((labels, y))

    labels = labels[1:,:]
    return labels


def unif_sample_from_Polyhedron(X, N_dim, epsilon=None, residual_dim=None):
    # uniformly sample from the Polyhedron X with N_dim grid points on each dimension
    nx = X.A.shape[1]
    if residual_dim is not None:
        X = X.project_to(residual_dim)
    lb, ub = find_bounding_box(X)
    box_grid_samples = grid_sample_from_box(lb, ub, N_dim, epsilon)
    idx_set = [X.contains(box_grid_samples[i, :]) for i in range(box_grid_samples.shape[0])]
    valid_samples = box_grid_samples[idx_set]

    if residual_dim is not None:
        aux_samples = np.zeros((valid_samples.shape[0], 1))
        for i in range(nx):
            if i in residual_dim:
                aux_samples = np.hstack((aux_samples, valid_samples[:, i].reshape(-1, 1)))
            else:
                aux_samples = np.hstack((aux_samples, np.zeros((valid_samples.shape[0], 1))))

        aux_samples = aux_samples[:, 1:]
        return aux_samples

    return valid_samples


def find_bounding_box(X):
    # find the smallest box that contains Polyhedron X
    A = X.A
    b = X.b

    nx = A.shape[1]

    lb_sol = [linear_program(np.eye(nx)[i], A, b) for i in range(nx)]
    lb_val = [lb_sol[i]['min'] for i in range(nx)]

    ub_sol = [linear_program(-np.eye(nx)[i], A, b) for i in range(nx)]
    ub_val = [-ub_sol[i]['min'] for i in range(nx)]

    return lb_val, ub_val


def grid_sample_from_box(lb, ub, Ndim, epsilon=None):
    # generate uniform grid samples from a box {lb <= x <= ub} with Ndim samples on each dimension
    nx = len(lb)
    assert nx == len(ub)

    if epsilon is not None:
        lb = [lb[i] + epsilon for i in range(nx)]
        ub = [ub[i] - epsilon for i in range(nx)]

    grid_samples = grid_sample(lb, ub, Ndim, nx)
    return grid_samples


def grid_sample(lb, ub, Ndim, idx):
    # generate samples using recursion
    nx = len(lb)
    cur_idx = nx - idx
    lb_val = lb[cur_idx]
    ub_val = ub[cur_idx]

    if idx == 1:
        cur_samples = np.linspace(lb_val, ub_val, Ndim)
        return cur_samples.reshape(-1, 1)

    samples = grid_sample(lb, ub, Ndim, idx - 1)
    n_samples = samples.shape[0]
    extended_samples = np.tile(samples, (Ndim, 1))

    cur_samples = np.linspace(lb_val, ub_val, Ndim)
    new_column = np.kron(cur_samples.reshape(-1, 1), np.ones((n_samples, 1)))

    new_samples = np.hstack((new_column, extended_samples))
    return new_samples

def load_pickle_file(file_name):
    with open(file_name, 'rb') as config_dictionary_file:
        data = pickle.load(config_dictionary_file)
    return data

def pickle_file(data, file_name):
    with open(file_name, 'wb') as config_dictionary_file:
          pickle.dump(data, config_dictionary_file)

def generate_training_data_traj(dyn_fcn, X, N_dim, step = 0):
    init_states = unif_sample_from_Polyhedron(X, N_dim)
    x_samples, y_samples = generate_traj_samples(dyn_fcn, init_states, step)
    return x_samples, y_samples


def generate_traj_samples(dyn_fcn, init_states, step = 0):
    N = init_states.shape[0]
    nx = init_states.shape[1]

    traj_list = []
    for i in range(N):
        x = init_states[i]
        traj = x
        for j in range(step+1):
            x_next = dyn_fcn(x)
            traj = np.vstack((traj, x_next))
            x = x_next
        traj_list.append(traj)
    x_samples = [traj[0:1, :] for traj in traj_list]
    y_samples = [traj[1:, :] for traj in traj_list]

    x_samples = np.stack(x_samples, axis = 0)
    y_samples = np.stack(y_samples, axis = 0)

    return x_samples, y_samples


def plot_multiple_traj(x_traj_list, **kwargs):
    num_traj = len(x_traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(x_traj_list[i], **kwargs)


'''
simulate closed-loop system with NN components
NN structure and data are from pytorch
'''

def simulate_NN_system(nn_model, init_states, step = 0):
    # init_states is a list of initial conditions in numpy

    N = init_states.shape[0]
    nx = init_states.shape[1]

    traj_list = []
    for i in range(N):
        x = torch.from_numpy(init_states[i])
        x = x.type(nn_model[0].weight.dtype)

        traj = x.unsqueeze(0)
        for j in range(step+1):
            x_next = nn_model(x)
            traj = torch.cat((traj, x_next.unsqueeze(0)))
            x = x_next
        traj_list.append(traj)
    return traj_list



def plot_multiple_traj_tensor_to_numpy(traj_list, **kwargs):
    num_traj = len(traj_list)
    for i in range(num_traj):
        plot_state_space_trajectory(traj_list[i].detach().numpy(), **kwargs)


def bounds_list_to_polyhedron_list(bounds_list):
    N = len(bounds_list)
    poly_list = []

    if 'A' in bounds_list[0].keys():
        for i in range(N):
            X = Polyhedron(bounds_list[i]['A'], bounds_list[i]['b'])
            poly_list.append(X)
    elif 'lb' in bounds_list[0].keys():
        for i in range(N):
            X = Polyhedron.from_bounds(bounds_list[i]['lb'], bounds_list[i]['ub'])
            poly_list.append(X)

    return poly_list

def plot_poly_list(poly_list, **kwargs):
    N = len(poly_list)
    for i in range(N):
        poly_list[i].plot(**kwargs)

def unif_normal_vecs(nx, n = 4):
    # fixme: we only consider the case nx = 2
    assert nx == 2
    theta = 2*np.pi/n
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    c = np.zeros((1,nx))
    c[0][0] = 1.0

    vec = c.flatten()
    for i in range(n-1):
        vec = rotation_mat@vec
        c = np.vstack((c, vec.reshape(1,-1)))
    return c


def compare_layerwise_bounds(pre_act_bds_list_iter, pre_act_bds_list_seq):
    lb_list_iter = []
    ub_list_iter = []
    lb_list_seq = []
    ub_list_seq = []
    for i in range(len(pre_act_bds_list_seq)):
        lb_list_iter = lb_list_iter + pre_act_bds_list_iter[i]['lb'].tolist()
        ub_list_iter = ub_list_iter + pre_act_bds_list_iter[i]['ub'].tolist()
        lb_list_seq = lb_list_seq + pre_act_bds_list_seq[i]['lb'].tolist()
        ub_list_seq = ub_list_seq + pre_act_bds_list_seq[i]['ub'].tolist()

    lb_index = sorted(range(len(lb_list_seq)), key=lambda k: lb_list_seq[k])
    ub_index = sorted(range(len(ub_list_seq)), key=lambda k: ub_list_seq[k])

    lbs_iter_sorted = [lb_list_iter[k] for k in lb_index]
    ubs_iter_sorted = [ub_list_iter[k] for k in ub_index]
    lbs_seq_sorted = [lb_list_seq[k] for k in lb_index]
    ubs_seq_sorted = [ub_list_seq[k] for k in ub_index]

    plt.figure()
    plt.plot(lbs_iter_sorted, 'b-.', label='lbs iter')
    plt.plot(ubs_iter_sorted, 'b-', label='ubs iter')
    plt.plot(lbs_seq_sorted, 'r-.', label='lbs seq')
    plt.plot(ubs_seq_sorted, 'r-', label='ubs seq')
    plt.legend()
    plt.ylabel(r'pre-activation bounds')
    # truncation = 50
    # plt.figure()
    # plt.semilogy(lbs_iter_sorted[-truncation:], 'b-.', label='lbs iter')
    # plt.semilogy(lbs_seq_sorted[-truncation:], 'r-.', label='lbs seq')
    #
    # plt.figure()
    # plt.semilogy(ubs_iter_sorted[-truncation:], 'b-', label='ubs iter')
    # plt.semilogy(ubs_seq_sorted[-truncation:], 'r-', label='ubs seq')

# random uniform sample from a box
def random_unif_sample_from_box(bounds_list, N):
    # box_list = [[min, max], [min, max], ...]
    box_list = [[item[0], item[1] - item[0]] for item in bounds_list]
    nx = len(box_list)
    rand_matrix = np.random.rand(N, nx)
    samples = np.vstack([rand_matrix[:, i] * box_list[i][1] + box_list[i][0] for i in range(nx)])
    samples = samples.T
    return samples