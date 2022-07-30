
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

from pympc.geometry.polyhedron import Polyhedron
import nn_reachability.utilities as ut
import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut


import sys
sys.path.append(r'D:\Shaoru\GithubDesk\auto_LiRPA')
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from tqdm import tqdm
from nn_reachability.ADMM import intermediate_bounds_from_ADMM, InitModule, ADMM_Session, run_ADMM

class SequentialModel(nn.Module):
    def __init__(self, nn_model, N):
        super(SequentialModel, self).__init__()
        self.base_model = nn_model
        self.horizon = N
        num_hidden_layers = find_number_of_hidden_layers(nn_model)
        self.num_activation_layers = num_hidden_layers*N
        self.base_num_activation_layers = num_hidden_layers

        self.layer_list = list(nn_model)*N
        self.pre_act_bounds_list = []

    def reset_horizon(self, N):
        self.horizon = N
        self.num_activation_layers = self.base_num_activation_layers*N
        self.layer_list = list(self.base_model) * N

    def forward(self, x):
        for i in range(self.horizon):
            x = self.base_model(x)
        return x

    def pre_activation_bounds_LP(self, bounds_list, layer_num, A, b, c = None):
        # find the preactivation bounds for the layer_num-th activation layer
        # layer_num index from 1
        assert len(bounds_list) >= layer_num - 1

        x = {}
        y = {}

        x0 = torch.zeros(A.shape[1])

        act_layer_count = 0
        layer_count = 0
        for layer in self.layer_list:
            if isinstance(layer, nn.ReLU):
                act_layer_count += 1
                if act_layer_count >= layer_num:
                    break
            dim_input = x0.shape[0]
            x0 = layer(x0)
            dim_output = x0.shape[0]

            x[layer_count] = cp.Variable(dim_input)
            y[layer_count] = cp.Variable(dim_output)
            layer_count += 1

        constr = [A@x[0] <= b]

        act_layer_count = 0
        layer_count = 0
        for layer in self.layer_list:
            if isinstance(layer, nn.ReLU):
                act_layer_count += 1
                # detect termination
                if act_layer_count >= layer_num:
                    break

                bound = bounds_list[act_layer_count-1]

                lb = bound['lb']
                ub = bound['ub']
                constr += [y[layer_count] >= x[layer_count]]
                constr += [y[layer_count] >= 0]
                constr += [y[layer_count][k] == x[layer_count][k] for k in range(x[layer_count].shape[0]) if lb[k] >= 0]
                constr += [y[layer_count][k] == 0 for k in range(x[layer_count].shape[0]) if ub[k] < 0]
                constr += [y[layer_count][k] <= ub[k]/(ub[k]-lb[k])*(x[layer_count][k] - lb[k]) for k in range(x[layer_count].shape[0])
                           if (ub[k] >0 and lb[k] < 0)]

            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy()
                constr += [ y[layer_count] == weight @ x[layer_count] + bias ]

            if layer_count > 0:
                constr += [ x[layer_count] == y[layer_count-1] ]

            layer_count += 1

        dim_output = y[layer_count-1].shape[0]
        c_vec = cp.Parameter(dim_output)
        obj = c_vec @ y[layer_count - 1]
        prob = cp.Problem(cp.Minimize(obj), constr)

        print('CVXPY model solving ...')
        total_solver_time = 0
        running_start = time.time()
        if c is None:
            id_mat = np.eye(dim_output)

            lb_vec = np.zeros(dim_output)
            for i in tqdm(range(dim_output), desc='output_lb'):
            # for i in range(dim_output):
                obj_vec = id_mat[i]
                c_vec.value = obj_vec
                prob.solve(solver=cp.GUROBI, verbose=False)
                total_solver_time += prob.solver_stats.solve_time
                lb_vec[i] = obj.value

            ub_vec = np.zeros(dim_output)
            for i in tqdm(range(dim_output), desc='output_ub'):
            # for i in range(dim_output):
                obj_vec = -id_mat[i]
                c_vec.value = obj_vec
                prob.solve(solver=cp.GUROBI, verbose=False)
                total_solver_time += prob.solver_stats.solve_time
                ub_vec[i] = -obj.value
            running_time = time.time() - running_start

            output_bd = {'lb': lb_vec, 'ub': ub_vec}
            diags = {'total_solver_time': total_solver_time, 'dim_output': dim_output,
                     'running_time': running_time}
            return output_bd, diags
        else:
            # compute the polytopic overapproximation given by c
            num_output_constr = c.shape[0]
            output_vec = np.zeros(num_output_constr)
            for i in range(num_output_constr):
                # note that we flip the sign of objective vector to make it a maximization problem
                c_vec.value = -c[i]
                prob.solve(solver=cp.GUROBI, verbose=False)
                output_vec[i] = -obj.value
                solver_time = prob.solver_stats.solve_time
                total_solver_time += solver_time

            output_bd = {'A': c, 'b': output_vec}
            running_time = time.time() - running_start
            diags = {'total_solver_time': total_solver_time, 'dim_output': dim_output,
                 'running_time': running_time}
            return output_bd, diags

    def output_inf_bounds_LP(self, A, b, c = None, file_name = None):
        # input set: Ax <= b
        # output set: cx <= y

        if file_name is None:
            file_name = 'layerwise_LP_bds'

        total_num_activation_layers = self.num_activation_layers

        diags_list = []
        time_sum = 0
        solver_time_list = []

        if self.horizon > 1:
            # reuse pre-activation bounds computed from previous steps if horizon > 1
            pre_act_bds = self.pre_act_bounds_list
        else:
            pre_act_bds = []

        num_existing_layers = len(pre_act_bds)
        for i in range(num_existing_layers, total_num_activation_layers):
            print('activation layer number {}'.format(i))
            layer_bd, diags = self.pre_activation_bounds_LP(pre_act_bds, i+1, A, b)
            pre_act_bds.append(layer_bd)
            diags_list.append(diags)
            time_sum += diags['total_solver_time']
            solver_time_list.append(diags['total_solver_time'])

        self.pre_act_bounds_list = pre_act_bds

        bounds_list = copy.copy(pre_act_bds)
        if c is None:
            print('output layer')
            layer_bd, diags = self.pre_activation_bounds_LP(bounds_list, total_num_activation_layers + 1, A, b)
            bounds_list.append(layer_bd)
            diags_list.append(diags)
            time_sum += diags['total_solver_time']
            solver_time_list.append(diags['total_solver_time'])
        else:
            print('output layer')
            layer_bd, diags = self.pre_activation_bounds_LP(bounds_list, total_num_activation_layers + 1, A, b, c)
            bounds_list.append(layer_bd)
            diags_list.append(diags)
            time_sum += diags['total_solver_time']
            solver_time_list.append(diags['total_solver_time'])

        data_to_save = {'pre_act_bds': bounds_list, 'diags': diags_list, 'solver_time': time_sum}
        torch.save(data_to_save, file_name + '.pt')
        return bounds_list, solver_time_list

# sequential and iterative LP-based
def Gurobi_reachable_set(nn_system, x0_lb, x0_ub, horizon, method = 'sequential'):
    nx = nn_system.nx

    # convert input set to Ax  <= b
    A_input = np.vstack((np.eye(nx), -np.eye(nx)))

    input_lb = x0_lb.to(torch.device('cpu')).numpy()
    input_ub = x0_ub.to(torch.device('cpu')).numpy()

    b_input = np.concatenate((input_ub, -input_lb)).flatten()

    if method == 'sequential':
        print('One-shot reachability analysis \n')
        output_bds_list_seq = []
        solver_time_seq_list = []
        seq_nn_system = SequentialModel(nn_system, 1)
        for i in range(horizon):
            print('step {} \n'.format(i + 1))
            seq_nn_system.reset_horizon(i + 1)
            bounds_list, solver_time_seq = seq_nn_system.output_inf_bounds_LP(A_input, b_input, None, file_name=None)
            output_bds = bounds_list[-1]
            output_bds_list_seq.append(output_bds)
            solver_time_seq_list.append(solver_time_seq)

        pre_act_bds_list_seq = bounds_list[:-1]
        result = {'output_bds': output_bds_list_seq, 'pre_act_bds': pre_act_bds_list_seq, 'solver_time': solver_time_seq_list, 'method':method}

    elif method == 'iterative':
        print('Recursive reachability analysis \n')
        output_bds_list_iter = []
        solver_time_iter_list = []
        A_input_iter, b_input_iter = A_input, b_input
        pre_act_bds_list_iter = []
        for i in range(horizon):
            base_nn_system = SequentialModel(nn_system, 1)
            bounds_list, solver_time_iter = base_nn_system.output_inf_bounds_LP(A_input_iter, b_input_iter, None,
                                                                                file_name=None)
            pre_act_bds_list_iter = pre_act_bds_list_iter + bounds_list[:-1]
            output_bds = bounds_list[-1]
            output_bds_list_iter.append(output_bds)
            output_lb, output_ub = output_bds['lb'], output_bds['ub']
            output_box = Polyhedron.from_bounds(output_lb, output_ub)
            A_input_iter, b_input_iter = output_box.A, output_box.b

            solver_time_iter_list.append(solver_time_iter)
        result = {'output_bds': output_bds_list_iter, 'pre_act_bds': pre_act_bds_list_iter,
                  'solver_time': solver_time_iter_list, 'method': method}
    else:
        raise NotImplementedError
    return output_bds, result

def ADMM_reachable_set(nn_system, x0_lb, x0_ub, horizon, alg_options = None, file_name = None, load_file = 0):
    # implement one-shot method for ADMM reachability analysis

    # admm parameters
    if alg_options is None:
        alg_options = {'rho': 0.1, 'eps_abs': 1e-4, 'eps_rel': 1e-3, 'residual_balancing': False, 'max_iter': 20000,
                       'record': False, 'verbose': True, 'alpha': 1.6}
    nx = x0_lb.size(1)

    device = x0_lb.device

    nn_system.to(device)
    base_nn_model_list = list(nn_system)
    nn_layers_list = base_nn_model_list * horizon

    ADMM_suffix = '_horizon_' + str(horizon) + '_eps_abs_' + str(alg_options['eps_abs']) + '.pt'

    pre_act_bds_file_name = 'ADMM_intermediate_bounds' + ADMM_suffix

    if load_file == 1:
        data = torch.load(pre_act_bds_file_name)
        pre_act_bds_admm = data['pre_act_bds']
        pre_act_bds_runtime = data['runtime']
    else:
        pre_act_bds_admm, pre_act_bds_runtime = intermediate_bounds_from_ADMM(nn_layers_list, x0_lb, x0_ub, alg_options, pre_act_bds_file_name)
        print('ADMM pre-act. bds runtime: {}'.format(pre_act_bds_runtime))
        temp_result = {'pre_adt_bds_admm': pre_act_bds_admm, 'runtime': pre_act_bds_runtime, 'alg_options': alg_options}
        torch.save(temp_result, 'ADMM_pre_act_bounds_result' + ADMM_suffix)

    # find the reachable sets based on the pre-activation bounds
    c_output = torch.cat((torch.eye(nx), -torch.eye(nx))).to(device)

    rho = alg_options['rho']

    num_batches = c_output.size(0)
    x_input = x0_lb.repeat(num_batches, 1).to(device)
    lb_input = x0_lb.repeat(num_batches, 1).to(device)
    ub_input = x0_ub.repeat(num_batches, 1).to(device)

    num_hidden_layers = find_number_of_hidden_layers(nn_system)
    output_bds_list = []
    runtime_output_list = []

    for i in tqdm(range(horizon), desc = 'admm_output'):
        truncated_nn_layers = nn_layers_list[:(num_hidden_layers*2+1)*(i+1)]

        init_module = InitModule(truncated_nn_layers, x_input, lb_input, ub_input, pre_act_bds_list=None)
        admm_module = init_module.init_ADMM_module()
        admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c_output, rho)

        pre_act_bds_truncation = pre_act_bds_admm[:num_hidden_layers*(i+1)]
        pre_act_bds = [{'lb': item['lb'].repeat(num_batches, 1), 'ub': item['ub'].repeat(num_batches, 1)} for item in
                       pre_act_bds_truncation]
        admm_sess.assign_pre_activation_bounds(pre_act_bds)

        objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)
        output_bds = {'lb': objective[:nx].to(torch.device('cpu')).numpy(), 'ub': -objective[nx:].to(torch.device('cpu')).numpy()}
        output_bds_list.append(output_bds)
        runtime_output_list.append(running_time)

    ADMM_result = {'pre_act_bds': pre_act_bds_admm, 'runtime_pre_act_bds': pre_act_bds_runtime, 'alg_options': alg_options,
                   'output_bds': output_bds_list, 'runtime_output_bds': runtime_output_list,
                   'x0_lb': x0_lb.to(torch.device('cpu')), 'x0_ub': x0_ub.to(torch.device('cpu')),
                   'horizon': horizon
                   }

    if file_name is None:
        file_name = 'ADMM_one_shot_result' + ADMM_suffix
    torch.save(ADMM_result, file_name)

    return output_bds_list, ADMM_result


# LiRPA based analysis
def output_Lp_bounds_LiRPA(nn_model, lb, ub, method = 'backward'):

    center = (lb + ub)/2
    radius = (ub - lb)/2

    model = BoundedModule(nn_model, center)
    ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(center, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds
    output_lb, output_ub = model.compute_bounds(x=(my_input,), method=method)
    return output_lb, output_ub


def iterative_output_Lp_bounds_LiRPA(nn_model, lb_0, ub_0, horizon, method = 'backward'):
    input_lb, input_ub = lb_0, ub_0
    box = {'lb': lb_0, 'ub': ub_0}
    bounds_list =[box]
    ori_model = nn_model
    for i in range(horizon):
        output_lb, output_ub = output_Lp_bounds_LiRPA(ori_model, input_lb, input_ub, method = method)
        box = {'lb': output_lb, 'ub': output_ub}
        bounds_list.append(box)
        input_lb, input_ub = output_lb.detach(), output_ub.detach()

    return bounds_list

def preactivation_bounds_of_sequential_nn_model_LiRPA(nn_model, horizon, lb_0, ub_0, method = 'backward'):
    nn_layer_list = list(nn_model)*horizon
    num_act_layers = find_number_of_hidden_layers(nn_model)
    pre_act_bds = preactivation_bounds_from_LiRPA(nn_layer_list, lb_0, ub_0, method)
    return pre_act_bds, num_act_layers, horizon

def preactivation_bounds_from_LiRPA(layer_list, lb_0, ub_0, method = 'backward'):
    # find the index of activation layers
    ind_list = []
    for i in range(len(layer_list)):
        if isinstance(layer_list[i], nn.ReLU):
            ind_list.append(i)

    pre_act_bds = []
    for index in ind_list:
        # fixme: should we check if the first layer is an activation layer?
        net = nn.Sequential(*layer_list[:index])
        output_lb, output_ub = output_Lp_bounds_LiRPA(net, lb_0, ub_0, method = method)
        bds = {'lb': output_lb, 'ub': output_ub}
        pre_act_bds.append(bds)

    return pre_act_bds

def pre_act_bds_tensor_to_numpy(pre_act_bds):
    # fixme: currently assume the tensors are of size [1, n] in pre_act_bds
    pre_act_bds_numpy = [{'lb':item['lb'][0].numpy(), 'ub': item['ub'][0].numpy()} for item in pre_act_bds]
    return pre_act_bds_numpy



def repeat_nn_model(nn_model, N):
    # repeat nn_model N times and concatenate them in one nn model
    nn_model_layers = list(nn_model)*N
    seq_nn_model = nn.Sequential(*(nn_model_layers))
    return seq_nn_model

def extract_linear_layers(nn_model):
    '''extract the linear layers of a FC NN'''
    linear_layer_list = []
    for i, layer in enumerate(nn_model):
        if isinstance(layer, nn.Linear):
            linear_layer_list.append(layer)
    return linear_layer_list


def find_number_of_hidden_layers(nn_model):
    '''find the number of hidden layers in a FC NN'''
    H = 0
    for i, layer in enumerate(nn_model):
        if isinstance(layer, nn.ReLU):
            H += 1
    return H

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

'''
neural network training
'''
class torch_nn_model(nn.Module):
    def __init__(self, nn_dims):
        super(torch_nn_model, self).__init__()
        self.dims = nn_dims
        self.L = len(nn_dims) - 2
        self.linears = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i+1]) for i in range(len(nn_dims)-1)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i in range(self.L):
            x = F.relu(self.linears[i](x))

        x = self.linears[-1](x)

        return x

# custom data set
class SystemDataSet(Dataset):
    def __init__(self, x_samples, y_samples):
        x_samples = torch.from_numpy(x_samples)
        y_samples = torch.from_numpy(y_samples)
        nx = x_samples.size(-1)
        ny = y_samples.size(-1)
        self.nx = nx
        self.ny = ny

        # we sample trajectories
        x_samples, y_samples = x_samples.type(torch.float32), y_samples.type(torch.float32)
        x_samples = x_samples.unsqueeze(1)
        y_samples = y_samples.unsqueeze(1)

        self.x_samples = x_samples
        self.y_samples = y_samples

    def __len__(self):
        return len(self.x_samples)

    def __getitem__(self, index):
        target = self.y_samples[index]
        data_val = self.x_samples[index]
        return data_val, target


def criterion(pred_traj, label_traj):
    batch_size = pred_traj.size(0)
    step = pred_traj.size(1)
    label_step = label_traj.size(1)
    if step > label_step:
        warnings.warn('prediction step mismatch')

    slice_step = min(step, label_step)

    label_traj_slice = label_traj[:, :slice_step, :]
    pred_traj_slice = pred_traj[:, :slice_step, :]

    # label_traj_slice_norm = torch.unsqueeze(torch.linalg.norm(label_traj_slice, 2, dim = 2), 2)
    # label_traj_slice = label_traj_slice/label_traj_slice_norm
    # pred_traj_slice = pred_traj_slice/label_traj_slice_norm
    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(batch_size*step)

    # err = 0.0
    # for i in range(batch_size):
    #     err += torch.linalg.norm(label_traj_slice[i].reshape(-1) - pred_traj_slice[i].reshape(-1), np.inf)/(torch.linalg.norm(label_traj_slice[i].reshape(-1), np.inf) + 1e-4)
    #
    # err = err/batch_size
    #
    #
    err = torch.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)/(batch_size*slice_step)

    # err = torch.linalg.norm(label_traj_slice.reshape(-1) - pred_traj_slice.reshape(-1), 2)**2/(pred_traj_slice.reshape(-1).size(0))

    return err

def torch_train_nn(nn_model, dataloader, l1 = None, epochs = 30, step = 5, lr = 1e-4, decay_rate = 1.0, clr = None):

    if clr is None:
        optimizer = optim.Adam(nn_model.parameters(), lr= lr)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay_rate, last_epoch=-1)
        lr_scheduler = lambda t: lr
        cycle = 1
        update_rate = 1
    else:
        lr_base = clr['lr_base']
        lr_max = clr['lr_max']
        step_size = clr['step_size']
        cycle = clr['cycle']
        update_rate = clr['update_rate']
        optimizer = optim.Adam(nn_model.parameters(), lr= lr_max)
        lr_scheduler = lambda t: np.interp([t], [0, step_size, cycle], [lr_base, lr_max, lr_base])[0]

    lr_test = {}
    cycle_loss = 0.0
    cycle_count = 0

    nn_model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        lr = lr_scheduler((epoch//update_rate)%cycle) # change learning rate every two epochs
        optimizer.param_groups[0].update(lr=lr)

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            batch_size = inputs.size(0)
            x = inputs
            y = nn_model(x)
            traj = y
            for _ in range(step):
                x = y
                y = nn_model(x)
                traj = torch.cat((traj, y), 1)

            loss_1 = criterion(traj, labels)

            # add l1 regularization
            if l1 is not None:
                l1_regularization = 0.0
                for param in nn_model.parameters():
                    '''attention: what's the correct l1 regularization'''
                    l1_regularization += torch.linalg.norm(param.view(-1), 1)
                loss = loss_1 + l1*l1_regularization
            else:
                loss = loss_1

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss_1.item()

            cycle_loss += loss_1.item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        if (epoch + 1) % update_rate == 0:
            lr_test[cycle_count] = cycle_loss / update_rate / len(dataloader)
            print('\n [%d, %.4f] cycle loss: %.6f' % (cycle_count, lr, lr_test[cycle_count]))

            cycle_count += 1
            cycle_loss = 0.0

        # scheduler.step()
    ut.pickle_file(lr_test, 'lr_test_temp')

    print('finished training')
    save_torch_nn_model(nn_model, 'torch_nn_model_dict_temp')
    return nn_model


def train_nn_torch(dataloader, nn_structure, num_epochs= 30, l1 = None, pred_step = 5, lr = 1e-4, decay_rate = 1.0, clr = None, path = 'torch_nn_model_temp'):
    nn_model = torch_train_nn(nn_structure, dataloader, l1 = l1, epochs = num_epochs, step = pred_step, lr = lr, decay_rate = decay_rate, clr = clr)
    save_torch_nn_model(nn_model, path)
    return nn_model

def load_torch_nn_model(nn_model, model_param_name):
    nn_model.load_state_dict(torch.load(model_param_name))
    return nn_model

def save_torch_nn_model(nn_model, path):
    torch.save(nn_model.state_dict(), path)

