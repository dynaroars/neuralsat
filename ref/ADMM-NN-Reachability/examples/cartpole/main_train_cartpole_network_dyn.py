
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")

import nn_reachability.utilities as ut

from nn_reachability.nn_models import SystemDataSet, train_nn_torch

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
    '''Train a neural network to approximate the closed-loop dynamics of a cartpole system from the vector field samples.'''

    nx = 4

    # train a neural network
    is_train = True
    nn_width = 400

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

    if is_train:
        '''train a neural network to approximate the closed-loop cart pole dynamics'''
        input_samples = ut.load_pickle_file('cartpole_train_X_batch_0')
        output_samples = ut.load_pickle_file('cartpole_train_Y_batch_0')

        train_data_set = SystemDataSet(input_samples, output_samples)

        train_batch_size = 10
        train_loader = DataLoader(train_data_set, batch_size=train_batch_size, shuffle=True)

        nn_structure = nn_dynamics(nx, nn_width)
        nn_model = train_nn_torch(train_loader, nn_structure, num_epochs=100, l1=None,
                                  pred_step=0, lr=1e-4, decay_rate=1.0, clr=None)

        torch.save(nn_model, nn_file_name)

    torch.set_grad_enabled(False)
    nn_system = torch.load(nn_file_name)
