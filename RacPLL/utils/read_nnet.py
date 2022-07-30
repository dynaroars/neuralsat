import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np 

import settings

class NetworkNNET(nn.Module):

    def __init__(self, nnet_path):
        super().__init__()

        layers = []
        weights, biases, lbs, ubs, means, ranges = read_nnet(nnet_path, with_norm=True)

        n_layers = len(weights)
        for i in range(n_layers):
            w = torch.Tensor(weights[i]).to(settings.DTYPE)
            b = torch.Tensor(biases[i]).to(settings.DTYPE)

            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data = w
            layer.bias.data = b

            layers.append(layer)
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        
        self.n_input = weights[0].shape[1]
        self.n_output = weights[-1].shape[0]
        
        self.input_lower_bounds = lbs
        self.input_upper_bounds = ubs
        self.input_means = means[:-1]
        self.input_ranges = ranges[:-1]

        self.output_mean = means[-1]
        self.output_range = ranges[-1]

        self.path = nnet_path

        self.layers = nn.Sequential(*layers)

        # update after loading model
        self.layers_mapping = None
        self.input_shape = (1, self.n_input)

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


    @torch.no_grad()
    def get_assignment(self, x):
        idx = 0
        implication = {}
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                s = torch.zeros_like(x, dtype=int) 
                s[x > 0] = 1
                implication.update(dict(zip(self.layers_mapping[idx], s.numpy().astype(dtype=bool))))
                idx += 1
        return implication

    @torch.no_grad()
    def get_concrete(self, x):
        idx = 0
        implication = {}
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                implication.update(dict(zip(self.layers_mapping[idx], x.view(-1))))
                idx += 1
            x = layer(x)
        return implication

    @torch.no_grad()
    def forward_layer(self, x, lid):
        relu_idx = 0
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                relu_idx += 1
            if relu_idx <= lid:
                continue
            x = layer(x)
        return x


def read_nnet(nnet_file, with_norm=False):
    '''
    Read a .nnet file and return list of weight matrices and bias vectors
    
    Inputs:
        nnet_file: (string) .nnet file to read
        with_norm: (bool) If true, return normalization parameters
        
    Returns: 
        weights: List of weight matrices for fully connected network
        biases: List of bias vectors for fully connected network
    '''   
        
    # Open NNet file
    f = open(nnet_file,'r')
    
    # Skip header lines
    line = f.readline()
    while line[:2]=="//":
        line = f.readline()
        
    # Extract information about network architecture
    record = line.split(',')
    n_layers   = int(record[0])
    input_size   = int(record[1])

    line = f.readline()
    record = line.split(',')
    layer_sizes = np.zeros(n_layers+1,'int')
    for i in range(n_layers+1):
        layer_sizes[i]=int(record[i])

    # Skip extra obsolete parameter line
    f.readline()
    
    # Read the normalization information
    line = f.readline()
    input_lb = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    input_ub = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    means = [float(x) for x in line.strip().split(",")[:-1]]

    line = f.readline()
    ranges = [float(x) for x in line.strip().split(",")[:-1]]

    # Read weights and biases
    weights=[]
    biases = []
    for layer_id in range(n_layers):

        previous_size = layer_sizes[layer_id]
        current_size = layer_sizes[layer_id+1]
        weights.append([])
        biases.append([])
        weights[layer_id] = np.zeros((current_size,previous_size))
        for i in range(current_size):
            line=f.readline()
            aux = [float(x) for x in line.strip().split(",")[:-1]]
            for j in range(previous_size):
                weights[layer_id][i,j] = aux[j]
        #biases
        biases[layer_id] = np.zeros(current_size)
        for i in range(current_size):
            line=f.readline()
            x = float(line.strip().split(",")[0])
            biases[layer_id][i] = x

    f.close()
    
    if with_norm:
        return weights, biases, input_lb, input_ub, means, ranges
    return weights, biases

