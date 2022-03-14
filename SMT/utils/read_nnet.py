import numpy as np 

class Layer:
    
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

        self.output_shape = (None, weight.shape[1])

    def get_weights(self):
        return self.weight, self.bias


class Network:

    def __init__(self, nnet_path):
        self.layers = []

        weights, biases, lbs, ubs, means, ranges = read_nnet(nnet_path, with_norm=True)

        for i in range(len(weights)):
            w = weights[i].transpose()
            b = biases[i]
            self.layers.append(Layer(w, b))

        self.input_shape = (None, weights[0].shape[1])
        self.input_lower_bounds = lbs
        self.input_upper_bounds = ubs
        self.input_means = means[:-1]
        self.input_ranges = ranges[:-1]

        self.output_mean = means[-1]
        self.output_range = ranges[-1]


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

