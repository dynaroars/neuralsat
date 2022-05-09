import numpy as np
import os

def writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,fileName):

    with open(fileName,'w') as f2:

        #####################
        # First, we write the header lines:
        # The first line written is just a line of text
        # The second line gives the four values:
        #     Number of fully connected layers in the network
        #     Number of inputs to the network
        #     Number of outputs from the network
        #     Maximum size of any hidden layer
        # The third line gives the sizes of each layer, including the input and output layers
        # The fourth line gives an outdated flag, so this can be ignored
        # The fifth line specifies the minimum values each input can take
        # The sixth line specifies the maximum values each input can take
        #     Inputs passed to the network are truncated to be between this range
        # The seventh line gives the mean value of each input and of all outputs
        # The eighth line gives the range of each input and of all outputs
        #     These two lines are used to map raw inputs to the 0 mean, unit range of the inputs and outputs
        #     used during training
        # The ninth line begins the network weights and biases
        ####################
        f2.write("// Neural Network File Format by Kyle Julian, Stanford 2016\n")

        #Extract the necessary information and write the header information
        numLayers = len(weights)
        inputSize = weights[0].shape[1]
        outputSize = len(biases[-1])
        maxLayerSize = inputSize
        
        # Find maximum size of any hidden layer
        for b in biases:
            if len(b)>maxLayerSize :
                maxLayerSize = len(b)

        # Write data to header 
        f2.write("%d,%d,%d,%d,\n" % (numLayers,inputSize,outputSize,maxLayerSize) )
        f2.write("%d," % inputSize )
        for b in biases:
            f2.write("%d," % len(b) )
        f2.write("\n")
        f2.write("0,\n") #Unused Flag

        f2.write(','.join(str(inputMins[i])  for i in range(inputSize)) + ',\n') 
        f2.write(','.join(str(inputMaxes[i]) for i in range(inputSize)) + ',\n') 
        f2.write(','.join(str(means[i])      for i in range(inputSize+1)) + ',\n') 
        f2.write(','.join(str(ranges[i])     for i in range(inputSize+1)) + ',\n') 

        for w,b in zip(weights,biases):
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    f2.write("%.5f," % w[i][j])
                f2.write("\n")
                
            for i in range(len(b)):
                f2.write("%.5f,\n" % b[i])

def weights_correctness(weights, num_neurons):
    error_msg = "weights is inconsistent with num_neurons"
    for l in range(1, len(num_neurons)):
        assert weights[l].shape == (num_neurons[l], num_neurons[l-1]), error_msg

def bias_correctness(bias, num_neurons):
    error_msg = "bias is inconsistent with num_neurons"
    for l in range(1, len(num_neurons)):
        assert bias[l].shape == (num_neurons[l], ), error_msg

class Model:

    def __init__(self, num_neurons, weights=None, bias=None, 
        min_input_val=-30.0, max_input_val=30.0):

        self.num_neurons = num_neurons
        self.min_input_val = min_input_val
        self.max_input_val = max_input_val

        self.num_layers = len(num_neurons)
        self.max_num_hidden = max(num_neurons[1:-1])

        self.weights = {}
        self.bias = {}

        if weights is not None:
            weights_correctness(weights, num_neurons)
            self.weights = weights
        else:
            for l in range(1, self.num_layers):
                self.weights[l] = np.round(np.random.normal(0, 0.1, size=(num_neurons[l], num_neurons[l-1]))* 10, 1) 

        if bias is not None:
            bias_correctness(bias, num_neurons)
            self.bias = bias
        else:
            for l in range(1, self.num_layers):
                self.bias[l] = np.round(np.random.normal(0, 0.1, size=(num_neurons[l])), 1) * 10

    def save(self, filename, mean_input_val=-3.0, range_input_val=6.0):
        weights_list = []
        bias_list = []
        for l in range(1, self.num_layers):
            weights_list.append(self.weights[l])
            bias_list.append(self.bias[l])
        num_inputs = self.num_neurons[0]

        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        writeNNet(weights_list, bias_list, 
            [self.min_input_val]*num_inputs, [self.max_input_val]*num_inputs, 
            [mean_input_val]*(num_inputs+1), [range_input_val]*(num_inputs+1), 
            filename)

if __name__ == '__main__':
    nnet_name = 'random.nnet'
    num_neurons = [2, 2, 2, 3]
    model = Model(num_neurons)
    model.save(nnet_name)
