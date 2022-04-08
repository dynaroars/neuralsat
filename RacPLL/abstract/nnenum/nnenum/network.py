'''
Stanley Bak

Network container classes for nnenum
'''

import numpy as np
import onnx
from scipy.signal import convolve2d

from nnenum.util import Freezable
from nnenum.timerutil import Timers

class NeuralNetwork(Freezable):
    'neural network container'

    def __init__(self, layers):

        assert layers, "layers should be a non-empty list"

        for i, layer in enumerate(layers):
            assert layer.layer_num == i, f"Layer {i} has incorrect layer num: {layer.layer_num}: {layer}"
        
        self.layers = layers
        self.check_io()

        for layer in layers:
            layer.network = self

        self.freeze_attrs()

    def __str__(self):
        return f'[NeuralNetwork with {len(self.layers)} layers with {self.layers[0].get_input_shape()} input and ' + \
          f'{self.get_output_shape()} output]'

    def num_relu_layers(self):
        'count the number of relu layers'

        rv = 0

        for l in self.layers:
            if isinstance(l, ReluLayer):
                rv += 1

        return rv

    def num_relu_neurons(self):
        'count the number of relu neurons'

        rv = 0

        for l in self.layers:
            if isinstance(l, ReluLayer):
                count = 1

                for dim in l.shape:
                    count *= dim

                rv += count

        return rv

    def get_input_shape(self):
        'get the input shape to the first layer'

        return self.layers[0].get_input_shape()

    def get_output_shape(self):
        'get the output shape from the last layer'

        return self.layers[-1].get_output_shape()

    def get_num_inputs(self):
        'get the scalar number of inputs'

        shape = self.get_input_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def get_num_outputs(self):
        'get the scalar number of outputs'

        shape = self.get_output_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def execute(self, input_vec, save_branching=False):
        '''execute the neural network with the given input vector

        if save_branching is True, returns (output, branch_list), where branch_list contains one list for each layer,
            and each layer-list is a list of the branching decisions taken by each neuron. For layers with ReLUs, this
            will be True/False values (True if positive branch is taken), for max pooling layers these will be ints, or
            lists of ints (if multiple max values are equal)
        
        otherwise, just returns output
        '''

        if save_branching:
            branch_list = []

        state = input_vec.copy() # test with float32 dtype?
        
        if state.shape != self.get_input_shape():
            state = nn_unflatten(state, self.get_input_shape())

        for layer in self.layers:
            if save_branching and isinstance(layer, ReluLayer) or isinstance(layer, PoolingLayer):
                state, layer_branch_list = layer.execute(state, save_branching=True)
                branch_list.append(layer_branch_list)
            else:
                if save_branching:
                    branch_list.append([])

                assert state.shape == layer.get_input_shape()
                state = layer.execute(state)
                assert state.shape == layer.get_output_shape()

        assert state.shape == self.get_output_shape()

        rv = (state, branch_list) if save_branching else state
        
        return rv

    def check_io(self):
        'check the neural network for input / output compatibility'

        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            prev_output_shape = self.layers[i-1].get_output_shape()
            my_input_shape = layer.get_input_shape()

            assert prev_output_shape == my_input_shape, f"output of layer {i-1} was {prev_output_shape}, " + \
              f"and this doesn't match input of layer {i} which is {my_input_shape}"

class ReluLayer(Freezable):
    'relu layer'

    def __init__(self, layer_num, shape, filter_func=None):
        '''
        filter_func(i) returns True if output i should have a relu branch
        '''

        self.layer_num = layer_num
        self.shape = shape

        self.filter_func = filter_func # returns True if relu should be applied for neuron i

    def __str__(self):
        return f'[ReluLayer with shape {self.shape}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.shape

    def execute(self, state, save_branching=False):
        '''execute the layer on a concrete state

        if save_branching is True, returns (output, branch_list), where branch_list is a list of booleans for each
            neuron in the layer that is True if the nonnegative branch of the ReLU was taken, False if negative
 
        otherwise, just returns output
        '''

        Timers.tic('execute relu')

        if save_branching:
            branch_list = []

        assert state.shape == self.get_input_shape(), f"state shape to fully connected layer was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        if save_branching:
            for i, val in enumerate(state):
                if self.filter_func is not None:
                    if not self.filter_func(i):
                        continue
                    
                branch_list.append(val >= 0)

        if self.filter_func is None:
            state = np.clip(state, 0, np.inf)
        else:
            res = []

            for i, val in enumerate(state):
                if not self.filter_func(i):
                    res.append(val)
                else:
                    res.append(max(0, val))

            state = np.array(res, dtype=float)
            
        rv = nn_unflatten(state, self.shape)

        rv = (rv, branch_list) if save_branching else rv

        Timers.toc('execute relu')
        
        return rv

class FlattenLayer(Freezable):
    'flatten onnx layer'

    def __init__(self, layer_num, input_shape):

        self.layer_num = layer_num
        self.input_shape = input_shape
        self.network = None # populated later when constructing network

        os = 1

        for i in input_shape:
            os *= i

        self.output_shape = (os, )

        self.freeze_attrs()

    def __str__(self):
        return f'[Flatten with input {self.get_input_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.output_shape

    def transform_star(self, star):
        'transform the star for this layer'

        # do nothing

    def transform_zono(self, zono):
        'transform the zono for this layer'

        # do nothing

    def execute(self, state):
        '''execute the layer on a concrete state
 
        returns output
        '''

        rv = nn_flatten(state)
        assert rv.shape == self.output_shape
        
        return rv

class AddLayer(Freezable):
    'add onnx layer'

    def __init__(self, layer_num, vec):

        self.layer_num = layer_num
        self.vec = vec
        self.network = None # populated later when constructing network

        self.freeze_attrs()

    def __str__(self):
        return f'[AddLayer with shape {self.get_input_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.vec.shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.vec.shape

    def transform_star(self, star):
        'transform the star'

        # well, hope star.bias is flat?

        star.bias += nn_flatten(self.vec)

    def transform_zono(self, zono):
        'transform the zono'

        zono.center += nn_flatten(self.vec)

    def execute(self, state):
        '''execute on a concrete state
 
        returns output
        '''

        return state + self.vec

class MatMulLayer(Freezable):
    'onnx matmul layer'

    def __init__(self, layer_num, mat, prev_layer_output_shape=None):

        assert prev_layer_output_shape is None or isinstance(prev_layer_output_shape, tuple)

        self.layer_num = layer_num
        self.mat = mat
        self.prev_layer_output_shape = prev_layer_output_shape
        self.network = None # populated later when constructing network

        assert len(mat.shape) == 2

        if prev_layer_output_shape is not None:
            expected_inputs = 1

            for x in prev_layer_output_shape:
                expected_inputs *= x

            assert expected_inputs == mat.shape[1], f"MatMulLayer matrix shape was {mat.shape}, but " + \
                f"prev_layer_output_shape {prev_layer_output_shape} needs {expected_inputs} columns"
        
        self.freeze_attrs()

    def __str__(self):
        return f'[MatMulLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        rv = self.prev_layer_output_shape

        if rv is None:
            rv = (self.mat.shape[1],)

        return rv

    def get_output_shape(self):
        'get the output shape from this layer'

        return (self.mat.shape[0],)

    def transform_star(self, star):
        'apply on star'

        star.a_mat = np.dot(self.mat, star.a_mat)
        star.bias = np.dot(self.mat, star.bias)

    def transform_zono(self, zono):
        'apply on zono'

        zono.mat_t = np.dot(self.mat, zono.mat_t)
        zono.center = np.dot(self.mat, zono.center)

    def execute(self, state):
        '''execute on a concrete state
 
        returns output
        '''

        Timers.tic('execute matmul')

        assert state.shape == self.get_input_shape(), f"state shape to matmul was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        rv = np.dot(self.mat, state)

        assert rv.shape == self.get_output_shape()

        Timers.toc('execute matmul')
        
        return rv

class FullyConnectedLayer(Freezable):
    'fully connected layer'

    def __init__(self, layer_num, weights, biases, prev_layer_output_shape=None):

        assert prev_layer_output_shape is None or isinstance(prev_layer_output_shape, tuple)

        if isinstance(weights, list):
            weights = np.array(weights, dtype=float)

        if isinstance(biases, list):
            biases = np.array(biases, dtype=float)
        
        self.layer_num = layer_num
        self.weights = weights
        self.biases = biases
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None

        assert biases.shape[0] == weights.shape[0], "biases vec in layer " + \
            f"{layer_num} has length {biases.shape[0]}, but weights matrix has height " + \
            f"{weights.shape[0]}"

        assert len(biases.shape) == 1, f'expected 1-d bias vector at layer {layer_num}, got {biases.shape}'
        assert len(weights.shape) == 2

        if prev_layer_output_shape is not None:
            expected_inputs = 1

            for x in prev_layer_output_shape:
                expected_inputs *= x

            assert expected_inputs == weights.shape[1], f"FC Layer weight matrix shape was {weights.shape}, but " + \
                f"prev_layer_output_shape {prev_layer_output_shape} needs {expected_inputs} columns"
        
        self.freeze_attrs()

    def __str__(self):
        return f'[FullyConnectedLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        rv = self.prev_layer_output_shape

        if rv is None:
            rv = (self.weights.shape[1],)

        return rv

    def get_output_shape(self):
        'get the output shape from this layer'

        return (self.weights.shape[0],)

    def transform_star(self, star):
        'apply the linear transformation part of the layer to the passed-in lp_star (not relu)'

        if star.a_mat is None:
            star.a_mat = self.weights.copy()
        else:
            star.a_mat = np.dot(self.weights, star.a_mat)

        if star.bias is None:
            star.bias = self.biases.copy()
        else:
            star.bias = np.dot(self.weights, star.bias) + self.biases

    def transform_zono(self, zono):
        'apply the linear transformation part of the layer to the passed-in zonotope (not relu)'

        zono.mat_t = np.dot(self.weights, zono.mat_t)
        zono.center = np.dot(self.weights, zono.center) + self.biases

    def execute(self, state):
        '''execute the fully connected layer on a concrete state
 
        returns output
        '''

        Timers.tic('execute fully connected')

        assert state.shape == self.get_input_shape(), f"state shape to fully connected layer was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        rv = np.dot(self.weights, state)

        assert len(self.biases.shape) == 1
        rv = rv + self.biases
        assert len(rv.shape) == 1

        assert rv.shape == self.get_output_shape()

        Timers.toc('execute fully connected')
        
        return rv

class Convolutional2dLayer(Freezable):
    '''a 2d convolutional layer which takes in multi-channel 2d input data and
    outputs multi-channel 2d data
    '''

    def __init__(self, layer_num, kernels, biases, prev_layer_output_shape, mode='same', boundary='fill'):
        self.layer_num = layer_num
        self.biases = biases
        self.mode = mode
        self.boundary = boundary

        assert isinstance(prev_layer_output_shape, tuple), f"prev_layer_shape was {prev_layer_output_shape}"
        
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None # assigned on network construction

        self.kernels = [] # a list of lists of 2d kernels

        assert len(prev_layer_output_shape) == 3, "previous layer should provide 3 channel output"

        assert len(kernels) >= 1, "need at least one kernel"
        assert isinstance(biases, np.ndarray)
        assert len(kernels.shape) == 4, "expected shape is 4: (# output channels, # input channels, x, y); " + \
                                f"got: {kernels.shape}"

        # for now, all kernels have same width and height so this is a good sanity check for input correctness
        assert kernels[0][0].shape[0] == kernels[0][0].shape[1], \
            f"kernel w and h are not the same: {kernels[0][0].shape}"

        num_output_channels = kernels.shape[0]
        assert biases.shape == (num_output_channels, ), "expected one bias per output channel, shape: " + \
                                                        f"({num_output_channels}, ), got {biases.shape}"

        for k in kernels:
            flipped_channel_kernel = []
            self.kernels.append(flipped_channel_kernel)
            
            for channel_kernel in k:
                assert len(channel_kernel.shape) == 2, "expected a list of list of 2d kernels"
                # flip each kernel since convolution2d works in reverse order
                flipped_channel_kernel.append(np.flipud(np.fliplr(channel_kernel)))
        
        self.freeze_attrs()

    def __str__(self):
        return f'[Convolutional2dLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.prev_layer_output_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        # prev_layer_output_shape: <height, width, depth>

        depth = len(self.kernels)
        height = self.prev_layer_output_shape[0]
        width = self.prev_layer_output_shape[1]

        if self.mode == 'valid':
            height -= self.kernels[0][0].shape[0] - 1
            width -= self.kernels[0][0].shape[1] - 1

        return (height, width, depth)

    def transform_star(self, star):
        'apply the linear transformation part of the layer to the passed-in lp_star (not relu)'

        shape = self.get_input_shape()

        # a_mat has one generator PER COLUMN
        result_columns = []

        for cindex in range(star.a_mat.shape[1]):
            column = star.a_mat[:, cindex]

            multichannel_state = nn_unflatten(column, shape)
            multichannel_state = self.execute(multichannel_state, zero_bias=True)
            flat = nn_flatten(multichannel_state)
            flat.shape = (flat.size, 1)
            result_columns.append(flat)

        star.a_mat = np.hstack(result_columns)

        # bias (anchor) transformation includes layer bias
        multichannel_state = nn_unflatten(star.bias, shape)
        multichannel_state = self.execute(multichannel_state)
        flat = nn_flatten(multichannel_state)
        star.bias = flat

        assert star.bias.size == star.a_mat.shape[0]

    def transform_zono(self, zono):
        'apply the linear transformation part of the layer to the passed-in zonotope (not relu)'

        # mat_t has one generator PER COLUMN
        shape = self.get_input_shape()

        result_columns = []

        for cindex in range(zono.mat_t.shape[1]):
            column = zono.mat_t[:, cindex]

            multichannel_state = nn_unflatten(column, shape)
            multichannel_state = self.execute(multichannel_state, zero_bias=True)

            flat = nn_flatten(multichannel_state)
            flat.shape = (flat.size, 1)
            result_columns.append(flat)
            
        zono.mat_t = np.hstack(result_columns)

        # center transformation includes layer bias
        multichannel_state = nn_unflatten(zono.center, shape)
        multichannel_state = self.execute(multichannel_state)
        flat = nn_flatten(multichannel_state)
        zono.center = flat

        assert zono.center.size == zono.mat_t.shape[0]

    def execute(self, state, zero_bias=False):
        '''execute the convolutional layer on a concrete state

        if save_branching is True, returns (output, branch_list), where branch_list is a list of booleans for each
            relu neuron that is True if input is nonnegative and False otherwise

        if zero_bias is True, use a zero bias instead of what's in the layer (used in ImageStar computations)       
 
        otherwise, just returns output
        '''

        Timers.tic('execute Convolutional2dLayer')

        assert state.shape == self.prev_layer_output_shape, f"expected shape {self.prev_layer_output_shape}, " + \
                                                            f"got {state.shape}"

        output = []
        output_shape = self.get_output_shape()

        for kernel, bias in zip(self.kernels, self.biases):
            out = np.empty(output_shape[:-1])

            bias = bias if not zero_bias else 0
            
            out.fill(bias)
            output.append(out)

            for i, channel_kernel in enumerate(kernel):
                # depth is last channel (bad for convolution performance I think)
                state2d = state[:, :, i] 
                Timers.tic('convolve2d')
                channel_out = convolve2d(state2d, channel_kernel, mode=self.mode, boundary=self.boundary)
                Timers.toc('convolve2d')

                Timers.tic('add')
                out += channel_out
                Timers.toc('add')
                
        Timers.tic('output transpose')
        output = np.array(output, dtype=float)
        # convert to y, x, z
        output = output.transpose((1, 2, 0))
        Timers.toc('output transpose')
    
        Timers.toc('execute Convolutional2dLayer')

        return output

class PoolingLayer(Freezable):
    '''a 2d max/mean pooling layer (multi channel)
    '''

    def __init__(self, layer_num, kernel_size, prev_layer_output_shape, method='max'):
        self.layer_num = layer_num
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None # assigned on network construction

        assert method in ['max', 'mean'], f"unknown method: {method}"

        self.method = method

        self.freeze_attrs()

    def __str__(self):
        s = self.kernel_size
        
        return f'[PoolingLayer ({self.method}) {s}x{s} with stride {self.stride}, ' + \
               f'input shape {self.get_input_shape()} and output shape {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.prev_layer_output_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        s = self.kernel_size

        height = self.prev_layer_output_shape[0] // s
        width = self.prev_layer_output_shape[1] // s

        rv = [height, width]

        if len(self.prev_layer_output_shape) > 2:
            rv += self.prev_layer_output_shape[2:]

        return tuple(rv)

    def execute(self, state, save_branching=False):
        '''execute pooling layer, potentially saving branching informaton

        branching info will be an int for each output (if max pool), or possibly a LIST of ints (if two inputs match)
        '''

        Timers.tic('execute PoolingLayer')

        ksize = self.kernel_size

        assert len(state.shape) == 3
        assert state.shape[0] % ksize == 0
        assert state.shape[1] % ksize == 0

        if save_branching:
            rv = self._execute_with_branching(state)
        else:
            rv = self._execute_without_branching(state)

        Timers.toc('execute PoolingLayer')

        return rv

    # based on code from:
    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    def _execute_without_branching(self, state):
        'fast max/mean pooling without storing branching information'
        
        ksize = self.kernel_size

        ny = state.shape[0] // ksize
        nx = state.shape[1] // ksize
        
        new_shape = (ny, ksize, nx, ksize) + state.shape[2:]

        if self.method == 'max':
            rv = np.nanmax(state.reshape(new_shape), axis=(1, 3))
        else:
            assert self.method == 'mean'
            rv = np.nanmean(state.reshape(new_shape), axis=(1, 3))

        return rv

    def _execute_with_branching(self, state):
        '''execute pooling layer on a concrete state

        branch_list will be an int for each output (if max pool), or possibly a LIST of ints (if two inputs match)

        note: this is about 50x slower than without branching on a 224x224x3 input
        '''

        Timers.tic('execute_pooling_with_branching')

        ksize = self.kernel_size 

        height = state.shape[0] // ksize
        width = state.shape[1] // ksize
        depth = state.shape[2]
        
        if self.method == 'max':
            output = np.full((height, width, depth), -np.inf, dtype=float)
            branch_list = [None] * (depth * width * height)
        else:
            output = np.zeros((height, width, depth), dtype=float)
            branch_list = []

        for d in range(state.shape[2]):
            depth_offset = d * (width * height)
                            
            for row_index in range(state.shape[0]):
                output_row = row_index // ksize
                height_offset = output_row * width

                for col_index in range(state.shape[1]):
                    block_index = col_index // ksize

                    val = state[row_index, col_index, d]

                    if self.method == 'max':
                        epsilon = 1e-9
                        
                        if val - epsilon > output[output_row, block_index, d]:
                            # new max value
                            output[output_row, block_index, d] = val

                            max_index = col_index % ksize
                            row_in_block = row_index % ksize
                            mindex = row_in_block * ksize + max_index
                            branch_list[depth_offset + height_offset + block_index] = mindex
                        elif val + epsilon > output[output_row, block_index, d]:
                            # two branches are both possible (within epsilon tolerance), both should be in branch string

                            output[output_row, block_index, d] = max(output[output_row, block_index, d], val)

                            max_index = col_index % ksize
                            row_in_block = row_index % ksize
                            mindex = row_in_block * ksize + max_index
                            bindex = depth_offset + height_offset + block_index

                            if isinstance(branch_list[bindex], int):
                                branch_list[bindex] = [branch_list[bindex], mindex]
                            else:
                                branch_list[bindex].append(mindex)
                            
                    else:
                        output[output_row, block_index, d] += val

        if self.method == 'mean':
            divider = self.kernel_size**2
            output = output / divider
            
        rv = (output, branch_list)

        Timers.toc('execute_pooling_with_branching')
            
        return rv

def images_to_init_box(min_image, max_image):
    'create an initial box from a min and max image'

    min_vec = nn_flatten(min_image)
    max_vec = nn_flatten(max_image)
    rv = []

    for a, b in zip(min_vec, max_vec):
        rv.append((a, b))

    return rv

def nn_flatten(image, order='C'):
    'flatten a multichannel image to a 1-d array'

    return image.flatten(order)

def nn_unflatten(image, shape, order='C'):
    '''unflatten to a multichannel image from a 1-d array

    this uses reshape, so may not be a copy
    '''

    assert len(image.shape) == 1

    rv = image.reshape(shape, order=order)

    return rv

def convert_weights(weights):
    'convert weights from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for weight_mat in weights:
        layers.append(np.array(weight_mat, dtype=float))

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def convert_biases(biases):
    'convert biases from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for biases_vec in biases:
        bias_ar = np.array(biases_vec, dtype=float)
        bias_ar.shape = (len(biases_vec),)
        
        layers.append(bias_ar)

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def weights_biases_to_nn(weights, biases, dtype=None):
    '''create a NeuralNetwork from a weights and biases matrix

    this assumes every layer is a fully-connected layer followed by a ReLU, except for the last one
    '''

    if isinstance(weights, list):
        weights = convert_weights(weights)

    if isinstance(biases, list):
        biases = convert_biases(biases)

    num_layers = weights.shape[0]
    assert biases.shape[0] == num_layers, f"nn has {num_layers} layers, but biases shape was {biases.shape}"

    layers = []

    index = 0

    for i, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):
        add_relu = i < num_layers - 1

        if dtype is not None:
            layer_weights = layer_weights.astype(dtype)
            layer_biases = layer_biases.astype(dtype)
        
        layers.append(FullyConnectedLayer(index, layer_weights, layer_biases))
        index += 1

        if add_relu:
            layers.append(ReluLayer(index, layers[-1].get_output_shape()))
            index += 1

    return NeuralNetwork(layers)

