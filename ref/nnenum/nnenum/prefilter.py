'''
Prefilter logic

Overapproximation for shortciruiting splitting decisions during enumeration

Note: the logic for a lack of prefilter... using lp at each step, is also encoded here.
'''

import time
import numpy as np

from nnenum.settings import Settings
from nnenum.util import Freezable
from nnenum.zonotope import Zonotope
from nnenum.timerutil import Timers

from nnenum.network import nn_flatten, nn_unflatten
from nnenum.lputil import update_bounds_lp

class LpCanceledException(Exception):
    'an exception used for when lp is cancelled'

def prod(l):
    'like math.prod in python 3.8'

    rv = 1

    for i in l:
        rv *= i

    return rv

def exec_relus_up_to(state, index):
    'execute reluts on the passed in state vector up to index'

    Timers.tic('exec_relus_up_to')

    for i in range(index):
        if state[i] < 0:
            state[i] = 0

    # clip is slower here, for some reason
    #state[:index] = np.clip(state[:index], 0, np.inf)

    Timers.toc('exec_relus_up_to')

def sort_splits(layer_bounds, splits):
    '''sort splitting neurons according to 

    Settings.SPLIT_SMALLEST and Settings.SPLIT_LARGEST
    '''

    Timers.tic('sort_splits')

    if splits.size <= 1 or Settings.SPLIT_ORDER == Settings.SPLIT_INORDER:
        rv = splits
    elif Settings.SPLIT_ORDER in [Settings.SPLIT_LARGEST, Settings.SPLIT_SMALLEST]:
        sizes = layer_bounds[splits, 1] - layer_bounds[splits, 0]

        reverse = Settings.SPLIT_ORDER == Settings.SPLIT_LARGEST
        
        new_branches = [n for _, n in sorted(zip(sizes, splits), reverse=reverse)]
        rv = np.array(new_branches)
    else:
        assert Settings.SPLIT_ORDER == Settings.SPLIT_ONE_NORM
        # sort by largest distance from zero
        sizes = []

        for sindex in splits:
            sizes.append(min(layer_bounds[sindex, 1], -layer_bounds[sindex, 0]))
                         
        reverse = True

        new_branches = [n for _, n in sorted(zip(sizes, splits), reverse=reverse)]
        rv = np.array(new_branches)

    Timers.toc('sort_splits')

    return rv

class OutputBounds(Freezable):
    'container object for output bounds for each neuron in a single layer'

    def __init__(self, prefilter_parent):
        '''
        Initialize the output bounds.
        '''

        self.prefilter = prefilter_parent
        
        self.layer_bounds = None # layer bounds for branching neurons
        self.branching_neurons = None

        self.freeze_attrs()

    def recompute_bounds(self, star, use_lp, start_time, depth):
        '''recompute the layer bounds for all splitting neurons

        This will assign layer_bounds and branching_neurons
        '''

        Timers.tic('recompute_bounds')

        if self.layer_bounds is None:
            self.layer_bounds = self.prefilter.zono.box_bounds()

            self.branching_neurons = np.nonzero(np.logical_and(self.layer_bounds[:, 0] < -Settings.SPLIT_TOLERANCE,
                                                               self.layer_bounds[:, 1] > Settings.SPLIT_TOLERANCE))[0]
        else:
            self.branching_neurons = self.prefilter.zono.update_output_bounds(self.layer_bounds, self.branching_neurons)

        #print(f".prefilter.full Branching neurons before lp: {len(bn)}: {bn}")

        if use_lp and self.prefilter.simulation is not None:
            # trim further using LP

            if start_time is not None and Settings.TIMEOUT != np.inf:
                def check_cancel_func():
                    'raise exception if we should cancel'

                    now = time.perf_counter()

                    if now - start_time > Settings.TIMEOUT:
                        raise LpCanceledException("timeout")
            else:
                check_cancel_func = None

            self.branching_neurons = update_bounds_lp(self.layer_bounds, star, self.prefilter.simulation[1],
                                                      self.branching_neurons, depth,
                                                      check_cancel_func=check_cancel_func)

        self.branching_neurons = sort_splits(self.layer_bounds, self.branching_neurons)

        Timers.toc('recompute_bounds')
                                
    def split(self, other_prefilter, i, self_gets_positive):
        '''a star with this prefilter is being split along neuron i, 

        return a copy of the output bounds for the other star, adjusting
        the bounds based on how we split
        '''

        # manually copy
        rv = OutputBounds(other_prefilter)
        rv.layer_bounds = self.layer_bounds.copy()

        assert self.branching_neurons[0] == i
        self.branching_neurons = self.branching_neurons[1:]
        rv.branching_neurons = self.branching_neurons # shallow copy is fine, as this gets recomputed and reassigned

        # trim to appropriate side on neuron i
        if self_gets_positive:
            self.layer_bounds[i, 0] = 0
            rv.layer_bounds[i, 1] = 0
        else:
            # self gets negative
            rv.layer_bounds[i, 0] = 0
            self.layer_bounds[i, 1] = 0

        return rv

class Prefilter(Freezable):
    'main container for prefilter data and logic'

    def __init__(self):
        self.simulation = None # 2-list [input, output]        
        self.zono = None

        # used for prefilter_zonotope
        self.output_bounds = None

        self.freeze_attrs()

    def init_from_star(self, star):
        'initialize the prefilter from an lp_star'

        self.simulation = star.minimize_vec(None, return_io=True)

        box_bounds = star.get_input_box_bounds()
        
        self.zono = Zonotope(star.bias, star.a_mat, box_bounds)

    def init_from_uncompressed_box(self, uncompressed_init_box, star, box_bounds):
        'initialize from an uncompressed initial box'

        # sim state is initially the center of the box
        sim_input = []
        sim_output = []
        tol = 1e-9

        dtype = type(uncompressed_init_box[0][0])

        for row, i in enumerate(uncompressed_init_box):
            mid = (i[0] + i[1]) / 2.0
            sim_output.append(mid) # sim output is uncompressed

            if Settings.COMPRESS_INIT_BOX:
                if abs(i[1] - i[0]) > tol:
                    sim_input.append(mid)
            else:   
                if not Settings.SKIP_COMPRESSED_CHECK:
                    assert abs(i[1] - i[0]) > tol, f"init box looks compressed (row {row} is range {i}), " + \
                        "use Settings.SKIP_COMPRESSED_CHECK to disable"

                sim_input.append(mid)

        sim_input = np.array(sim_input, dtype=dtype)
        sim_output = np.array(sim_output, dtype=dtype)

        self.simulation = [sim_input, sim_output]

        self.zono = Zonotope(star.bias, star.a_mat, box_bounds.copy())

        assert star.a_mat is self.zono.mat_t
        assert star.bias is self.zono.center

    def clear_output_bounds(self):
        'clear_output_bounds (new layer started)'

        self.output_bounds = None

    def apply_linear_layer(self, layer, star):
        '''do the linear transform part of the non-splitting layer

        the passed-in star is already-transformed
        '''

        if self.simulation is not None:
            shape = layer.get_input_shape()
            input_tensor = nn_unflatten(self.simulation[1], shape)

            if star.a_mat is not None:
                input_tensor = input_tensor.astype(star.a_mat.dtype)
                
            output_tensor = layer.execute(input_tensor)
            self.simulation[1] = nn_flatten(output_tensor)

        # zonotope sanity checks
        assert self.zono.mat_t is star.a_mat
        assert self.zono.center is star.bias

    def init_relu_layer(self, star, layer, start_time, depth):
        'initialize the layer bounds when we first get to a relu layer'

        self.output_bounds = OutputBounds(self)

        star_has_splits = star.lpi.get_num_rows() > 0
        self.output_bounds.recompute_bounds(star, star_has_splits, start_time, depth)

        if layer.filter_func is not None:
            new_bn = []

            for i in self.output_bounds.branching_neurons:
                if layer.filter_func(i):
                    new_bn.append(i)

            self.output_bounds.branching_neurons = np.array(new_bn)

        # unit testing function
        if Settings.TEST_FUNC_BEFORE_ASSIGNMENT is not None:
            Settings.TEST_FUNC_BEFORE_ASSIGNMENT()

        zero_indices = np.nonzero(self.output_bounds.layer_bounds[:, 1] <= Settings.SPLIT_TOLERANCE)[0]
        self.assign_zeros(star, zero_indices)

    def split_relu(self, neuron_index, pos_star, neg_star, self_gets_positive, start_time, depth):
        '''split the star associated with this prefilter. This updates this
        prefilter's prediction, as well as returns a new prefilter to be used with
        the other star associated with the split

        neuron_index - the neuron index on which we're splitting
        take_positive - does the self prefilter take the >= case (false is negative case)
        pos_star - the >= 0 direction split lp_star
        neg_star - the negative direction split lp_star

        returns a Prefilter object for the other star
        '''

        i = neuron_index
        rv = Prefilter()

        # self gets positive
        if self_gets_positive:
            pos, neg = self, rv
        else:
            neg, pos = self, rv

        if self.output_bounds is not None:
            rv.output_bounds = self.output_bounds.split(rv, i, self_gets_positive)

        # zono: only deep copy the init_bounds
        if self_gets_positive:
            rv.zono = Zonotope(neg_star.bias, neg_star.a_mat, self.zono.init_bounds.copy())
            assert rv.zono.mat_t is neg_star.a_mat
            assert rv.zono.center is neg_star.bias
        else:
            rv.zono = Zonotope(pos_star.bias, pos_star.a_mat, self.zono.init_bounds.copy())
            assert rv.zono.mat_t is pos_star.a_mat
            assert rv.zono.center is pos_star.bias

        ### try shrinking prefilter zonotopes ###
        if Settings.CONTRACT_ZONOTOPE:
            row = pos_star.a_mat[i]
            bias = pos_star.bias[i]
        
            Timers.tic("contract_zonotope")
            pos.zono.contract_domain(-row, bias)
            neg.zono.contract_domain(row, -bias)
            Timers.toc("contract_zonotope")
            
        if Settings.CONTRACT_ZONOTOPE_LP:
            row = pos_star.a_mat[i]
            bias = pos_star.bias[i]
            
            Timers.tic("contract_zonotope_lp")
            pos.zono.contract_lp(pos_star, -row, bias)
            neg.zono.contract_lp(neg_star, row, -bias)
            Timers.toc("contract_zonotope_lp")

        if Settings.EAGER_BOUNDS:
            ### RECOMPUTE LAYER BOUNDS eagerly (for remaining neurons) (and witnesses)
            pos.domain_shrank(pos_star, start_time, depth)
            neg.domain_shrank(neg_star, start_time, depth)

            # tolerance for lp solver is about 1e-6
            assert pos.simulation[1][i] >= -1e-3, f"pos sim for {i} was {pos.simulation[1][i]}"

            # neg should exactly be equal to zero, since we assigned a_mat and bias to zero
            assert abs(neg.simulation[1][i]) <= Settings.SPLIT_TOLERANCE, f"neg sim for {i} was {neg.simulation[1][i]}"

        return rv

    def domain_shrank(self, star, start_time, depth):
        '''the domain star was contracted (from split or violation intersection)

        update the prefilter state to reflect this (other than zono)
        '''

        # find new witness
        Timers.tic('witness_lp')
        self.simulation = star.minimize_vec(None, return_io=True)
        Timers.toc('witness_lp')

        if self.output_bounds.branching_neurons.size > 0:
            old_branching_neurons = self.output_bounds.branching_neurons.copy()

            self.output_bounds.recompute_bounds(star, True, start_time, depth)

            # update stars based on branching neurons that became fully negative
            new_zeros = []
            for i in old_branching_neurons:
                if self.output_bounds.layer_bounds[i, 1] < Settings.SPLIT_TOLERANCE:
                    new_zeros.append(i)

            self.assign_zeros(star, new_zeros)

    def assign_zeros(self, star, zero_indices):
        '''
        assign the zeros to the star and prefilter eagerly (right after updating bounds)
        '''

        Timers.tic('assign_zeros')

        star.bias[zero_indices] = 0
        star.a_mat[zero_indices] = 0

        if self.simulation is not None:
            self.simulation[1][zero_indices] = 0

        Timers.toc('assign_zeros')
        
