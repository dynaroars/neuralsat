'''
LP Star State (includes enumeration state variables)
Stanley Bak
'''

import numpy as np

from nnenum.lp_star import LpStar
from nnenum.prefilter import Prefilter
from nnenum.timerutil import Timers
from nnenum.util import Freezable, compress_init_box
from nnenum.network import FullyConnectedLayer, ReluLayer, FlattenLayer, AddLayer, MatMulLayer
from nnenum.specification import DisjunctiveSpec

from nnenum.settings import Settings

class LpStarState(Freezable):
    'variables and methods associated with verification using lp star representation'

    # if not None, split to get to this branch
    TARGET_BRANCH_TUPLE = None
    # 1-9 7, unsafe: '++++-++--+++++++++---++-++-++++++-++-++++--+++++'
    #                '++++++-+++-++---+++----++-+++-++'
    #                '++++++-+++-++---+++----++-+++-++'

    def __init__(self, uncompressed_init_box=None, spec=None, safe_spec_list=None):
        
        self.star = None
        self.prefilter = None
        
        self.cur_layer = 0
        self.work_frac = 1.0 # fraction of work represented by this star

        self.should_try_overapprox = True

        if safe_spec_list is not None:
            self.safe_spec_list = safe_spec_list
        elif isinstance(spec, DisjunctiveSpec):
            self.safe_spec_list = [False] * len(spec.spec_list)
        else:
            self.safe_spec_list = None

        # a list of 3-tuples describing the branching taken for this star (layer_index, neuron_index, branch_type)
        # see network.execute for the values of branch_type used by the different layer types
        self.branch_tuples = []

        self.distance_to_unsafe = None

        if uncompressed_init_box is not None:
            assert isinstance(uncompressed_init_box, np.ndarray), "init bounds should be given in a numpy array"
            assert uncompressed_init_box.dtype in [np.float32, np.float64], \
                f"init bounds dtype was not floating-point type: {uncompressed_init_box.dtype}"

            Timers.tic('from_init_box')
            self.from_init_box(uncompressed_init_box)
            Timers.toc('from_init_box')

        self.freeze_attrs()

    def __del__(self):
        # delete the circular reference which would prevent the memory from being freed
        if self.prefilter is not None and self.prefilter.output_bounds:
            self.prefilter.output_bounds.prefilter = None

    def __str__(self):
        split_str = "no splits"
        n = self.remaining_splits()

        if n > 0:
            str_list = [str(s) for s in self.prefilter.output_bounds.branching_neurons]
            split_str = f"{n} splits remaining: " + ", ".join(str_list)
        
        return f"LpStateState at layer {self.cur_layer} with {split_str}"

    def branch_str(self):
        'get the branch tuples string'

        assert self.branch_tuples is not None

        rv = "".join(["+" if tup[2] else "-" for tup in self.branch_tuples])

        return rv

    def remaining_splits(self):
        'get the number of remaining splits on the current layer'

        rv = 0

        # for the first star (initial set), output_bounds is NOT assigned
        if self.prefilter and self.prefilter.output_bounds:
            rv = self.prefilter.output_bounds.branching_neurons.size

        return rv

    def from_init_star(self, star):
        'initialize from an initial LpStar (makes a copy)'

        self.star = star.copy() # copy the star
        self.prefilter = Prefilter()
        self.prefilter.init_from_star(self.star)

    def from_init_box(self, uncompressed_init_box):
        'initialize from an initial box'

        Timers.tic('make bm')

        if Settings.COMPRESS_INIT_BOX:
            init_bm, init_bias, init_box = compress_init_box(uncompressed_init_box)
        else:
            dims = len(uncompressed_init_box)
            #init_bm = np.identity(dims)
            #init_bias = np.zeros(dims)
            init_bm = None
            init_bias = None
            init_box = uncompressed_init_box

        Timers.toc('make bm')

        # for finding concrete counterexamples
        Timers.tic('star')
        self.star = LpStar(init_bm, init_bias, init_box)
        Timers.toc('star')

        self.prefilter = Prefilter()
        self.prefilter.init_from_uncompressed_box(uncompressed_init_box, self.star, init_box)

    def is_finished(self, network):
        'is the current star finished?'

        return self.cur_layer >= len(network.layers)

    def propagate_up_to_split(self, network, start_time):
        'propagate up to the next split or until we finish with the network'

        depth = len(self.branch_tuples)

        while not self.is_finished(network):
            layer = network.layers[self.cur_layer]

            if isinstance(layer, ReluLayer):
                if self.prefilter.output_bounds is None:
                    # start of a relu layer
                    self.prefilter.init_relu_layer(self.star, layer, start_time, depth)

                if self.prefilter.output_bounds.branching_neurons.size > 0:
                    break

                self.next_layer()
            else:
                # non-relu layer
                self.apply_linear_layer(network)
                
                self.next_layer()

    def next_layer(self):
        'advance to the next layer'

        self.cur_layer += 1

        if self.prefilter:
            self.prefilter.clear_output_bounds()

    def apply_linear_layer(self, network):
        'apply linear transformation part of a layer'

        Timers.tic('starstate.apply_linear_layer')

        layer = network.layers[self.cur_layer]
        assert not isinstance(layer, ReluLayer)
        assert self.star
        assert self.prefilter

        layer.transform_star(self.star)

        # update zonotope shallow copy
        self.prefilter.zono.mat_t = self.star.a_mat
        self.prefilter.zono.center = self.star.bias

        self.prefilter.apply_linear_layer(layer, self.star)

        Timers.toc('starstate.apply_linear_layer')

    def split_enumerate(self, i, network, spec, start_time):
        '''
        helper for execute_relus

        split using enumerative strategy, returns the child LpStarState object

        ss is the lp star state
        i is the output (neuron) index we're splitting on
        '''

        #print(f".state spliting on neuron {i}")

        Timers.tic('split_enumerate')

        child = LpStarState()
        child.star = self.star.copy()
            
        # prefilter gets copied later

        if self.safe_spec_list is not None:
            child.safe_spec_list = self.safe_spec_list.copy()

        child.cur_layer = self.cur_layer

        # split work among 2 children
        self.work_frac /= 2.0
        child.work_frac = self.work_frac

        # choose which branch to go down
        if not LpStarState.TARGET_BRANCH_TUPLE:
            #sim_is_positive = self.prefilter.simulation[1][i] >= 0
            #self_gets_positive = sim_is_positive
            self_gets_positive = True
        else:
            self_gets_positive = LpStarState.TARGET_BRANCH_TUPLE[0] == '+'
            LpStarState.TARGET_BRANCH_TUPLE = LpStarState.TARGET_BRANCH_TUPLE[1:]

            took = 'pos' if self_gets_positive else 'neg'
            print(f"Info: Using TARGET_BRANCH_TUPLE for splitting on {i}, took {took}")

        # first do child, as it may be infeasible
        if self_gets_positive:
            pos, neg = self, child
        else:
            neg, pos = self, child
        
        ### ADD INITIAL STATE INTERSECTION
        row = self.star.a_mat[i]
        bias = self.star.bias[i]
        
        # pos gets output >= 0
        # neg gets output <= 0

        Timers.tic('check child feasible')
        # checking feasibility doesn't add too much time as it's done again layer for witnesses
        if self_gets_positive:
            neg.star.lpi.add_dense_row(row, -bias)
            neg.star.a_mat[i] = 0
            neg.star.bias[i] = 0 # reset the current bias as well

            #child_feasible = neg.star.lpi.is_feasible()
            child_feasible = True
        else:
            pos.star.lpi.add_dense_row(-row, bias)

            #child_feasible = pos.star.lpi.is_feasible()
            child_feasible = True

        Timers.toc('check child feasible')

        if not child_feasible:
            rv = None

            # if eager is false this can happen?

            ob = self.prefilter.output_bounds
            ob.branching_neurons = ob.branching_neurons[1:]

        else:
            rv = child
            
            if self_gets_positive:
                pos.star.lpi.add_dense_row(-row, bias)
            else:
                ### ASSIGN NEURON i OUTPUT
                # neg has 0 output
                neg.star.a_mat[i] = 0
                neg.star.bias[i] = 0 # reset the current bias as well

            # update branch_tuples
            child.branch_tuples = self.branch_tuples.copy()
            pos.branch_tuples.append((pos.cur_layer, i, True))
            neg.branch_tuples.append((neg.cur_layer, i, False))

            Timers.tic('prefilter_split_relu')

            depth = len(self.branch_tuples)
            child.prefilter = self.prefilter.split_relu(i, pos.star, neg.star, self_gets_positive, start_time, depth)

            assert child.prefilter.zono.mat_t is child.star.a_mat
            assert child.prefilter.zono.center is child.star.bias

            Timers.toc('prefilter_split_relu')

        Timers.toc('split_enumerate')

        return rv

    def do_first_relu_split(self, network, spec, start_time):
        '''
        do the first relu split for the current layer

        returns a new StarState from the split
        '''

        Timers.tic('do_first_relu_split')

        layer = network.layers[self.cur_layer]
        assert isinstance(layer, ReluLayer)
        assert self.prefilter.output_bounds is not None
        assert self.prefilter.output_bounds.branching_neurons.size > 0
        
        index = self.prefilter.output_bounds.branching_neurons[0]

        rv = self.split_enumerate(index, network, spec, start_time)

        Timers.toc('do_first_relu_split')

        return rv
