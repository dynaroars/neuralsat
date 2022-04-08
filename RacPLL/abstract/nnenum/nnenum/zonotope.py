'''
Zonotope nnenum implementation

Stanley Bak
'''

import numpy as np

from nnenum.util import Freezable
from nnenum.timerutil import Timers
from nnenum import kamenev
from nnenum.settings import Settings

def zono_from_compressed_init_box(init_bm, init_bias, init_box):
    '''create a Zonotope from a compressed init box (deep copy)

    parameters are those obtained from nnenum.util.compress_init_box()
    '''

    center = init_bias.copy()

    generators = []
    init_bounds = []

    dtype = init_bm.dtype

    for index, (lb, ub) in enumerate(init_box):
        vec = np.array([1 if d == index else 0 for d in range(len(init_box))], dtype=dtype)
        generators.append(vec)
        init_bounds.append((lb, ub))

    generators = np.array(generators, dtype=dtype)
    generators.shape = (len(init_box), len(generators))

    gen_mat_t = np.dot(init_bm, generators.transpose())

    return Zonotope(center, gen_mat_t, init_bounds)

class Zonotope(Freezable):
    'zonotope class'

    def __init__(self, center, gen_mat_t, init_bounds=None):
        '''
        gen_mat_t has one generator per COLUMN

        init_bounds for a traditional zonotope is [-1, 1]
        '''

        if center is not None:
            self.dtype = center.dtype
            assert isinstance(center, np.ndarray)
            assert len(center.shape) == 1 or center.shape[0] == 1, f"Expected 1-d center, got {center.shape}"
        else:
            assert init_bounds is not None
            self.dtype = init_bounds.dtype
        
        self.center = center # note: shallow copy

        if gen_mat_t is None:
            assert init_bounds is not None
        else:
            assert len(gen_mat_t.shape) == 2, f"expected 2-d gen_mat_t, got {gen_mat_t.shape}"
            assert isinstance(gen_mat_t, np.ndarray), f"gen_mat_t was {type(gen_mat_t)}"
        
            if gen_mat_t.size > 0:
                assert len(self.center) == gen_mat_t.shape[0], f"center has {len(self.center)} dims but " + \
                    f"gen_mat_t has {gen_mat_t.shape[0]} entries per column (rows)"

                if init_bounds is None:
                    init_bounds = [(-1, 1) for _ in range(gen_mat_t.shape[0])]

        if isinstance(init_bounds, np.ndarray):
            # convert to list of tuples
            assert init_bounds.shape[1] == 2
            
            l = []
            for row in init_bounds:
                l.append((row[0], row[1]))

            init_bounds = l

        assert isinstance(init_bounds, list)
        assert isinstance(init_bounds[0], tuple)

        assert self.dtype in [np.float32, np.float64], \
            f"zonotope dtype was {self.dtype}"

        self.mat_t = gen_mat_t # note: shallow copy
        self.pos1_gens = None #np.array(self.mat_t.shape[1] * [1.0])
        self.neg1_gens = None

        self.init_bounds = init_bounds # no copy either, done externally

        self.init_bounds_nparray = None

        self.freeze_attrs()

    def __str__(self):
        return f"[Zonotope with center {self.center} and generator matrix_t:\n{self.mat_t}" + \
            f" and init_bounds: {self.init_bounds}"

    def deep_copy(self):
        'return a deep copy of this zonotope'

        return Zonotope(self.center.copy(), self.mat_t.copy(), self.init_bounds.copy())

    def get_domain_center(self):
        'get the center of the domain box'

        rv = [(ib[0] + ib[1]) / 2 for ib in self.init_bounds]

        return rv

    def contract_lp(self, star, hyperplane_vec, rhs):
        '''do lp zonotope contraction

        returns True if domain was tightened
        '''

        Timers.tic("contract_lp")

        #cur_box = self.init_bounds
        new_bounds_list = star.update_input_box_bounds(hyperplane_vec, rhs, count_lps=True)

        for dim, lb, ub in new_bounds_list:
            self.update_init_bounds(dim, (lb, ub))

        Timers.toc("contract_lp")

        return new_bounds_list

    def update_init_bounds(self, i, new_bounds, skip_check=False):
        'update init bounds in the zonotope'
        
        assert isinstance(new_bounds, tuple)
        assert new_bounds[0] <= new_bounds[1], f"new bounds was: {new_bounds}"

        self.init_bounds_nparray = None

        ep = 1e-4

        if skip_check:
            lb = new_bounds[0]
        elif new_bounds[0] != -np.inf:
            assert new_bounds[0] + ep >= self.init_bounds[i][0], f"new lower bound ({new_bounds[0]} was worse " + \
                                                             f"then before {self.init_bounds[i][0]}"

            lb = max(self.init_bounds[i][0], new_bounds[0])
        else:
            lb = self.init_bounds[i][0]

        if skip_check:
            ub = new_bounds[1]
        elif new_bounds[1] != np.inf:
            assert new_bounds[1] - ep <= self.init_bounds[i][1], f"new upper bound ({new_bounds[1]}) was worse" + \
                                                                 f"then before {self.init_bounds[i][1]}"

            ub = min(self.init_bounds[i][1], new_bounds[1])
        else:
            ub = self.init_bounds[i][1]

        if lb > ub:
            lb = ub = (lb + ub) / 2.0

        self.init_bounds[i] = (lb, ub)

        # update these
        if self.neg1_gens is not None:
            self.neg1_gens[i] = lb
            self.pos1_gens[i] = ub

    def maximize(self, vector):
        'get the maximum point of the zonotope in the passed-in direction'

        Timers.tic('zonotope.maximize')

        rv = self.center.copy()

        # project vector (a generator) onto row, to check if it's positive or negative
        #res_vec = np.dot(self.mat_t.transpose(), vector) # slow? since we're taking transpose
        res_vec = np.dot(vector, self.mat_t)

        # todo: I think this can be vectorized using np.where
        for res, row, ib in zip(res_vec, self.mat_t.transpose(), self.init_bounds):
            factor = ib[1] if res >= 0 else ib[0]

            rv += factor * row

        Timers.toc('zonotope.maximize')

        return rv

    def minimize_val(self, vector):
        '''get the minimum value of the zonotope projected onto the passed-in direction

        similar to zonotope.maximize but slightly faster
        '''

        Timers.tic('zonotope.minimize_val')

        rv = self.center.dot(vector)

        # project vector (a generator) onto row, to check if it's positive or negative
        #res_vec = np.dot(self.mat_t.transpose(), vector) # slow? since we're taking transpose
        res_vec = np.dot(vector, self.mat_t)

        #Timers.tic('loop')
        #for res, ib in zip(res_vec, self.init_bounds):
        #    factor = ib[1] if res <= 0 else ib[0]

        #    rv += factor * res

        #Timers.toc('loop')

        if self.init_bounds_nparray is None:
            self.init_bounds_nparray = np.array(self.init_bounds, dtype=self.dtype)

        ib = self.init_bounds_nparray
        
        res = np.where(res_vec <= 0, ib[:, 1], ib[:, 0])

        rv += res.dot(res_vec)

        Timers.toc('zonotope.minimize_val')

        return rv

    def box_bounds(self):
        '''compute box bounds for the zonotope

        returns bounds
        '''

        Timers.tic('zono.box_bounds')

        mat_t = self.mat_t
        size = self.center.size

        # pos_1_gens may need to be updated if matrix size changed due to assignment
        if self.pos1_gens is None or self.pos1_gens.shape[0] != self.mat_t.shape[1]:
            self.neg1_gens = np.array([i[0] for i in self.init_bounds], dtype=self.dtype)
            self.pos1_gens = np.array([i[1] for i in self.init_bounds], dtype=self.dtype)

            assert self.pos1_gens.shape[0] == self.mat_t.shape[1]

        pos_mat = np.clip(mat_t, 0, np.inf)
        neg_mat = np.clip(mat_t, -np.inf, 0)

        pos_pos = np.dot(self.pos1_gens, pos_mat.T)
        neg_neg = np.dot(self.neg1_gens, neg_mat.T)
        pos_neg = np.dot(self.pos1_gens, neg_mat.T)
        neg_pos = np.dot(self.neg1_gens, pos_mat.T)

        rv = np.zeros((size, 2), dtype=self.dtype)
        rv[:, 0] = self.center + pos_neg + neg_pos
        rv[:, 1] = self.center + pos_pos + neg_neg
        
        Timers.toc('zono.box_bounds')

        return rv

    def get_single_output_bounds(self, index):
        'get the lb and ub for a single output'

        lb = ub = self.center[index]

        if self.mat_t.size > 0:
            for col in range(self.mat_t.shape[1]):
                val = self.mat_t[index, col]

                option1 = val * self.init_bounds[col][0]
                option2 = val * self.init_bounds[col][1]

                if option1 < option2:
                    lb += option1
                    ub += option2
                else:
                    lb += option2
                    ub += option1

        return lb, ub

    def update_output_bounds(self, layer_bounds, update_indices):
        '''
        update the passed-in bounds to the values in the zonotope outer approximation. 

        this only updates the indices in the passed-in update_indices array

        returns split_indices (subset of update_indices)
        '''

        Timers.tic('zono.update_output_bounds')

        split_indices = []

        if self.mat_t.size == 0:
            # no generators
            for i in update_indices:
                layer_bounds[i, 0] = layer_bounds[i, 1] = self.center[i]
        else:

            # pos_1_gens may need to be updated if matrix size changed due to assignment
            if self.pos1_gens is None or self.pos1_gens.shape[0] != self.mat_t.shape[1]:
                self.neg1_gens = np.array([i[0] for i in self.init_bounds], dtype=self.dtype)
                self.pos1_gens = np.array([i[1] for i in self.init_bounds], dtype=self.dtype)

                assert self.pos1_gens.shape[0] == self.mat_t.shape[1]

            mat_t = self.mat_t[update_indices, :]

            # is a copy here better???

            # possible optimization: use vectorization as in box_bounds
            # downside: we need to assign in-place here to layer_bounds
            if mat_t.size > 0:
                tol = Settings.SPLIT_TOLERANCE
                
                pos_mat = np.clip(mat_t, 0, np.inf)
                neg_mat = np.clip(mat_t, -np.inf, 0)

                pos_pos = np.dot(self.pos1_gens, pos_mat.T)
                neg_neg = np.dot(self.neg1_gens, neg_mat.T)
                pos_neg = np.dot(self.pos1_gens, neg_mat.T)
                neg_pos = np.dot(self.neg1_gens, pos_mat.T)

                for i, pp_row, nn_row, pn_row, np_row in zip(update_indices, pos_pos, neg_neg, pos_neg, neg_pos):
                    ub = pp_row + nn_row
                    lb = pn_row + np_row
                    c = self.center[i]

                    layer_bounds[i, 0] = c + lb
                    layer_bounds[i, 1] = c + ub
                    
                    if c + lb < -tol and tol < c + ub:
                        split_indices.append(i)

        Timers.toc('zono.update_output_bounds')

        return np.array(split_indices, dtype=int)

    def contract_domain(self, hyperplane_vec, rhs):
        'this adjusts bounds based on contract-simple strategy'

        tuple_list = self.contract_domain_new(hyperplane_vec, rhs)

        for d, lb, ub in tuple_list:
            self.update_init_bounds(d, (lb, ub))

    def contract_domain_new(self, hyperplane_vec, rhs):
        '''intersect the initial box with the given halfspace, trying to shrink it

        returns a list of 3-tuples for each of the bounds that was adjusted:
        (dim, lb, ub)
        '''

        rv = []

        Timers.tic('contract domain')

        if not isinstance(hyperplane_vec, np.ndarray):
            hyperplane_vec = np.array(hyperplane_vec, dtype=self.dtype)

        dims = len(self.init_bounds)

        assert len(hyperplane_vec) == dims, f"dims in init_bounds is {dims}, hyperplane dims is {len(hyperplane_vec)}"
        ib = self.init_bounds

        sat_corner_list = [ib[d][0] if hyperplane_vec[d] > 0 else ib[d][1] for d in range(dims)]
        unsat_corner_list = [ib[d][1] if hyperplane_vec[d] > 0 else ib[d][0] for d in range(dims)]

        # construct matrix of sat corners
        sat_corner_matrix = np.array([sat_corner_list] * dims, dtype=self.dtype)

        for d in range(dims):
            sat_corner_matrix[d, d] = unsat_corner_list[d]

        # do it as matrix-vec mult
        dot_res = sat_corner_matrix.dot(hyperplane_vec)
        
        for d in range(dims):
            # make sure it's nonzero

            # todo: test the effect of removing this? division by k... better to keep here
            if abs(hyperplane_vec[d]) < 1e-6:
                continue

            #old_sat_corner_d = sat_corner[d]
            #sat_corner[d] = unsat_corner[d]

            # sat_corner is now potentially unsat
            #lhs = np.dot(sat_corner, hyperplane_vec)
            lhs = dot_res[d]
            
            if lhs > rhs:
                # constraint is tighter! find the new bound
                #sat_corner[d] = 0
                #tot = np.dot(sat_corner, hyperplane_vec)
                tot = lhs - unsat_corner_list[d] * hyperplane_vec[d]

                k = hyperplane_vec[d]
                solved_rhs = (rhs - tot) / k

                if k > 0:
                    # change ub to solved_rhs
                    rv.append((d, ib[d][0], solved_rhs))
                else:
                    # change lb to solved_rhs
                    rv.append((d, solved_rhs, ib[d][1]))

            # restore sat_corner[d]
            #sat_corner[d] = old_sat_corner_d
                        
        Timers.toc('contract domain')

        return rv

    def verts(self, xdim=0, ydim=1, epsilon=1e-7):
        'get verts'

        dims = len(self.center)

        assert 0 <= xdim < dims, f"xdim was {xdim}, but num zonotope dims was {dims}"
        assert 0 <= ydim < dims, f"ydim was {ydim}, but num zonotope dims was {dims}"

        def max_func(vec):
            'projected max func for kamenev'

            max_vec = [0] * dims
            max_vec[xdim] += vec[0]
            max_vec[ydim] += vec[1]
            max_vec = np.array(max_vec, dtype=self.dtype)

            res = self.maximize(max_vec)

            return np.array([res[xdim], res[ydim]], dtype=self.dtype)

        return kamenev.get_verts(2, max_func, epsilon=epsilon)

