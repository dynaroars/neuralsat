'''
LP star implementation

Stanley Bak
'''

import numpy as np

from nnenum.lpinstance import LpInstance
from nnenum.util import Freezable
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum import kamenev

class LpStar(Freezable):
    '''generalized star set with lp constraints

    this is set up to efficiently encode: 
    linear transformation (using a_mat, a_rhs),
    constraints on initial variables (using csr / rhs)
    '''

    def __init__(self, a_mat, bias, box_bounds=None):
        assert a_mat is None or isinstance(a_mat, np.ndarray)
        assert bias is None or isinstance(bias, np.ndarray)
        
        self.a_mat = a_mat
        self.bias = bias

        # initialize lpi with no constraints
        self.lpi = None

        # for finding concrete counterexamples
        self.init_bm = None
        self.init_bias = None

        self.input_bounds_witnesses = None # a list of min, max for each dim

        # cached lp result
        self.last_lp_result = None

        self.num_lps = 0 # stat, number of lps solved

        # box_bounds may be None if we're initializing in lp_star.copy()
        if box_bounds is not None:
            if Settings.CONTRACT_LP_TRACK_WITNESSES:
                min_pt = np.array([a for a, b in box_bounds], dtype=float)
                max_pt = np.array([b for a, b in box_bounds], dtype=float)

                self.input_bounds_witnesses = []

                for _ in range(len(box_bounds)):
                    self.input_bounds_witnesses.append([min_pt, max_pt])
            
            self.lpi = LpInstance()

            for i, (lb, ub) in enumerate(box_bounds):
                self.lpi.add_double_bounded_cols([f"i{i}"], lb, ub)

            # assign init bm and init bias as well in this case
            if a_mat is None:
                self.init_bm = None
            else:
                self.init_bm = self.a_mat.copy()

            if bias is None:
                self.init_bias = None
            else:
                self.init_bias = self.bias.copy()

        self.freeze_attrs()

    def __str__(self):
        rv = "LpStar with a_mat:\n"

        for i, row in enumerate(self.a_mat):
            rv += f"a_mat row {i}: {row}\n"

        rv += "bias:\n"

        for i, val in enumerate(self.bias):
            rv += f"bias {i}: {val}\n"

        rv += "and LP:\n"
        rv += str(self.lpi)

        return rv

    def copy(self):
        'return a copy of this lpstar'

        rv = LpStar(a_mat=self.a_mat.copy(), bias=self.bias.copy())

        rv.lpi = LpInstance(self.lpi)

        Timers.tic('copy init bm bias')

        if self.input_bounds_witnesses is not None:
            rv.input_bounds_witnesses = []

            for a, b in self.input_bounds_witnesses:
                rv.input_bounds_witnesses.append([a, b])

        if self.init_bm is None:
            rv.init_bm = None
        else:
            rv.init_bm = self.init_bm.copy()

        if self.init_bias is None:
            rv.init_bias = None
        else:
            rv.init_bias = self.init_bias.copy()
        Timers.toc('copy init bm bias')

        return rv

    def to_full_input(self, compressed_input):
        'convert possibly compressed input to full input'

        #print(f".to_full_input, init_bm = {self.init_bm}\ninit_bias = {self.init_bias}")

        if self.init_bm is None:
            rv = compressed_input.copy()
        else:
            rv = np.dot(self.init_bm, compressed_input)

        if self.init_bias is not None:
            rv += self.init_bias

        if rv.dtype != self.a_mat.dtype:
            rv = rv.astype(self.a_mat.dtype)

        return rv
    
    def update_input_box_bounds_old(self, cur_box, should_skip, count_lps=True):
        '''get input box bounds on this set, compared with the current bounds using lp
        returns a list of 3-tuples for each of the bounds that was adjusted:
        (dim, lb, ub)
        '''

        dims = self.lpi.get_num_cols()
        rv = []
        tol = 1e-8

        Timers.tic("star.update_input_box_bounds_old")

        for dim in range(dims):

            if not should_skip[dim, 0]:
                # adjust lb
                vec = np.array([1 if i == dim else 0 for i in range(dims)], dtype=float)
                Timers.tic('lpi.minimize')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize')

                if self.input_bounds_witnesses:
                    self.input_bounds_witnesses[dim][0] = res

                if count_lps:
                    self.num_lps += 1

                if cur_box is None or abs(res[dim] - cur_box[dim][0]) >= tol:
                    rv.append([dim, res[dim], np.inf])

            if not should_skip[dim, 1]:
                # adjust ub
                vec = np.array([-1 if i == dim else 0 for i in range(dims)], dtype=float)

                Timers.tic('lpi.minimize')
                res = self.lpi.minimize(vec)
                Timers.toc('lpi.minimize')

                if self.input_bounds_witnesses:
                    self.input_bounds_witnesses[dim][1] = res

                if count_lps:
                    self.num_lps += 1

                if cur_box is None or abs(res[dim] - cur_box[dim][1]) >= tol:
                    if not rv or rv[-1][0] != dim:
                        rv.append([dim, -np.inf, res[dim]])
                    else:
                        rv[-1][2] = res[dim]

        Timers.toc("star.update_input_box_bounds_old")

        return rv

    def check_input_box_bounds_slow(self):
        '''
        run a sanity check that the input box bounds witnesses are correct
        this uses LP and is slow, so it's meant to help with debugging
        '''

        print("Warning: check_input_box_bounds_slow called")

        cur_bounds = self.get_input_box_bounds()

        dims = self.lpi.get_num_cols()
        should_skip = np.zeros((dims, 2), dtype=bool)
        correct_bounds_list = self.update_input_box_bounds_old(None, should_skip)

        for d, min_val, max_val in correct_bounds_list:
            assert abs(min_val - cur_bounds[d][0]) < 1e-5, f"dim {d} min was {cur_bounds[d][0]}, should be {min_val}"
            assert abs(max_val - cur_bounds[d][1]) < 1e-5, f"dim {d} max was {cur_bounds[d][1]}, should be {max_val}"

    def get_input_box_bounds(self):
        'gets the input box bounds from witnesses'

        rv = []

        if self.input_bounds_witnesses is not None:
            dims = self.lpi.get_num_cols()

            assert len(self.input_bounds_witnesses) == dims, \
                f"dims:{dims}, num witneses: {len(self.input_bounds_witnesses)}"

            for d in range(dims):
                min_wit, max_wit = self.input_bounds_witnesses[d]

                if min_wit[d] > max_wit[d]:
                    # can happen due to numerical error
                    assert min_wit[d] - 1e-5 < max_wit[d]

                    mid = (min_wit[d] + max_wit[d]) / 2
                    min_wit[d] = max_wit[d] = mid
                
                rv.append((min_wit[d], max_wit[d]))
        else:
            Timers.tic('full input bounds')
            should_skip = np.zeros((dims, 2), dtype=bool)
            correct_bounds_list = self.update_input_box_bounds_old(None, should_skip)

            for _d, lb, ub in correct_bounds_list:
                rv.append((lb, ub))

            Timers.toc('full input bounds')

        return rv

    def update_input_box_bounds(self, hyperplane_vec_list, rhs_list, count_lps=True):
        '''update the input box bounds on the set after some constaints are added

        hyperplane_vec_list and rhs_list (can also be individual items) 
        define the new constraint that was added (optimized bounds using witnesses)
        '''

        dims = self.lpi.get_num_cols()
        should_skip = np.ones((dims, 2), dtype=bool)

        # TODO: check if cur_box is the same as get_input_box_bounds
        cur_box = self.get_input_box_bounds()

        if not isinstance(hyperplane_vec_list, list):
            hyperplane_vec_list = [hyperplane_vec_list]

            assert not isinstance(rhs_list, list)
            rhs_list = [rhs_list]

        if self.input_bounds_witnesses is not None:
            for d in range(dims):
                min_wit, max_wit = self.input_bounds_witnesses[d]

                for hyperplane_vec, rhs in zip(hyperplane_vec_list, rhs_list):
                    if should_skip[d, 0] and np.dot(hyperplane_vec, min_wit) > rhs:
                        should_skip[d, 0] = False

                    if should_skip[d, 1] and np.dot(hyperplane_vec, max_wit) > rhs:
                        should_skip[d, 1] = False

        # print(self.lpi)

        if Settings.CONTRACT_LP_OPTIMIZED and cur_box is not None:
            rv = self.update_input_box_bounds_new(cur_box, should_skip, count_lps)
        else:
            rv = self.update_input_box_bounds_old(cur_box, should_skip, count_lps)

        return rv

    def update_input_box_bounds_new(self, cur_box, should_skip, count_lps=True):
        '''compute new input box bounds on this set, compared with the current bounds using lp

        returns a list of 3-tuples for each of the bounds that was adjusted:
        (dim, lb, ub)

        note: lb may be -np.inf and ub may be np.inf
        '''

        assert cur_box is not None

        Timers.tic("star.update_input_box_bounds_new")

        dims = self.lpi.get_num_cols()
        rv = []

        tol = 1e-8
        num_lps = 0

        vec = np.ones((dims, ), dtype=float)

        for d in range(dims):
            if should_skip[d, 0]:
                vec[d] = 0

        while True:
            Timers.tic('lpi.minimize pre1')
            res = self.lpi.minimize(vec)
            num_lps += 1
            Timers.toc('lpi.minimize pre1')

            skipped_all = True
            skipped_some = False

            for dim in range(dims):
                if should_skip[dim, 0]:
                    continue

                if abs(res[dim] - cur_box[dim][0]) < tol:

                    if self.input_bounds_witnesses is not None:
                        self.input_bounds_witnesses[dim][0] = res
                        
                    vec[dim] = 0
                    should_skip[dim, 0] = True
                    skipped_some = True
                else:
                    skipped_all = False

            if skipped_all or not skipped_some:
                break

        for dim in range(dims):
            
            # adjust lb
            if not should_skip[dim, 0]:
                # possible optimization: this may be solving an extra lp if the above loops only involved a single dim
                
                vec = np.zeros((dims, ), dtype=float)
                vec[dim] = 1
                
                Timers.tic('lpi.minimize post1')
                res = self.lpi.minimize(vec)
                min_val = res[dim]
                num_lps += 1
                Timers.toc('lpi.minimize post1')

                if self.input_bounds_witnesses is not None:
                    self.input_bounds_witnesses[dim][0] = res

                if abs(min_val - cur_box[dim][0]) >= tol:
                    rv.append([dim, min_val, np.inf])

        # other side
        vec = -1 * np.ones((dims, ), dtype=float)

        for d in range(dims):
            if should_skip[d, 1]:
                vec[d] = 0
        
        while True:
            Timers.tic('lpi.minimize pre2')
            res = self.lpi.minimize(vec)
            num_lps += 1
            Timers.toc('lpi.minimize pre2')

            skipped_all = True
            skipped_some = False

            for dim in range(dims):
                if should_skip[dim, 1]:
                    continue

                if abs(res[dim] - cur_box[dim][1]) < tol:
                    if self.input_bounds_witnesses is not None:
                        self.input_bounds_witnesses[dim][1] = res

                        vec[dim] = 0
                    should_skip[dim, 1] = True
                    skipped_some = True
                else:
                    skipped_all = False

            if skipped_all or not skipped_some:
                break

        for dim in range(dims):
            # adjust ub
            if not should_skip[dim, 1]:
                vec = np.zeros((dims, ), dtype=float)
                vec[dim] = -1

                Timers.tic('lpi.minimize post2')
                res = self.lpi.minimize(vec)
                max_val = res[dim]
                num_lps += 1
                Timers.toc('lpi.minimize post2')

                if self.input_bounds_witnesses is not None:
                    self.input_bounds_witnesses[dim][1] = res

                if abs(max_val - cur_box[dim][1]) >= tol:

                    found = False

                    for index, (rv_dim, _rv_min, _rv_max) in enumerate(rv):
                        if rv_dim == dim:
                            found = True
                            rv[index][2] = max_val
                            break

                    if not found:
                        rv.append([dim, -np.inf, max_val])

        if count_lps:
            self.num_lps += num_lps

        Timers.toc("star.update_input_box_bounds_new")

        return rv

    def minimize_output(self, output_index, maximize=False):
        '''
        get the output value when one of the outputs is minimized (or maximized)

        if stop_at_zero is set, this will terminate the search once zero is crossed

        if you want the (input, output) pair to produce this output, use consutrct_last_io()
        '''

        Timers.tic('minimize_output')

        if self.a_mat.size == 0:
            value = self.bias
        else:
            row = self.a_mat[output_index]

            if maximize:
                row = -1 * row

            self.last_lp_result = lp_result = self.lpi.minimize(row)
            self.num_lps += 1

            num_init_vars = self.a_mat.shape[1]
            assert len(lp_result) == num_init_vars

            # single row
            value = self.a_mat[output_index].dot(lp_result) + self.bias[output_index]

        Timers.toc('minimize_output')

        return value

    def construct_last_io(self):
        '''construct the last concrete input/output pair from the optimization performed when minimize_output was called

        note that the input will be the compressed input if input space is not full dimensional
        '''

        Timers.tic('construct_last_io')
        
        i = self.last_lp_result
        
        o = np.dot(self.a_mat, i) + self.bias

        Timers.toc('construct_last_io')

        return [i, o]

    def minimize_vec(self, vec, return_io=False, fail_on_unsat=True):
        '''optimize over this set

        vec is the vector of outputs we're optimizing over, None means use zero vector

        if return_io is true, returns a tuple (input, output); otherwise just output
        note that the cinput will be the compressed input if input space is not full dimensional

        returns all the outputs (coutput) if return_io=False, else (cinput, coutput)
        '''

        Timers.tic('star.minimize_vec')

        dtype = float if self.a_mat is None else self.a_mat.dtype

        if self.a_mat.size == 0:
            rv = self.bias
            self.last_lp_result = lp_result = np.array([], dtype=dtype)
        else:
            assert len(self.a_mat.shape) == 2, f"a_mat shape was {self.a_mat.shape}"

            if vec is None:
                vec = np.zeros((self.a_mat.shape[0],), dtype=dtype)

            assert len(vec) == self.a_mat.shape[0], f"minimize called with vector with {len(vec)} elements, " + \
                f"but set has {self.a_mat.shape[0]} outputs"

            #Timers.tic('setup')
            assert isinstance(vec, np.ndarray)

            lp_vec = np.dot(self.a_mat.T, vec)
            #lp_vec = vec.T.dot(self.a_mat).T

            num_init_vars = self.a_mat.shape[1]
            lp_vec.shape = (len(lp_vec),)
            #Timers.toc('setup')

            #Timers.tic('lpi.minimize')
            lp_result = self.lpi.minimize(lp_vec, fail_on_unsat=fail_on_unsat)

            if lp_result is None:
                rv = None
            else:
                if lp_result.dtype != dtype:
                    lp_result = lp_result.astype(dtype)

                self.last_lp_result = lp_result

                self.num_lps += 1
                #Timers.toc('lpi.minimize')
                assert len(lp_result) == num_init_vars

                #print("--------")
                #print(f"lp_result: {lp_result}")

                #Timers.tic('a_mat mult')
                rv = np.dot(self.a_mat, lp_result) + self.bias
                #Timers.toc('a_mat mult')

        # return input as well
        if rv is not None and return_io:
            rv = [lp_result, rv]

        Timers.toc('star.minimize_vec')

        return rv

    def verts(self, xdim=0, ydim=1, epsilon=1e-7):
        'get a 2-d projection of this lp_star'

        dims = self.a_mat.shape[0]

        if isinstance(xdim, int):
            assert 0 <= xdim < dims, f"xdim {xdim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[xdim] = 1
            xdim = vec
        else:
            assert xdim.size == dims

        if isinstance(ydim, int):
            assert 0 <= ydim < dims, f"ydim {ydim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[ydim] = 1
            ydim = vec
        else:
            assert ydim.size == dims

        def supp_point_func(vec2d):
            'maximize a support function direction'

            Timers.tic('supp_point_func')

            # use negative to maximize
            lpdir = -vec2d[0] * xdim + -vec2d[1] * ydim

            res = self.minimize_vec(lpdir)

            Timers.toc('supp_point_func')

            # project onto x and y
            resx = np.dot(xdim, res)
            resy = np.dot(ydim, res)

            return np.array([resx, resy], dtype=float)

        Timers.tic('kamenev.get_verts')
        verts = kamenev.get_verts(2, supp_point_func, epsilon=epsilon)
        Timers.toc('kamenev.get_verts')

        #assert np.allclose(verts[0], verts[-1])
        
        return verts

    def box_verts(self, xdim=0, ydim=1):
        'get box bounds of a 2d projection of the star'

        dims = self.a_mat.shape[0]

        if isinstance(xdim, int):
            assert 0 <= xdim < dims, f"xdim {xdim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[xdim] = 1
            xdim = vec
        else:
            assert xdim.size == dims

        if isinstance(ydim, int):
            assert 0 <= ydim < dims, f"ydim {ydim} out of bounds for star with {dims} dims"
            vec = np.zeros(dims, dtype=float)
            vec[ydim] = 1
            ydim = vec
        else:
            assert ydim.size == dims

        def supp_point_func(vec2d):
            'maximize a support function direction'

            Timers.tic('supp_point_func')

            # use negative to maximize
            lpdir = -vec2d[0] * xdim + -vec2d[1] * ydim

            res = self.minimize_vec(lpdir)

            Timers.toc('supp_point_func')

            # project onto x and y
            resx = np.dot(xdim, res)
            resy = np.dot(ydim, res)

            return np.array([resx, resy], dtype=float)

        # box bounds
        x0 = supp_point_func([1, 0])[0]
        x1 = supp_point_func([-1, 0])[0]
        y0 = supp_point_func([0, 1])[1]
        y1 = supp_point_func([0, -1])[1]

        verts = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        
        return verts

    def execute_relus_overapprox(self, layer_num, layer_bounds, split_indices, zero_indices):
        '''
        run the relu part of the star update for the *entire* layer, using overapproximation
        '''

        Timers.tic('execute_relus_overapprox')

        # overapprox can set star to None if no LP optimization is needed
        num_outputs = self.a_mat.shape[0]
        assert len(layer_bounds) == num_outputs, f"outputs is {num_outputs}, but num " + \
            f"layer bounds {len(layer_bounds)}"

        #new_generators_bm = [[] for _ in range(num_outputs)]
        new_generators_bm = np.zeros((num_outputs, split_indices.size), dtype=self.a_mat.dtype)

        self.a_mat[zero_indices, :] = 0
        self.bias[zero_indices] = 0

        for i in split_indices:
            lb, ub = layer_bounds[i]
            self.split_overapprox(layer_num, new_generators_bm, i, lb, ub)

        Timers.tic('overapprox bm update')
        num_zeros = self.lpi.get_num_cols() - self.a_mat.shape[1]

        if num_zeros > 0:
            #for bm_row in new_generators_bm:
            #    bm_row += [0] * (num_zeros - len(bm_row))

            self.a_mat = np.hstack([self.a_mat, new_generators_bm])
            assert self.a_mat.shape[1] == self.lpi.get_num_cols()

        Timers.toc('overapprox bm update')

        Timers.toc('execute_relus_overapprox')

    def split_overapprox(self, layer_num, new_generators_bm, i, lb, ub):
        '''helper for execute_relus_overapprox

        split a ReLU using a star overapproximation'''

        Timers.tic('split_overapprox')

        # make a new variable y for the output
        self.lpi.add_positive_cols([f'y{layer_num}_{i}'])
        num_cols = self.lpi.get_num_cols()
        num_zeros = num_cols - self.a_mat.shape[1] - 1

        #zero_row = np.zeros((self.star.a_mat.shape[1],))

        # create 3 constraints for new variable
        # (1) y >= 0 === -y <= 0
        # this constraint is automatically added in lp_star for all non-cur variables

        # (2) y >= x[i] === x[i] - y <= 0
        # x[i] equals row i in the basis matrix (also the bias on the rhs)

        row = np.zeros((num_cols,), dtype=self.a_mat.dtype)
        a_mat_width = self.a_mat.shape[1]

        assert a_mat_width <= num_cols, f"a_mat_width: {a_mat_width}, num_cols: {num_cols}"

        row[:a_mat_width] = self.a_mat[i, :]
        row[-1] = -1
        self.lpi.add_dense_row(row, -self.bias[i])

        # (3) y <= ub*(x[i]-lb) / (ub - lb) === y - ub*x[i] / (ub - lb) - (ub*(-lb) / (ub - lb)) <= 0
        # === y - ub(ub - lb) * x[i] <= ub*(-lb) / (ub - lb)
        # x[i] equals row i in the basis matrix
        factor = ub / (ub - lb)
        row = np.zeros((num_cols,), dtype=self.a_mat.dtype)
        row[:self.a_mat.shape[1]] = -1 * factor * self.a_mat[i]
        row[-1] = 1
        rhs = -lb * factor + self.bias[i] * factor
        self.lpi.add_dense_row(row, rhs)

        # reset the current bias
        # the rhs of the current variable is not referenced by other constraints (constraints never ref rhs)
        self.bias[i] = 0

        # ReLU case, introduce new variable
        self.a_mat[i] = 0

        new_generators_bm[i, num_zeros] = 1

        Timers.toc('split_overapprox')
