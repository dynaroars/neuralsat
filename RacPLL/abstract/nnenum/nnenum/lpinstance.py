'''
Stanley Bak
May 2018
GLPK python interface using swiglpk
'''

import sys
import math
import time

import numpy as np
from scipy.sparse import csr_matrix

from termcolor import colored

import swiglpk as glpk
from nnenum.util import Freezable
from nnenum.timerutil import Timers
from nnenum.settings import Settings

def get_lp_params(alternate_lp_params=False):
    'get the lp params object'

    if not hasattr(get_lp_params, 'obj'):
        params = glpk.glp_smcp()
        glpk.glp_init_smcp(params)

        #params.msg_lev = glpk.GLP_MSG_ERR
        params.msg_lev = glpk.GLP_MSG_ERR
        params.meth = glpk.GLP_PRIMAL if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_DUAL

        params.tm_lim = int(Settings.GLPK_TIMEOUT * 1000)
        params.out_dly = 2 * 1000 # start printing to terminal delay
        
        get_lp_params.obj = params

        # make alternative params
        params2 = glpk.glp_smcp()
        glpk.glp_init_smcp(params2)
        params2.meth = glpk.GLP_DUAL if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_PRIMAL
        params2.msg_lev = glpk.GLP_MSG_ON

        params2.tm_lim = int(Settings.GLPK_TIMEOUT * 1000)
        params2.out_dly = 1 * 1000 # start printing to terminal status after 1 secs
        
        get_lp_params.alt_obj = params2
        
    if alternate_lp_params:
        #glpk.glp_term_out(glpk.GLP_ON)
        rv = get_lp_params.alt_obj
    else:
        #glpk.glp_term_out(glpk.GLP_OFF)
        rv = get_lp_params.obj

    return rv

class LpInstance(Freezable):
    'Linear programming wrapper using glpk (through swiglpk python interface)'

    lp_time_limit_sec = 15.0

    def __init__(self, other_lpi=None):
        'initialize the lp instance'

        self.lp = glpk.glp_create_prob() # pylint: disable=invalid-name
        
        if other_lpi is None:
            # internal bookkeeping
            self.names = [] # column names

            # setup lp params
        else:
            # initialize from other lpi
            self.names = other_lpi.names.copy()
                
            Timers.tic('glp_copy_prob')
            glpk.glp_copy_prob(self.lp, other_lpi.lp, glpk.GLP_OFF)
            Timers.toc('glp_copy_prob')

        self.freeze_attrs()

    def __del__(self):
        if hasattr(self, 'lp') and self.lp is not None:

            if not isinstance(self.lp, tuple):
                glpk.glp_delete_prob(self.lp)
                
            self.lp = None

    def serialize(self):
        'serialize self.lp from a glpk instance into a tuple'

        Timers.tic('serialize')

        # get constraints as csr matrix
        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()

        inds_row = SwigArray.get_int_array(lp_cols + 1)
        vals_row = SwigArray.get_double_array(lp_cols + 1)

        data = []
        glpk_indices = []
        indptr = [0]

        for row in range(lp_rows):
            got_len = glpk.glp_get_mat_row(self.lp, row + 1, inds_row, vals_row)

            for i in range(1, got_len+1):
                data.append(vals_row[i])
                glpk_indices.append(inds_row[i])

            indptr.append(len(data))
            
        # rhs
        rhs = []
        
        for row in range(lp_rows):
            assert glpk.glp_get_row_type(self.lp, row + 1) == glpk.GLP_UP

            rhs.append(glpk.glp_get_row_ub(self.lp, row + 1))

        col_bounds = self._get_col_bounds()

        # remember to free lp object before overwriting with tuple
        glpk.glp_delete_prob(self.lp)
        self.lp = (data, glpk_indices, indptr, rhs, col_bounds)

        Timers.toc('serialize')

    # removed this, as get_col_bounds shouldn't be used externally
    #def set_col_bounds(self, col, lb, ub):
    #    'set double-bounded column bounds'

    #    col_type = glpk.glp_get_col_type(self.lp, col + 1)

    #    if col_type != glpk.GLP_DB:
    #        print(f"Warning: Contract col {col} to {lb, ub} skipped (col type is not GLP_DB):\n{self}")
    #    else:
    #        glpk.glp_set_col_bnds(self.lp, col + 1, glpk.GLP_DB, lb, ub)  # double-bounded variable

    def _get_col_bounds(self):
        '''get column bounds

        for external use use star's get_input_bounds which may be tighter
        '''

        lp_cols = self.get_num_cols()

        # column lower and upper bounds
        col_bounds = []
        
        for col in range(lp_cols):
            col_type = glpk.glp_get_col_type(self.lp, col + 1)

            ub = np.inf
            lb = -np.inf

            if col_type == glpk.GLP_DB:
                ub = glpk.glp_get_col_ub(self.lp, col + 1)
                lb = glpk.glp_get_col_lb(self.lp, col + 1)
            elif col_type == glpk.GLP_LO:
                lb = glpk.glp_get_col_lb(self.lp, col + 1)
            elif col_type == glpk.GLP_FX:
                lb = ub = glpk.glp_get_col_lb(self.lp, col + 1)
            else:
                assert col_type == glpk.GLP_FR, "unsupported col type in _get_col_bounds()"

            col_bounds.append((lb, ub))

        return col_bounds

    def deserialize(self):
        'deserialize self.lp from a tuple into a glpk_instance'

        assert isinstance(self.lp, tuple)

        Timers.tic('deserialize')

        data, glpk_indices, indptr, rhs, col_bounds = self.lp

        self.lp = glpk.glp_create_prob()

        # add cols
        names = self.names
        self.names = [] # adding columns populates self.names

        num_cols = len(col_bounds)

        for i, (lb, ub) in enumerate(col_bounds):
            name = names[i]

            if ub == np.inf:
                if lb == -np.inf:
                    # free variable
                    self.add_cols([name])
                else:
                    assert lb == 0

                    self.add_positive_cols([name])
            else:
                self.add_double_bounded_cols([name], lb, ub)

        # add rows
        num_rows = len(rhs)
        self.add_rows_less_equal(rhs)

        # set constraints
        shape = (num_rows, num_cols)
        self.set_constraints_csr(data, glpk_indices, indptr, shape)

        Timers.toc('deserialize')

    def _column_names_str(self):
        'get the line in __str__ for the column names'

        rv = "    "
        dbl_max = sys.float_info.max

        for col, name in enumerate(self.names):
            name = self.names[col]

            lb = glpk.glp_get_col_lb(self.lp, col + 1)
            ub = glpk.glp_get_col_ub(self.lp, col + 1)

            if lb != -dbl_max or ub != dbl_max:
                name = "*" + name

            name = name.rjust(6)[:6] # fix width to exactly 6

            rv += name + " "

        rv += "\n"
        
        return rv

    def _opt_dir_str(self, zero_print):
        'get the optimization direction line for __str__'

        lp = self.lp
        cols = self.get_num_cols()
        rv = "min "

        for col in range(1, cols + 1):
            val = glpk.glp_get_obj_coef(lp, col)

            num = f"{val:.6f}"
            num = num.rjust(6)[:6] # fix width to exactly 6
            
            if val == 0:
                rv += zero_print(num) + " "
            else:
                rv += num + " "

        rv += "\n"
        
        return rv

    def _col_stat_str(self):
        'get the column statuses line for __str__'

        lp = self.lp
        cols = self.get_num_cols()

        stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS", "?(6)?"]
        rv = "   "

        for col in range(1, cols + 1):
            rv += "{:>6} ".format(stat_labels[glpk.glp_get_col_stat(lp, col)])

        rv += "\n"

        return rv

    def _constraints_str(self, zero_print):
        'get the constraints matrix lines for __str__'

        rv = ""
        lp = self.lp
        rows = self.get_num_rows()
        cols = self.get_num_cols()
        
        stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS"]
        inds = SwigArray.get_int_array(cols + 1)
        vals = SwigArray.get_double_array(cols + 1)

        for row in range(1, rows + 1):
            stat = glpk.glp_get_row_stat(lp, row)
            assert 0 <= stat <= len(stat_labels)
            rv += "{:2}: {} ".format(row, stat_labels[stat])

            num_inds = glpk.glp_get_mat_row(lp, row, inds, vals)

            for col in range(1, cols + 1):
                val = 0

                for index in range(1, num_inds+1):
                    if inds[index] == col:
                        val = vals[index]
                        break

                num = f"{val:.6f}"
                num = num.rjust(6)[:6] # fix width to exactly 6

                rv += (zero_print(num) if val == 0 else num) + " "

            row_type = glpk.glp_get_row_type(lp, row)

            assert row_type == glpk.GLP_UP
            val = glpk.glp_get_row_ub(lp, row)
            rv += " <= "

            num = f"{val:.6f}"
            num = num.rjust(6)[:6] # fix width to exactly 6

            rv += (zero_print(num) if val == 0 else num) + " "

            rv += "\n"

        return rv

    def _var_bounds_str(self):
        'get the variable bounds string used in __str__'

        rv = ""

        dbl_max = sys.float_info.max
        added_label = False

        for index, name in enumerate(self.names):
            lb = glpk.glp_get_col_lb(self.lp, index + 1)
            ub = glpk.glp_get_col_ub(self.lp, index + 1)

            if not added_label and (lb != -dbl_max or ub != dbl_max):
                added_label = True
                rv += "(*) Bounded variables:"

            if lb != -dbl_max or ub != dbl_max:
                lb = "-inf" if lb == -dbl_max else lb
                ub = "inf" if ub == dbl_max else ub

                rv += f"\n{name} in [{lb}, {ub}]"

        return rv

    def __str__(self, plain_text=False):
        'get the LP as string (useful for debugging)'

        if plain_text:
            zero_print = lambda x: x
        else:
            def zero_print(s):
                'print function for zeros'

                return colored(s, 'white', attrs=['dark'])

        rows = self.get_num_rows()
        cols = self.get_num_cols()
        rv = "Lp has {} columns (variables) and {} rows (constraints)\n".format(cols, rows)

        rv += self._column_names_str()

        rv += self._opt_dir_str(zero_print)

        rv += "subject to:\n"

        rv += self._col_stat_str()

        rv += self._constraints_str(zero_print)
        
        rv += self._var_bounds_str()

        return rv

    def get_num_rows(self):
        'get the number of rows in the lp'

        return glpk.glp_get_num_rows(self.lp)

    def get_num_cols(self):
        'get the number of columns in the lp'

        cols = glpk.glp_get_num_cols(self.lp)

        assert cols == len(self.names), f"lp had {cols} columns, but names list had {len(self.names)} names"

        return cols

    def add_rows_less_equal(self, rhs_vec):
        '''add rows to the LP with <= constraints

        rhs_vector is the right-hand-side values of the constriants
        '''

        if isinstance(rhs_vec, list):
            rhs_vec = np.array(rhs_vec, dtype=float)

        assert isinstance(rhs_vec, np.ndarray) and len(rhs_vec.shape) == 1, "expected 1-d right-hand-side vector"

        if rhs_vec.shape[0] > 0:
            num_rows = glpk.glp_get_num_rows(self.lp)

            # create new row for each constraint
            glpk.glp_add_rows(self.lp, len(rhs_vec))

            for i, rhs in enumerate(rhs_vec):
                glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint

    def get_types(self):
        '''get the constraint types. These are swiglpk.GLP_FX, swiglpk.GLP_UP, or swiglpk.GLP_LO'''

        lp_rows = glpk.glp_get_num_rows(self.lp)
        rv = []

        for row in range(lp_rows):
            rv.append(glpk.glp_get_row_type(self.lp, row + 1))

        return rv

    def add_positive_cols(self, names):
        'add a certain number of columns to the LP with positive bounds'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:
            num_cols = self.get_num_cols()

            self.names += names
            glpk.glp_add_cols(self.lp, num_vars)

            for i in range(num_vars):
                glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_LO, 0, 0)  # var with lower bounds (0, inf)

    def add_cols(self, names):
        'add a certain number of columns to the LP'

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:
            num_cols = self.get_num_cols()

            self.names += names
            glpk.glp_add_cols(self.lp, num_vars)

            for i in range(num_vars):
                glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_FR, 0, 0)  # free variable (-inf, inf)

    def add_double_bounded_cols(self, names, lb, ub):
        'add a certain number of columns to the LP with the given lower and upper bound'

        assert lb != -np.inf

        lb = float(lb)
        ub = float(ub)
        assert lb <= ub, f"lb ({lb}) <= ub ({ub}). dif: {ub - lb}"

        assert isinstance(names, list)
        num_vars = len(names)

        if num_vars > 0:
            num_cols = self.get_num_cols()

            self.names += names
            glpk.glp_add_cols(self.lp, num_vars)

            for i in range(num_vars):
                if lb == ub:
                    glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_FX, lb, ub)  # fixed variable
                elif ub == np.inf:
                    glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_LO, lb, ub)  # lower-bounded variable
                else:
                    assert lb < ub
                    glpk.glp_set_col_bnds(self.lp, num_cols + i + 1, glpk.GLP_DB, lb, ub)  # double-bounded variable

    def add_dense_row(self, vec, rhs, normalize=True):
        '''
        add a row from a dense nd.array, row <= rhs
        '''

        Timers.tic('add_dense_row')

        assert isinstance(vec, np.ndarray)
        assert len(vec.shape) == 1 or vec.shape[0] == 1
        assert len(vec) == self.get_num_cols(), f"vec had {len(vec)} values, but lpi has {self.get_num_cols()} cols"

        if normalize and not Settings.SKIP_CONSTRAINT_NORMALIZATION:
            norm = np.linalg.norm(vec)
            
            if norm > 1e-9:
                vec = vec / norm
                rhs = rhs / norm

        rows_before = self.get_num_rows()

        self.add_rows_less_equal([rhs])

        data_vec = SwigArray.as_double_array(vec, vec.size)
        indices_vec = SwigArray.get_sequential_int_array(vec.size)

        glpk.glp_set_mat_row(self.lp, rows_before + 1, vec.size, indices_vec, data_vec)

        Timers.toc('add_dense_row')

    def set_constraints_csr(self, data, glpk_indices, indptr, shape):
        '''
        set the constrains row by row to be equal to the passed-in csr matrix attribues

        glpk_indices is already offset by one
        '''

        Timers.tic('set_constraints_csr')
        assert shape[0] <= self.get_num_rows()
        assert shape[1] <= self.get_num_cols()

        if glpk_indices:
            assert isinstance(glpk_indices[0], int), f"indices type was not int: {type(glpk_indices[0])}"

        # actually set the constraints row by row
        assert isinstance(data, list), "data was not a list"

        for row in range(shape[0]):
            # we must copy the indices since glpk is offset by 1 :(
            count = int(indptr[row + 1] - indptr[row])

            indices_list = glpk_indices[indptr[row]:indptr[row+1]]
            indices_vec = SwigArray.as_int_array(indices_list, count)

            #data_row_list = [float(d) for d in data[indptr[row]:indptr[row+1]]]
            #data_vec = SwigArray.as_double_array(data_row_list)
            data_vec = SwigArray.as_double_array(data[indptr[row]:indptr[row+1]], count)

            glpk.glp_set_mat_row(self.lp, 1 + row, count, indices_vec, data_vec)

        Timers.toc('set_constraints_csr')

    def get_rhs(self, row_indices=None):
        '''get the rhs vector of the constraints
        row_indices - a list of requested indices (None=all)
        this returns an np.array of rhs values for the requested indices
        '''

        rv = []

        if row_indices is None:
            lp_rows = glpk.glp_get_num_rows(self.lp)
            row_indices = range(lp_rows)

        for row in row_indices:
            row_type = glpk.glp_get_row_type(self.lp, row + 1)

            assert row_type == glpk.GLP_UP, "Error: Unsupported type ({}) in getRhs() in row {}".format(row_type, row)
            
            limit = glpk.glp_get_row_ub(self.lp, row + 1)
            rv.append(limit)

        return np.array(rv, dtype=float)

    def set_rhs(self, rhs_vec):
        'set (overwrite) the rhs for exising rows'

        assert rhs_vec.size == self.get_num_rows()

        for i, rhs in enumerate(rhs_vec):
            glpk.glp_set_row_bnds(self.lp, i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint

    def get_constraints_csr(self):
        '''get the LP matrix as a csr_matrix
        '''

        lp_rows = self.get_num_rows()
        lp_cols = self.get_num_cols()
        nnz = glpk.glp_get_num_nz(self.lp)

        data = np.zeros((nnz,), dtype=float)
        inds = np.zeros((nnz,), dtype=np.int32)
        indptr = np.zeros((lp_rows+1,), dtype=np.int32)

        inds_row = SwigArray.get_int_array(lp_cols + 1)
        vals_row = SwigArray.get_double_array(lp_cols + 1)
        data_index = 0
        indptr[0] = 0

        for row in range(1, lp_rows + 1):
            got_len = glpk.glp_get_mat_row(self.lp, row, inds_row, vals_row)

            for i in range(1, got_len + 1):
                data[data_index] = vals_row[i]
                inds[data_index] = inds_row[i] - 1
                data_index += 1

            indptr[row] = data_index

        csr_mat = csr_matrix((data, inds, indptr), shape=(lp_rows, lp_cols), dtype=float)
        csr_mat.check_format()

        return csr_mat
        
    def is_feasible(self):
        '''check if the lp is feasible

        returns a feasible point or None
        '''

        return self.minimize(None, fail_on_unsat=False) is not None

    def contains_point(self, pt, tol=1e-9):
        '''does this lpi contain the point?
        this is slow, will pull the constraints and check them
        '''

        print("Warning: called lpi.contains_point() (slow, used for testing)")

        csr = self.get_constraints_csr()
        rhs = self.get_rhs()

        # all rows are upper bounds

        vec = csr.dot(pt)

        assert vec.size == rhs.size
        rv = True

        for mine, bound in zip(vec, rhs):
            if mine - 1e-9 > bound:
                rv = False
                break

        return rv

    def set_minimize_direction(self, direction):
        '''set the optimization direction'''

        assert len(direction) == self.get_num_cols(), f"expected {self.get_num_cols()} cols, but optimization " + \
            f"vector had {len(direction)} variables"
        
        for i, d in enumerate(direction):
            col = int(1 + i)

            glpk.glp_set_obj_coef(self.lp, col, float(d))

    def reset_basis(self, basis_type='std'):
        'reset initial lp basis'

        if basis_type == 'std':
            glpk.glp_std_basis(self.lp)
        elif basis_type == 'adv':
            glpk.glp_adv_basis(self.lp, 0)
        else:
            assert basis_type == 'cpx'
            glpk.glp_cpx_basis(self.lp)

    def minimize(self, direction_vec, fail_on_unsat=True):
        '''minimize the lp, returning a list of assigments to each of the variables

        if direction_vec is not None, this will first assign the optimization direction

        returns None if UNSAT, otherwise the optimization result.
        '''

        assert not isinstance(self.lp, tuple), "self.lp was tuple. Did you call lpi.deserialize()?"

        if direction_vec is None:
            direction_vec = [0] * self.get_num_cols()

        self.set_minimize_direction(direction_vec)

        if Settings.GLPK_RESET_BEFORE_MINIMIZE:
            self.reset_basis()
        
        start = time.perf_counter()
        simplex_res = glpk.glp_simplex(self.lp, get_lp_params())

        if simplex_res != 0: # solver failure (possibly timeout)
            r = self.get_num_rows()
            c = self.get_num_cols()

            diff = time.perf_counter() - start
            print(f"GLPK timed out / failed ({simplex_res}) after {round(diff, 3)} sec with primary " + \
                  f"settings with {r} rows and {c} cols")

            print("Retrying with reset")
            self.reset_basis()
            start = time.perf_counter()
            simplex_res = glpk.glp_simplex(self.lp, get_lp_params())
            diff = time.perf_counter() - start
            print(f"result with reset  ({simplex_res}) {round(diff, 3)} sec")

            print("Retrying with reset + alternate GLPK settings")
                    
            # retry with alternate params
            params = get_lp_params(alternate_lp_params=True)
            self.reset_basis()
            start = time.perf_counter()
            simplex_res = glpk.glp_simplex(self.lp, params)
            diff = time.perf_counter() - start
            print(f"result with reset & alternate settings ({simplex_res}) {round(diff, 3)} sec")
            
        rv = self._process_simplex_result(simplex_res)

        if rv is None and fail_on_unsat:
            # extra logic to try harder if fail_on_unsafe is True
            # glpk can sometimes be cajoled into providing a solution
            
            print("Note: minimize failed with fail_on_unsat was true, trying to reset basis...")

            self.reset_basis()
            rv = self.minimize(direction_vec, fail_on_unsat=False)

            if rv is None:
                print("still unsat after reset basis, trying no-dir optimization")
                self.reset_basis()
            
                result_nodir = self.minimize(None, fail_on_unsat=False)

                # lp became infeasible when I picked an optimization direction
                if result_nodir is not None:
                    print("Using result from no-direction optimization") 
                    rv = result_nodir
                else:
                    print("Error: No-dir result was also infeasible!")
                    
                    if self.get_num_rows() < 50 and self.get_num_cols() < 50:
                        print(f"{self}")
            else:
                print("Using result after reset basis (soltion was now feasible)")

        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsat was True")

        return rv

    @staticmethod
    def get_simplex_error_string(simplex_res):
        '''get the error message when simplex() fails'''

        codes = [glpk.GLP_EBADB, glpk.GLP_ESING, glpk.GLP_ECOND, glpk.GLP_EBOUND, glpk.GLP_EFAIL, glpk.GLP_EOBJLL,
                 glpk.GLP_EOBJUL, glpk.GLP_EITLIM, glpk.GLP_ETMLIM, glpk.GLP_ENOPFS, glpk.GLP_ENODFS]

        msgs = [ \
            "Unable to start the search, because the initial basis specified " + \
            "in the problem object is invalid-the number of basic (auxiliary " + \
            "and structural) variables is not the same as the number of rows " + \
            "in the problem object.", 

            "Unable to start the search, because the basis matrix corresponding " + \
            "to the initial basis is singular within the working " + \
            "precision.",

            "Unable to start the search, because the basis matrix corresponding " + \
            "to the initial basis is ill-conditioned, i.e. its " + \
            "condition number is too large.",

            "Unable to start the search, because some double-bounded " + \
            "(auxiliary or structural) variables have incorrect bounds.",

            "The search was prematurely terminated due to the solver " + \
            "failure.",

            "The search was prematurely terminated, because the objective " + \
            "function being maximized has reached its lower " + \
            "limit and continues decreasing (the dual simplex only).",

            "The search was prematurely terminated, because the objective " + \
            "function being minimized has reached its upper " + \
            "limit and continues increasing (the dual simplex only).",

            "The search was prematurely terminated, because the simplex " + \
            "iteration limit has been exceeded.",

            "The search was prematurely terminated, because the time " + \
            "limit has been exceeded.",

            "The LP problem instance has no primal feasible solution " + \
            "(only if the LP presolver is used).",

            "The LP problem instance has no dual feasible solution " + \
            "(only if the LP presolver is used).",
            ]

        rv = "Unknown Error"

        for code, message in zip(codes, msgs):
            if simplex_res == code:
                rv = message
                break

        return rv

    def _process_simplex_result(self, simplex_res):
        '''process the result of a glp_simplex call

        returns None on UNSAT, otherwise the optimization result with the requested columns
        if columns is None, will return full result
        '''

        rv = None

        if simplex_res != glpk.GLP_ENOPFS:  # skip if no primal feasible w/ presolver
            
            if simplex_res != 0: # simplex failed, report the error
                raise RuntimeError("glp_simplex returned nonzero status ({}): {}".format(
                    simplex_res, LpInstance.get_simplex_error_string(simplex_res)))
            
            status = glpk.glp_get_status(self.lp)

            if status == glpk.GLP_NOFEAS: # infeasible
                rv = None
            elif status == glpk.GLP_OPT: # optimal
                lp_cols = self.get_num_cols()
                rv = np.zeros(lp_cols)

                for col in range(lp_cols):
                    rv[col] = glpk.glp_get_col_prim(self.lp, int(1 + col))

            else: # neither infeasible nor optimal (for example, unbounded)
                error_msg = "<Unknown Status>"
                
                codes = [glpk.GLP_OPT, glpk.GLP_FEAS, glpk.GLP_INFEAS, glpk.GLP_NOFEAS, glpk.GLP_UNBND, glpk.GLP_UNDEF]
                msgs = ["solution is optimal",
                        "solution is feasible",
                        "solution is infeasible",
                        "problem has no feasible solution",
                        "problem has unbounded solution",
                        "solution is undefined"]

                for code, message in zip(codes, msgs):
                    if status == code:
                        error_msg = message
                        break

                if status == glpk.GLP_UNBND:
                    ray = glpk.glp_get_unbnd_ray(self.lp)
                    error_msg += f"; unbounded ray was variable #{ray}"

                raise RuntimeError(f"LP status after minimize() was {status}: {error_msg}")

        return rv

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'

class SwigArray():
    '''
    This is my workaround to fix a memory leak in swig arrays, see: https://github.com/biosustain/swiglpk/issues/31)

    The general idea is to only allocate a single time for each type, and reuse the array
    '''

    dbl_array = []
    dbl_array_size = -1

    int_array = []
    int_array_size = -1

    seq_array = []
    seq_array_size = -1

    @classmethod
    def get_double_array(cls, size):
        'get a double array of the requested size (or greater)'

        if size > cls.dbl_array_size:
            cls.dbl_array_size = 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.dbl_array = glpk.doubleArray(cls.dbl_array_size)

            #print(f"allocated dbl array of size {cls.dbl_array_size} (requested {size})")

        return cls.dbl_array

    @classmethod
    def get_int_array(cls, size):
        'get a int array of the requested size (or greater)'

        if size > cls.int_array_size:
            cls.int_array_size = 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.int_array = glpk.intArray(cls.int_array_size)

            #print(f".allocated int array of size {cls.int_array_size} (requested {size})")

        #print(f".returning {cls.int_array} of size {cls.int_array_size} (requested {size})")

        return cls.int_array

    @classmethod
    def as_double_array(cls, list_data, size):
        'wrapper for swig as_doubleArray'

        # about 3x slower than glpk.as_doubleArray, but doesn't leak memory
        arr = cls.get_double_array(size + 1)

        for i, val in enumerate(list_data):
            arr[i+1] = float(val)
            
        return arr

    @classmethod
    def as_int_array(cls, list_data, size):
        'wrapper for swig as_intArray'

        # about 3x slower than glpk.as_intArray, but doesn't leak memory
        arr = cls.get_int_array(size + 1)

        for i, val in enumerate(list_data):
            #print(f"setting {i+1} <- val: {val} ({type(val)}")
            arr[i+1] = val

        return arr

    @classmethod
    def get_sequential_int_array(cls, size):
        'creates or returns a swig int array that counts from 1, 2, 3, 4, .. size'

        if size > (cls.seq_array_size - 1):
            cls.seq_array_size = 1 + 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.seq_array = glpk.intArray(cls.seq_array_size)

            #print(f"allocated seq array of size {cls.seq_array_size} (requested {size})")

            for i in range(cls.seq_array_size):
                cls.seq_array[i] = i

        return cls.seq_array
