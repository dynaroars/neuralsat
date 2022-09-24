from termcolor import colored
import swiglpk as glpk
import numpy as np
import math
import time
import sys

# row: constraints
# col: variables

def get_lp_params(alternate_lp_params=False):
    'get the lp params object'

    if not hasattr(get_lp_params, 'obj'):
        params = glpk.glp_smcp()
        glpk.glp_init_smcp(params)

        #params.msg_lev = glpk.GLP_MSG_ERR
        params.msg_lev = glpk.GLP_MSG_ERR
        params.meth = glpk.GLP_PRIMAL # if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_DUAL

        params.tm_lim = 1000
        params.out_dly = 2 * 1000 # start printing to terminal delay
        
        get_lp_params.obj = params

        # make alternative params
        params2 = glpk.glp_smcp()
        glpk.glp_init_smcp(params2)
        params2.meth = glpk.GLP_DUAL # if Settings.GLPK_FIRST_PRIMAL else glpk.GLP_PRIMAL
        params2.msg_lev = glpk.GLP_MSG_ON

        params2.tm_lim = 1000
        params2.out_dly = 1 * 1000 # start printing to terminal status after 1 secs
        
        get_lp_params.alt_obj = params2
        
    if alternate_lp_params:
        #glpk.glp_term_out(glpk.GLP_ON)
        rv = get_lp_params.alt_obj
    else:
        #glpk.glp_term_out(glpk.GLP_OFF)
        rv = get_lp_params.obj

    return rv


class GLPKSolver:

    def __init__(self, num_vars):

        self.names = [f'x{i}' for i in range(num_vars)]
        self.lp = None
        self.backsub_dict = None

    def set_var_bnds(self, lbs, ubs):
        for i in range(len(lbs)):
            glpk.glp_set_col_bnds(self.lp, i + 1, glpk.GLP_DB, lbs[i], ubs[i])  # double-bounded variable


    def add_rhs_less_equal(self, rhs_vec):
        '''add rows to the LP with <= constraints

        rhs_vector is the right-hand-side values of the constriants
        '''

        if isinstance(rhs_vec, list):
            rhs_vec = np.array(rhs_vec, dtype=float)

        num_rows = glpk.glp_get_num_rows(self.lp)

        # create new row for each constraint
        glpk.glp_add_rows(self.lp, len(rhs_vec))

        for i, rhs in enumerate(rhs_vec):
            glpk.glp_set_row_bnds(self.lp, num_rows + i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint


    def add_dense_row(self, vec, rhs, normalize=True):

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
                rhs = rhs / norm

        rows_before = glpk.glp_get_num_rows(self.lp)

        data_vec = SwigArray.as_double_array(vec, len(vec))
        indices_vec = SwigArray.get_sequential_int_array(len(vec))
        self.add_rhs_less_equal(rhs)

        glpk.glp_set_mat_row(self.lp, rows_before + 1, len(vec), indices_vec, data_vec)


    def build(self, backsub_dict, assignment, lbs, ubs, normalize=True):
        self.backsub_dict = backsub_dict

        if self.lp is not None:
            glpk.glp_delete_prob(self.lp)

        self.lp = glpk.glp_create_prob()
        glpk.glp_add_cols(self.lp, len(lbs))

        self.set_var_bnds(lbs, ubs)

        for idx, status in assignment.items():
            mat = backsub_dict[idx].numpy().astype(np.float64)
            vec = mat[:-1]
            rhs = -1 * mat[-1:]

            if status:
                self.add_dense_row(-1 * vec, -1 * rhs - 1e-8, normalize=normalize) 
            else:
                self.add_dense_row(vec, rhs, normalize=normalize)


    def minimize_output(self, idx, maximize=False):
        mat = self.backsub_dict[idx].numpy().astype(np.float64)
        row = mat[:-1]
        bias = mat[-1]

        if maximize:
            row = -1 * row

        self.minimize(row)

        val = glpk.glp_get_obj_val(self.lp)
        if maximize:
            val = -1 * val

        # status = glpk.glp_get_status(self.lp)
        # print(rv, status, glpk.GLP_FEAS, glpk.GLP_INFEAS, glpk.GLP_NOFEAS, glpk.GLP_UNBND, glpk.GLP_UNDEF)

        return val + bias

    def set_minimize_direction(self, direction):
        '''set the optimization direction'''

        for i, d in enumerate(direction):
            col = int(1 + i)
            glpk.glp_set_obj_coef(self.lp, col, float(d))


    def minimize(self, direction_vec, fail_on_unsat=True):
        '''minimize the lp, returning a list of assigments to each of the variables

        if direction_vec is not None, this will first assign the optimization direction

        returns None if UNSAT, otherwise the optimization result.
        '''

        assert not isinstance(self.lp, tuple), "self.lp was tuple. Did you call lpi.deserialize()?"

        if direction_vec is None:
            direction_vec = [0] * self.get_num_cols()

        self.set_minimize_direction(direction_vec)

        # print(direction_vec)

        # if Settings.GLPK_RESET_BEFORE_MINIMIZE:
        glpk.glp_std_basis(self.lp)
        
        start = time.perf_counter()
        simplex_res = glpk.glp_simplex(self.lp, get_lp_params())

        # if simplex_res != 0: # solver failure (possibly timeout)
        #     r = self.get_num_rows()
        #     c = self.get_num_cols()

        #     diff = time.perf_counter() - start
        #     print(f"GLPK timed out / failed ({simplex_res}) after {round(diff, 3)} sec with primary " + \
        #           f"settings with {r} rows and {c} cols")

        #     print("Retrying with reset")
        #     self.reset_basis()
        #     start = time.perf_counter()
        #     simplex_res = glpk.glp_simplex(self.lp, get_lp_params())
        #     diff = time.perf_counter() - start
        #     print(f"result with reset  ({simplex_res}) {round(diff, 3)} sec")

        #     print("Retrying with reset + alternate GLPK settings")
                    
        #     # retry with alternate params
        #     params = get_lp_params(alternate_lp_params=True)
        #     self.reset_basis()
        #     start = time.perf_counter()
        #     simplex_res = glpk.glp_simplex(self.lp, params)
        #     diff = time.perf_counter() - start
        #     print(f"result with reset & alternate settings ({simplex_res}) {round(diff, 3)} sec")
            
        return simplex_res


    def get_num_rows(self):
        'get the number of rows in the lp'

        return glpk.glp_get_num_rows(self.lp)

    def get_num_cols(self):
        'get the number of columns in the lp'

        return glpk.glp_get_num_cols(self.lp)


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

        rows = glpk.glp_get_num_rows(self.lp)
        cols = glpk.glp_get_num_cols(self.lp)
        rv = "Lp has {} columns (variables) and {} rows (constraints)\n".format(cols, rows)

        rv += self._column_names_str()

        rv += self._opt_dir_str(zero_print)

        rv += "subject to:\n"

        rv += self._col_stat_str()

        rv += self._constraints_str(zero_print)
        
        rv += self._var_bounds_str()

        return rv




class SwigArray:
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
        



