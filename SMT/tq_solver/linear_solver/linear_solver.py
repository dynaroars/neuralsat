from tq_solver.linear_solver.algebra_utils.lu_factorization import LUFactorization
from tq_solver.linear_solver.algebra_utils.eta_matrix import EtaMatrix
from solver.solver import Solver
from itertools import chain
import numpy as np

class LinearSolver(Solver):

    Bland = "Bland"
    Dantzig = "Dantzig"
    FirstPositive = "FirstPositive"


    def __init__(self, a_matrix, b, c, entering_selection_rule=Bland, 
                 auxiliary=False, refactorization_threshold=100,
                 epsilon=np.float64(1e-10), stability_testing_period=100):
        """
        :param a_matrix: the coefficient matrix.
        :param b: the constraint vector.
        :param c: the objective function.
        :param entering_selection_rule: the entering selection rule, either LinearSolver.Bland, LinearSolver.Dantzig
        or LinearSolver.FirstPositive (which picks the first positive variable in the current objective function).
        :param auxiliary: True iff this is an auxiliary problem.
        :param refactorization_threshold: if the solver accumulated more eta matrices than the threshold, refactorize the base.
        :param epsilon: the tolerance for picking the entering variable (the component of c_n - y * a_matrix_n that will
        be picked will have to be greater than epsilon), and for checking that a_matrix_b * x_b_star is epsilon-close to
        b (the constraint vector).
        :param stability_testing_period: the time period after which epsilon-closeness to b is checked.
        """

        super().__init__()

        # debug

        # print(a_matrix, b, c, auxiliary)
        # print()


        self._score = np.float64(0.0)   # Current score

        self._rows = np.size(a_matrix, 0)
        self._cols = np.size(a_matrix, 1)

        self._epsilon = epsilon
        self._stability_testing_period = stability_testing_period
        self._step_count = 0

        self._refactorization_threshold = refactorization_threshold
        self._eta_matrices = []
        self._pivot_list = []

        # coefficient matrix for "base" variables
        self._a_matrix_b = np.identity(self._rows, dtype=np.float64)
        # Indices of current base variables
        self._x_b_vars = np.arange(self._rows) + self._cols  
        # Note that astype copies the array.
        self._original_b = b.astype(np.float64)
        # Current assignment for base variables
        self._x_b_star = b.astype(np.float64)  
        # The objective function for each base variable
        self._c_b = np.zeros(self._rows, dtype=np.float64)  

        # Coefficient matrix for non-base variables
        self._a_matrix_n = a_matrix.astype(np.float64)

        self._x_n_vars = np.arange(self._cols)
        self._x_n_star = np.zeros(self._cols)
        self._c_n = c.astype(np.float64)

        if entering_selection_rule == LinearSolver.Bland:
            self._entering_selection_rule = self._bland_rule
        elif entering_selection_rule == LinearSolver.Dantzig:
            self._entering_selection_rule = self._dantzig_rule
        elif entering_selection_rule == LinearSolver.FirstPositive:
            self._entering_selection_rule = self._first_positive_rule

        self._aux_solver = None
        if auxiliary:
            self._initial_auxiliary_step()


    def _initial_auxiliary_step(self):
        # The entering variable is always the new variable created for the auxiliary
        # problem which has an index of 0, so a = _a_matrix_n[:, 0] = [-1, ..., -1].
        # The leaving variable is the one corresponding to the minimal b_i.
        # Because this is the first iteration, _a_matrix_b is the identity matrix,
        # so d = a * (_a_matrix_b^-1) = a * (I^-1) = a * I = a, thus t = -min_b_i
        entering_var = 0
        leaving_var = np.argmin(self._x_b_star)
        t = -self._x_b_star[leaving_var]
        d = self._a_matrix_n[:, entering_var].copy()
        self._pivot(entering_var, leaving_var, t, d)
        

    def _pivot(self, entering_col_idx: int, leaving_col_idx: int, t: np.array, d: np.array):
        # Update the matrices
        entering_col = self._a_matrix_n[:, entering_col_idx].copy()
        self._a_matrix_n[:, entering_col_idx] = self._a_matrix_b[:, leaving_col_idx]
        self._a_matrix_b[:, leaving_col_idx] = entering_col

        # Update etas
        self._eta_matrices.append(EtaMatrix(leaving_col_idx, d))

        # Update the objective function
        self._c_b[leaving_col_idx], self._c_n[entering_col_idx] = \
            self._c_n[entering_col_idx], self._c_b[leaving_col_idx]

        # Update indices
        self._x_b_vars[leaving_col_idx], self._x_n_vars[entering_col_idx] = \
            self._x_n_vars[entering_col_idx], self._x_b_vars[leaving_col_idx]

        # Update the assignment
        self._x_b_star -= t * d
        self._x_b_star[leaving_col_idx] = t
        

    @staticmethod
    def _first_positive_rule(cur_objective_func):
        for idx in range(len(cur_objective_func)):
            if cur_objective_func[idx] > 0:
                return idx
        return -1


    def _bland_rule(self, cur_objective_func):
        return np.where(cur_objective_func > 0, self._x_n_vars, np.inf).argmin()


    @staticmethod
    def _dantzig_rule(cur_objective_func):
        return np.argmax(cur_objective_func)


    def _solve_auxiliary_problem(self) -> bool:
        print('he')
        new_a_matrix = np.concatenate((-np.ones((self._rows, 1)), self._a_matrix_n), axis=1)
        new_c = np.concatenate((np.array([-1]), np.zeros(self._cols)))
        self._aux_solver = LinearSolver(new_a_matrix, self._x_b_star, new_c, auxiliary=True)
        self._aux_solver.solve()
        # The auxiliary problem has an additional first variable, its ID is 0
        return self._aux_solver.get_assignment()[0] == 0


    def _update_to_match_auxiliary_problem(self):
        """
        Update the data-structure of the current problem to match the auxiliary's.
        """
        # Can prove the new variable is not in the basis.
        self._x_b_vars = self._aux_solver._x_b_vars - 1 # All variables (including slack ones) are shifted by 1
        self._x_b_star = self._aux_solver._x_b_star
        self._a_matrix_b = self._aux_solver._a_matrix_b
        self._pivot_list = self._aux_solver._pivot_list
        self._eta_matrices = self._aux_solver._eta_matrices

        # Remove the new variable from all data structures
        new_var_idx = np.argmin(self._aux_solver._x_n_vars)
        self._x_n_vars = np.delete(self._aux_solver._x_n_vars - 1, new_var_idx)
        self._a_matrix_n = np.delete(self._aux_solver._a_matrix_n, new_var_idx, axis=1)

        # Reorder _c_b and _c_n accordingly
        for idx, var in enumerate(self._x_b_vars):
            if var < self._cols:  # var is not slack
                self._c_b[idx] = self._c_n[var]

        new_c_n = np.zeros(self._cols, dtype=np.float64)
        for idx, var in enumerate(self._x_n_vars):
            if var >= self._cols:   # var is slack
                new_c_n[idx] = 0
            else:
                new_c_n[idx] = self._c_n[var]
        self._c_n = new_c_n


    def _refactorize_base(self):
        self._pivot_list, self._eta_matrices = LUFactorization.plu_factorization(self._a_matrix_b)


    def _is_a_matrix_b_epsilon_close(self) -> bool:
        return np.all(
            np.isclose(
                self._original_b, 
                EtaMatrix.iteratively_solve_right_mult(
                    self._eta_matrices, 
                    LUFactorization.pivot_array(
                        self._pivot_list, 
                        self._x_b_star, 
                        in_place=False
                    )
                ), 
                rtol=0, 
                atol=self._epsilon
            )
        )


    def _single_iteration(self):
        self._step_count = (self._step_count + 1) % self._stability_testing_period
        if ((self._step_count == 0) and (not self._is_a_matrix_b_epsilon_close())) or \
                (len(self._eta_matrices) > self._refactorization_threshold):
            self._refactorize_base()

        entering_col_idx = self._choose_entering_col(self._btran())  # y = self._btran()
        if entering_col_idx == -1:
            return np.matmul(self._c_b, self._x_b_star)

        d = self._ftran(entering_col_idx)
        leaving_col_idx, t = self._choose_leaving_col(d)
        if t == np.inf:
            self._x_n_star[entering_col_idx] = np.inf
            return np.inf

        self._pivot(entering_col_idx, leaving_col_idx, t, d)
        return None


    def _choose_entering_col(self, y):
        # Can do y * a_matrix_n using eta matrices!
        cur_objective_func = self._c_n - np.matmul(y, self._a_matrix_n) - self._epsilon
        positive_indices = cur_objective_func > 0
        if not np.any(positive_indices):
            return -1
        return self._entering_selection_rule(cur_objective_func)


    def _choose_leaving_col(self, d):
        all_ts = self._x_b_star / d
        if np.all(all_ts <= 0):
            return -1, np.inf
        # Cool numpy method for finding smallest positive value, found at:
        # https://stackoverflow.com/questions/55769522/how-to-find-maximum-negative-and-minimum-positive-number-in-a-numpy-array
        min_ratio = np.where(all_ts > 0, all_ts, np.inf).min()
        # Choose var which achieves min-ratio and has the minimal index
        min_ratio_idx = np.where(all_ts == min_ratio, self._x_b_vars, np.inf).argmin()
        return min_ratio_idx, min_ratio


    def get_assignment(self):
        assignment = {var: 0 for var in range(self._cols)}
        for var, value in chain(zip(self._x_b_vars, self._x_b_star), zip(self._x_n_vars, self._x_n_star)):
            if var in assignment:
                assignment[var] = value
        return assignment


    def _btran(self):
        """
        :return: the solution 'y' of y * _a_matrix_b = _c_b
        """
        # return np.matmul(self._c_b, np.linalg.inv(self._a_matrix_b))
        return LUFactorization.pivot_array(
            self._pivot_list,
            EtaMatrix.iteratively_solve_left_mult(
                self._eta_matrices, 
                self._c_b
            ),
            reverse=True
        )


    def _ftran(self, entering_col):
        """
        :return: np.linalg.solve(self._a_matrix_b, self._a_matrix_n[:, entering_col])
        """
        return EtaMatrix.iteratively_solve_right_mult(
            self._eta_matrices,
            LUFactorization.pivot_array(
                self._pivot_list,
                self._a_matrix_n[:, entering_col], 
                in_place=False
            )
        )


    def is_sat(self) -> bool:
        return np.all(self._x_b_star >= 0) or self._solve_auxiliary_problem()


    def solve(self) -> bool:
        if not self.is_sat():
            return False
        elif self._aux_solver is not None:
            self._update_to_match_auxiliary_problem()

        while True:
            result = self._single_iteration()
            if result is not None:
                self._score = result
                return True

