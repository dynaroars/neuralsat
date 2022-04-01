import numpy as np


class EtaMatrix:

    def __init__(self, col_idx: int, col_vals: np.array):
        """
        :param col_idx: column count starts from 0.
        :param col_vals: the values of the column, as 1 dimensional numpy array
        """
        self._col_idx, self._col_vals = col_idx, col_vals


    def invert(self):
        """
        Inverts the matrix, in place.
        :return: this matrix, inverted.
        >>> m = EtaMatrix(1, np.array([-4, 3, 2]))
        >>> np.all(np.isclose(m.invert()._col_vals, np.array([ 1.33333333,  0.33333333, -0.66666667])))
        True
        """
        inverse_diag_element = 1 / self._col_vals[self._col_idx]
        self._col_vals = np.multiply(self._col_vals, -inverse_diag_element)
        self._col_vals[self._col_idx] = inverse_diag_element
        return self


    def solve_left_mult(self, y: np.array) -> np.array:
        """
        :return: solution of [x1, ..., xn] * self = y.
        >>> m = EtaMatrix(1, np.array([-4, 3, 2]))
        >>> m.solve_left_mult(np.array([1, 2, 3]))
        array([1, 0, 3])
        >>> m.solve_left_mult(np.array([3, 2, 1]))
        array([3, 4, 1])
        """
        result = np.copy(y)
        result[self._col_idx] = 0
        result[self._col_idx] = (np.matmul(-self._col_vals, result) + y[self._col_idx]) / self._col_vals[self._col_idx]
        return result


    @staticmethod
    def iteratively_solve_left_mult(eta_matrices, y: np.array) -> np.array:
        """
        :return: solution of [x1, ..., xn] * E1 * E2 * ... = y, where E1, ... are eta matrices.
        >>> e1 = EtaMatrix(1, np.array([1, 1, 3]))
        >>> e2 = EtaMatrix(0, np.array([2, 1, 1]))
        >>> EtaMatrix.iteratively_solve_left_mult([e1, e2], np.array([19, 12, 0]))
        array([3.5, 8.5, 0. ])
        """
        cur_y = y.astype(np.float64)    # Copies y
        for cur_matrix in reversed(eta_matrices):
            cur_y = cur_matrix.solve_left_mult(cur_y)
        return cur_y


    def solve_right_mult(self, y: np.array) -> np.array:
        """
        :return: solution of self * [x1, ..., xn] = y.
        >>> m = EtaMatrix(1, np.array([-4., 3., 2.]))
        >>> m.solve_right_mult(np.array([1., 2., 3.]))
        array([3.66666667, 0.66666667, 1.66666667])
        """
        eta_val = y[self._col_idx] / self._col_vals[self._col_idx]
        result = y - self._col_vals * eta_val
        result[self._col_idx] = eta_val
        return result


    @staticmethod
    def iteratively_solve_right_mult(eta_matrices, y: np.array) -> np.array:
        """
        :return: solution of E1 * E2 * ... * [x1, ..., xn] = y, where E1, ... are eta matrices.
        """
        cur_y = y.astype(np.float64)    # Copies y
        for cur_matrix in eta_matrices:
            cur_y = cur_matrix.solve_right_mult(cur_y)
        return cur_y


    def get_full_matrix(self) -> np.array:
        """
        :return: this eta matrix, as a "regular" numpy matrix.
        >>> e1 = EtaMatrix(1, np.array([1, 1, 3]))
        >>> np.all(e1.get_full_matrix() == np.array([[1., 1., 0.], [0., 1., 0.], [0., 3., 1.]]))
        True
        >>> e2 = EtaMatrix(0, np.array([2, 1, 1]))
        >>> np.all(e2.get_full_matrix() == np.array([[2., 0., 0.], [1., 1., 0.], [1., 0., 1.]]))
        True
        """
        matrix = np.identity(len(self._col_vals))
        matrix[:, self._col_idx] = np.copy(self._col_vals)
        return matrix
