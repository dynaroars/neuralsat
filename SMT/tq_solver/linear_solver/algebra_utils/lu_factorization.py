from tq_solver.linear_solver.algebra_utils.eta_matrix import EtaMatrix
import numpy as np


class LUFactorization:
    """
    Based on https://github.com/AvivYaish/LUM/
    """

    @staticmethod
    def generate_pivot_list(matrix):
        """
        This function finds a permutation of the matrix's rows such that the resulting matrix has a LU decomposition.
        :return: pivot_list, a list of (row1, row2) tuples, where row1 and row2 were switched at that step.
        """
        cur_matrix, pivot_list = matrix.astype(np.float64), []
        for row_idx in np.arange(np.size(matrix, 1)):
            max_row = np.argmax(np.abs(cur_matrix[row_idx:, row_idx])) + row_idx
            if row_idx != max_row:
                cur_matrix[[row_idx, max_row], :] = cur_matrix[[max_row, row_idx], :]
                pivot_list.append((row_idx, max_row))
            cur_matrix[row_idx + 1:] += np.outer(cur_matrix[row_idx + 1:, row_idx],
                                                 cur_matrix[row_idx] / cur_matrix[row_idx, row_idx])
        return pivot_list

    @staticmethod
    def pivot_array(pivot_list, array: np.array, reverse=False, in_place=True) -> np.array:
        """
        :param reverse: True if the pivots of pivot_list should be done in a reverse order (from last to first).
        :return: the array, after pivoting according to 'pivots'.
        """
        if in_place:
            array_to_return = array
        else:
            array_to_return = np.copy(array)

        if reverse:
            pivot_iterable = reversed(pivot_list)
        else:
            pivot_iterable = pivot_list
            
        for row1, row2 in pivot_iterable:
            array_row1 = np.copy(array_to_return[row1])
            array_to_return[row1] = array_to_return[row2]
            array_to_return[row2] = array_row1
        return array_to_return

    @staticmethod
    def lu_factorization(matrix, in_place=True):
        """
        :return: A list [L_1^-1, L_2^-1, ..., U_N, U_N-1, ...] such that
        matrix = L_1^-1 * ... * L_N^-1 * U_N * ... U_1, and all L_i^-1, U_i are eta matrices.
        """
        if in_place:
            cur_matrix = matrix
        else:
            cur_matrix = matrix.astype(np.float64)
        # Achieves the best theoretical run-time
        row_num, matrices = np.size(matrix, 0), []

        # Create L matrices
        for row_idx, cur_eta_col in enumerate(np.identity(row_num)):
            cur_eta_col[row_idx+1:] = -cur_matrix[row_idx+1:, row_idx] / cur_matrix[row_idx, row_idx]
            cur_matrix[row_idx+1:] += np.outer(cur_eta_col[row_idx+1:], cur_matrix[row_idx])
            matrices.append(EtaMatrix(row_idx, cur_eta_col).invert())

        # Create U matrices
        for row_idx, cur_eta_col in enumerate(reversed(cur_matrix.T)):
            matrices.append(EtaMatrix(row_num - row_idx - 1, cur_eta_col))
        return matrices

    @staticmethod
    def plu_factorization(matrix):
        """
        :return: pivot_list and a list [L_1^-1, L_2^-1, ..., U_N, U_N-1, ...] such that after pivoting the
        input matrix according to pivot_list, it is equal to = L_1^-1 * ... * L_N^-1 * U_N * ... U_1, and
        all L_i^-1, U_i are eta matrices.
        """
        pivot_list = LUFactorization.generate_pivot_list(matrix)
        return pivot_list, LUFactorization.lu_factorization(
            LUFactorization.pivot_array(pivot_list, matrix, in_place=False)
        )
