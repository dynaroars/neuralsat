from .base import *

class BoundedLinear(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        # Gemm:
        #   A = A if transA == 0 else A.T
        #   B = B if transB == 0 else B.T
        #   C = C if C is not None else np.array(0)
        #   Y = alpha * np.dot(A, B) + beta * C
        #   return Y

        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
