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

        # Defaults in ONNX
        self.transA = 0
        self.transB = 0
        self.alpha = 1.0
        self.beta = 1.0
        if attr is not None:
            self.transA = attr['transA'] if 'transA' in attr else self.transA
            self.transB = attr['transB'] if 'transB' in attr else self.transB
            self.alpha = attr['alpha'] if 'alpha' in attr else self.alpha
            self.beta = attr['beta'] if 'beta' in attr else self.beta


    """Handle tranpose and linear coefficients."""
    def _preprocess(self, a, b, c=None):
        if self.transA and isinstance(a, torch.Tensor):
            a = a.transpose(-2,-1)
        if self.alpha != 1.0:
            a = self.alpha * a
        if not self.transB and isinstance(b, torch.Tensor):
            # our code assumes B is transposed (common case), so we transpose B only when it is not transposed in gemm.
            b = b.transpose(-2,-1)
        if c is not None:
            if self.beta != 1.0:
                c = self.beta * c
        return a, b, c


    @Bound.save_io_shape
    def forward(self, x, w, b=None):
        x, w, b = self._preprocess(x, w, b)
        self.input_shape = self.x_shape = x.shape
        self.y_shape = w.t().shape
        res = x.matmul(w.t())
        if b is not None:
            res += b
        return res
