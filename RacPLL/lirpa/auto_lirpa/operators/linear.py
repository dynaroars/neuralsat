from auto_lirpa.utils import *
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


    def bound_backward(self, last_lA, last_uA, *x):
        # print('[+] Bound backward from', self)
        # print('\t- Input:', x)
        has_bias = len(x) == 3

        # x[0]: input node, x[1]: weight, x[2]: bias
        input_lb = [xi.lower if hasattr(xi, 'lower') else None for xi in x]
        input_ub = [xi.upper if hasattr(xi, 'upper') else None for xi in x]

        input_lb = self._preprocess(*input_lb)
        input_ub = self._preprocess(*input_ub)

        lA_y = uA_y = lA_bias = uA_bias = None
        lbias = ubias = 0
        batch_size = last_lA.shape[1] if last_lA is not None else last_uA.shape[1]


        # Case 1: No weight/bias perturbation, only perturbation on input.
        if not self.is_input_perturbed(1) and (not has_bias or not self.is_input_perturbed(2)):
            # If last_lA and last_uA are indentity matrices.
            if isinstance(last_lA, eyeC) and isinstance(last_uA, eyeC):
                # Use this layer's W as the next bound matrices. Duplicate the batch dimension. Other dimensions are kept 1.
                # Not perturbed, so we can use either lower or upper.
                lA_x = uA_x = input_lb[1].unsqueeze(1).repeat([1, batch_size] + [1] * (input_lb[1].ndim - 1))
                if has_bias:
                    lbias = ubias = input_lb[2].unsqueeze(1).repeat(1, batch_size)
            elif isinstance(last_lA, OneHotC) or isinstance(last_uA, OneHotC):
                # We need to select several rows from the weight matrix (its shape is output_size * input_size).
                lA_x, lbias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_lA, batch_size)
                if last_lA is last_uA:
                    uA_x = lA_x
                    ubias = lbias
                else:
                    uA_x, ubias = self.onehot_mult(input_lb[1], input_lb[2] if has_bias else None, last_uA, batch_size)
            else:
                def _bound_oneside(last_A):
                    if last_A is None:
                        return None, 0
                    # Just multiply this layer's weight into bound matrices, and produce biases.
                    next_A = last_A.to(input_lb[1]).matmul(input_lb[1])
                    sum_bias = (last_A.to(input_lb[2]).matmul(input_lb[2]) 
                        if has_bias else 0.0)
                    return next_A, sum_bias

                lA_x, lbias = _bound_oneside(last_lA)
                uA_x, ubias = _bound_oneside(last_uA)


        # Case 2: Weight is perturbed. bias may or may not be perturbed.
        elif self.is_input_perturbed(1):
            raise NotImplementedError()
            
        # Case 3: Only bias is perturbed, weight is not perturbed.
        elif not self.is_input_perturbed(1) and has_bias and self.is_input_perturbed(2):
            raise NotImplementedError()

        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias


    """Multiply weight matrix with a diagonal matrix with selected rows."""
    def onehot_mult(self, weight, bias, C, batch_size):

        if C is None:
            return None, 0.0

        new_weight = None
        new_bias = 0.0

        if C.index.ndim == 2:
            # Shape is [spec, batch]
            index = C.index.transpose(0,1)
            coeffs = C.coeffs.transpose(0,1)
        else:
            index = C.index
            coeffs = C.coeffs

        if C.index.ndim == 1:
            # Every element in the batch shares the same rows.
            if weight is not None:
                new_weight = self.non_deter_index_select(weight, dim=0, index=index).unsqueeze(1).expand([-1, batch_size] + [-1] * (weight.ndim - 1))
            if bias is not None:
                new_bias = self.non_deter_index_select(bias, dim=0, index=index).unsqueeze(1).expand(-1, batch_size)
        elif C.index.ndim == 2:
            # Every element in the batch has different rows, but the number of rows are the same. This essentially needs a batched index_select function.
            if weight is not None:
                new_weight = batched_index_select(weight.unsqueeze(0), dim=1, index=index)
            if bias is not None:
                new_bias = batched_index_select(bias.unsqueeze(0), dim=1, index=index)
        if C.coeffs is not None:
            if weight is not None:
                new_weight = new_weight * coeffs.unsqueeze(-1)
            if bias is not None:
                new_bias = new_bias * coeffs
        if C.index.ndim == 2:
            # Eventually, the shape of A is [spec, batch, *node] so need a transpose.
            new_weight = new_weight.transpose(0, 1)
            new_bias = new_bias.transpose(0, 1)
        return new_weight, new_bias

















class BoundedMatMul(BoundedLinear):

    # Reuse most functions from BoundLinear.
    
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.transA = 0
        self.transB = 1  # MatMul assumes B is transposed.
        self.nonlinear = True