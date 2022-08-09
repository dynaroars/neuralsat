from .operators import *

bound_op_map = {
    'onnx::Gemm': BoundedLinear,
    'prim::Constant': BoundedPrimConstant,
}