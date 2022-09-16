from .base import *


class BoundedFlatten(Bound):
    
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']
        self.use_default_ibp = True

    @Bound.save_io_shape
    def forward(self, x):
        return torch.flatten(x, self.axis)


    def bound_backward(self, last_lA, last_uA, x):
        # print('[+] Bound backward from', self)
        def _bound_oneside(A):
            if A is None:
                return None
            return A.reshape(A.shape[0], A.shape[1], *self.input_shape[1:])

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0
        
class BoundedShape(Bound):
    
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)


class BoundedTranspose(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)


class BoundedReshape(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
