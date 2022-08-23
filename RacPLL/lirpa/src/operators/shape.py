from .base import *


class BoundedFlatten(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.axis = attr['axis']

    @Bound.save_io_shape
    def forward(self, x):
        return torch.flatten(x, self.axis)


class BoundedShape(Bound):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
