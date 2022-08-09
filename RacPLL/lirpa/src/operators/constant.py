from .base import *

class BoundedPrimConstant(Bound):
    def __init__(self, input_name, name, ori_name, attr, input, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, input, output_index, options, device)
        self.value = attr['value']

    @Bound.save_io_shape
    def forward(self):
        return torch.tensor([], device=self.device)
