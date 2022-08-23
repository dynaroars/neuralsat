from .base import *
from torch.nn import functional as F

class BoundedActivation(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)



class BoundedOptimizableActivation(BoundedActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)



class BoundedRelu(BoundedOptimizableActivation):
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.flattened_nodes = None

       
    @Bound.save_io_shape
    def forward(self, x):
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return F.relu(x)
