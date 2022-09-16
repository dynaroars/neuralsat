from .base import *

class BoundedBatchNormalization(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device, training):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
