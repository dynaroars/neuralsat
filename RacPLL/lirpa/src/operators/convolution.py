from .base import *
from torch.nn import functional as F

class BoundedConv(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']

        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False

        self.to(device)


    @Bound.save_io_shape
    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv2d(x[0], x[1], bias, self.stride, self.padding, self.dilation, self.groups)
        return output