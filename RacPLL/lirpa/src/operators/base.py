import torch.nn as nn
import torch
import time
import os

class Bound(nn.Module):

    r"""
    Base class for supporting the bound computation of an operator.

    Args:
        input_name (list): The name of input nodes.
        name (str): The name of this node.
        ori_name (str): Name in the original model.
        attr (dict): Attributes of the operator.
        inputs (list): A list of input nodes.
        output_index (int): The index in the output if the operator has multiple outputs. Usually output_index=0.
        options (dict): Bound options.
        device (str or torch.device): Device of the bounded module.
    """

    def __init__(self, input_name, name, ori_name, attr={}, inputs=[], output_index=0, options={}, device=None):
        super().__init__()

        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.options, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, options, device

        self.from_input = False

        self.perturbed = False
        self.forward_value = None
        # If set to true, the backward bound output of this node is 0.
        self.zero_backward_coeffs_l = False
        self.zero_backward_coeffs_u = False
        # If set to true, the A matrix accumulated on this node is 0.
        self.zero_lA_mtx = False
        self.zero_uA_mtx = False
        
    """save input and output shapes uniformly by the decorator"""
    @staticmethod
    def save_io_shape(func):
        def wrapper(self, *args, **kwargs):
            if len(args) > 0:
                # x should always be the first input
                self.input_shape = args[0].shape  

            output = func(self, *args, **kwargs)

            if isinstance(output, torch.Tensor):
                self.output_shape = output.shape
            return output

        return wrapper

    def infer_batch_dim(self, batch_size, *x):
        # Default implementation assuming the batch dimension is always at 0.
        # Do not use it if the operator can alter the shape
        assert x[0] in [0, -1]
        return x[0]