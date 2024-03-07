from functools import reduce
import operator

import torch
from torch import nn

from onnx2pytorch.operations.base import Operator
from onnx2pytorch.utils import assign_values_to_dim, get_selection, PRINT_DEBUG


def prod(x):
    return reduce(operator.mul, x, 1)

# https://github.com/KaidiXu/onnx2pytorch/

class Reshape(Operator):
    """
    In the initial pass it stores the initial_input_shape.
    It uses it to infer the new reshape value from a
    smaller pruned input in the following passes.
    """

    def __init__(self, enable_pruning, shape=None, keep_size=True, quirks=None):
        super().__init__()
        self.enable_pruning = enable_pruning
        self.shape = shape
        self.initial_input_shape = None
        self.feature_dim = -1
        self.input_indices = None
        self.placeholder = None
        self.keep_size = keep_size
        self.quirks = {} if quirks is None else quirks
        assert isinstance(self.quirks, dict)

    def forward(self, input: torch.Tensor, shape=None):
        shape = shape if shape is not None else self.shape
        if shape[0] == 1 and len(shape) in [2, 3, 4, 5] and self.quirks.get("fix_batch_size") is True:
            incomplete_indices = (shape == -1).nonzero()
            assert incomplete_indices.numel() <= 1, "at most one dimension can be -1 in reshape"
            if incomplete_indices.numel() == 1 and shape[0] != -1:
                # Have a -1 shape not at the batch dimension.
                incomplete_loc = incomplete_indices[0, 0].item()
                # Need to compute the actual shape if we already have a -1.
                incomplete_shape = -1 * torch.prod(shape[1:])
                inferred_shape = prod(input.shape[1:]) // incomplete_shape
                shape[incomplete_loc] = inferred_shape
            shape[0] = -1
            if self.initial_input_shape is None:
                print("Enabling quirks for Reshape operation: fix the first " "dimension shape to be -1 to support batchsize != 1.")
                print(f"input shape {input.shape}, new shape is {shape}.")
        else:
            # This raises RuntimeWarning: iterating over a tensor.
            # FIXME: this looks not right.
            # if the first dim is batch size, merge the batch size with the channel dim
            if (torch.prod(torch.tensor(input.shape)) != torch.prod(shape) and len(input.size()) == len(shape) + 1
                    and torch.prod(torch.tensor(input.shape)) == input.shape[0] * torch.prod(shape)
                    and self.quirks.get("merge_batch_size_with_channel") is True):
                shape = torch.tensor([input.shape[0] * input.shape[1] // 2] + shape.tolist()[1:], device=shape.device)
            shape = [x if x != 0 else input.size(i) for i, x in enumerate(shape)]
        if not self.enable_pruning:
            return torch.reshape(input, tuple(shape))

        inp_shape = torch.tensor(input.shape)
        if self.initial_input_shape is None:
            self.initial_input_shape = inp_shape
        elif len(shape) == 2 and shape[-1] == -1:
            pass
        elif torch.equal(self.initial_input_shape, inp_shape):
            # input's shape did not change
            pass
        elif self.input_indices is not None:
            self.placeholder *= 0
            selection = get_selection(self.input_indices, self.feature_dim)
            self.placeholder[selection] += input
            input = self.placeholder
        elif torch.prod(inp_shape) == torch.prod(torch.tensor(shape)):
            # If input's shape changed but shape changed to account for this,
            # no additional work is needed.
            # This happens when shape is dynamically computed by the network.
            pass
        else:
            # If input's shape changed but shape has not accounted for this,
            # the reshaped shape must change as well.
            c = torch.true_divide(inp_shape, self.initial_input_shape)
            if len(c) < len(shape) and shape[0] == 1:
                c = torch.cat((torch.tensor([1]), c))
            shape = (c * torch.tensor(shape)).to(int)
        return torch.reshape(input, tuple(shape))

    def set_input_indices(self, input):
        input_shape = input[0].shape
        if self.feature_dim < 0:
            self.feature_dim += len(input_shape)
        axis = self.get_axis(input_shape, self.feature_dim)
        mask = input[0] != 0
        s = mask.sum(axis=tuple(axis))
        mask = s != 0
        (non_zeros,) = torch.where(mask)
        self.input_indices = non_zeros
        self.placeholder = nn.Parameter(
            torch.zeros(*self.initial_input_shape, device=input[0].device, dtype=input[0].dtype),
            requires_grad=False,
        )

    def extra_repr(self) -> str:
        return "shape={}".format(self.shape)
