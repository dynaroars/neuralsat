import torch
from torch import Tensor

from ..extension import _assert_has_ops

__all__ = [
    'any_eq_any',
]


def any_eq_any(input: Tensor, other: Tensor) -> Tensor:
    r"""
    Tests if any row of :attr:`input` is equal to any element of :attr:`other`.

    Arguments:
        input (tensor): 2D input tensor of shape (batch_size, in_features).
        other (tensor): 1D tensor to be compared.
    """
    _assert_has_ops()
    return torch.ops.haioc.any_eq_any(input, other)
