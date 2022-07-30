"""
This file contains abstractions for layers (FC, Conv ...).

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np
import torch.nn as nn

from src.algorithm.mappings.abstract_mapping import AbstractMapping, \
    ActivationFunctionAbstractionException


class FC(AbstractMapping):
    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return False

    @property
    def required_params(self) -> list:
        return ["weight", "bias"]

    @classmethod
    def abstracted_torch_funcs(cls) -> list:
        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.linear.Linear]

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:
        """
        Propagates through the mapping by applying the fully-connected mapping.

        Args:
            x           : The input as a np.array.
                          Assumed to be a NxM vector where the rows represent nodes and the columns represent
                          coefficients of the symbolic bounds. Can be used on concrete values instead of equations by
                          shaping them into an Nx1 array.
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """
        w = self.params["weight"]

        if x.ndim == 2:
            x = w @ x

            if add_bias:
                x[:, -1] += self.params["bias"]

            return x
        else:
            assert x.ndim == 3  # orig_node, node, param
            new_x = w @ x

            assert not add_bias

            return new_x

    def propagate_back(self, symbolic_bounds: np.array) -> np.array:
        weight = self.params["weight"]
        bias = self.params["bias"]
        new_symbolic_bounds = np.empty(
            (2, symbolic_bounds.shape[1], weight.shape[1] + 1))
        new_symbolic_bounds[:, :, -1] = symbolic_bounds[:, :, -1] + np.sum(
            symbolic_bounds[:, :, :-1] * bias[np.newaxis, np.newaxis, :], axis=2)
        new_symbolic_bounds[:, :, :-1] = symbolic_bounds[:, :, :-1] @ weight
        return new_symbolic_bounds

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:
        """
        Not implemented since function is linear.
        """

        msg = f"linear_relaxation(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    # noinspection PyTypeChecker
    def split_point(self, xl: float, xu: float) -> float:
        """
        Not implemented since function is linear.
        """

        msg = f"split_point(...) not implemented for {self.__name__} since it is linear"
        raise ActivationFunctionAbstractionException(msg)

    def out_shape(self, in_shape: np.array) -> np.array:
        """
        Returns the output-shape of the data as seen in the original network.
        """

        return np.array((self.params["weight"].shape[0]))
