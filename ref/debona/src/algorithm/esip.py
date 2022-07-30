"""
This file contains the Error-based Symbolic Interval Propagation (ESIP)

ESIP Calculates linear bounds on the networks output nodes, given bounds on the networks input

The current implementation supports box-constraints on the input. In the future we plan to extend this to L1,
Brightness and Contrast constraints.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import Optional, List

import torch
import torch.nn as nn
import numpy as np

from src.algorithm.mappings.abstract_mapping import AbstractMapping
from src.neural_networks.verinet_nn import VeriNetNN
from src.algorithm.esip_util import concretise_symbolic_bounds_jit


class ESIP:
    """
    Implements the error-based symbolic interval propagation (ESIP) algorithm
    """

    def __init__(self, model: VeriNetNN, input_shape):
        """
        Args:

            model                       : The VeriNetNN neural network as defined in src/neural_networks/verinet_nn.py
            input_shape                 : The shape of the input, (input_size,) for 1D input or
                                          (channels, height, width) for 2D.
        """

        self._model = model
        self._input_shape = input_shape

        self._mappings: List[AbstractMapping]
        self._layer_sizes: list
        self._layer_shapes: list

        self._bounds_concrete_cached: list
        self._bounds_symbolic_cached: list

        self._split_impact_indicator: list
        self._error_matrix_to_node_indices: list

        self._forced_input_bounds: list

        self._read_mappings_from_torch_model(model)
        self._init_datastructure()

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @property
    def num_layers(self):
        return len(self.layer_sizes)

    @property
    def mappings(self):
        return self._mappings

    @property
    def forced_input_bounds(self):
        return self._forced_input_bounds

    @forced_input_bounds.setter
    def forced_input_bounds(self, val: np.array):
        self._forced_input_bounds = val

    def _get_symbolic_equation(self,
                               layer_num: int,
                               pseudo_last_layer: np.array = None) -> np.array:
        """
        Calculates the symboli bounds for.

        Requests are cached if possible, to avoid costly recomputations.

        Args:
            layer_num    : The layer number
            pseudo_last_layer : An additional layer on top of the normal output, to compute relationships between outputs

        Returns:
            The symbolic upper and lower bounds
        """

        if layer_num == -1:
            layer_num = self.num_layers - 1
        assert len(self._mappings) == self.num_layers
        if layer_num == 0:
            return np.tile(np.eye(self.layer_sizes[0], self.layer_sizes[0] + 1), reps=(2, 1, 1))

        if layer_num == self.num_layers or self._bounds_symbolic_cached[layer_num] is None:
            if pseudo_last_layer is not None:
                assert layer_num == self.num_layers, "%d != %d" % (
                    layer_num, self.num_layers)
                symbolic_bounds = pseudo_last_layer
            else:
                symbolic_bounds = np.tile(np.eye(self.layer_sizes[layer_num], self.layer_sizes[layer_num] + 1),
                                          reps=(2, 1, 1))
            for current_layer_num in range(min(layer_num, self.num_layers - 1), 0, -1):
                if pseudo_last_layer is not None:
                    self.node_impact[current_layer_num] += np.maximum(
                        0, symbolic_bounds[1, 0, :-1])
                mapping = self._mappings[current_layer_num]
                if mapping.is_linear:
                    new_symbolic_bounds = mapping.propagate_back(
                        symbolic_bounds)
                else:
                    pos_symbolic_bounds = np.where(
                        symbolic_bounds > 0, symbolic_bounds, 0)
                    neg_symbolic_bounds = np.where(
                        symbolic_bounds < 0, symbolic_bounds, 0)
                    new_symbolic_bounds = np.empty_like(symbolic_bounds)
                    new_symbolic_bounds[:, :, :-1] = (pos_symbolic_bounds[:, :, :-1] *
                                                      mapping.params["relaxations"][:, :, 0][:, np.newaxis, :])
                    new_symbolic_bounds[0, :, :-1] += (neg_symbolic_bounds[0, :, :-1] *
                                                       mapping.params["relaxations"][1, :, 0][np.newaxis, :])
                    new_symbolic_bounds[1, :, :-1] += (neg_symbolic_bounds[1, :, :-1] *
                                                       mapping.params["relaxations"][0, :, 0][np.newaxis, :])
                    added_relaxation_bias = np.sum(
                        (pos_symbolic_bounds[:, :, :-1] *
                         mapping.params["relaxations"][:, :, 1][:, np.newaxis, :]),
                        axis=2)
                    added_relaxation_bias[0, :] += np.sum(
                        (neg_symbolic_bounds[0, :, :-1] *
                         mapping.params["relaxations"][1, :, 1][np.newaxis, :]),
                        axis=1)
                    added_relaxation_bias[1, :] += np.sum(
                        (neg_symbolic_bounds[1, :, :-1] *
                         mapping.params["relaxations"][0, :, 1][np.newaxis, :]),
                        axis=1)
                    new_symbolic_bounds[:, :, -1] = symbolic_bounds[:,
                                                                    :, -1] + added_relaxation_bias
                symbolic_bounds = new_symbolic_bounds
            if self.forced_input_bounds[0] is None:
                symbolic_difference = symbolic_bounds[1] - symbolic_bounds[0]
                input_constraints = np.stack(
                    [self.input_constraints, self.input_constraints], axis=0)
                concrete_difference = concretise_symbolic_bounds_jit(
                    input_constraints, symbolic_difference)
                assert np.all(concrete_difference[:, 0] + 0.01 >= 0), (layer_num, concrete_difference,
                                                                       symbolic_difference)

            if layer_num == self.num_layers:
                return symbolic_bounds
            else:
                self._bounds_symbolic_cached[layer_num] = symbolic_bounds

        return self._bounds_symbolic_cached[layer_num]

    def reset_datastruct(self):
        """
        Resets the symbolic datastructure
        """

        self._init_datastructure()

    def calc_bounds(self, input_constraints: np.array, from_layer: int = 1) -> bool:
        """
        Calculate the bounds for all layers in the network starting at from_layer.

        Notice that from_layer is usually larger than 1 after a split. In this case, the split constraints are
        added to the layer before from_layer by adjusting the forced bounds. For this reason, we update the
        concrete_bounds for the layer before from_layer.

        Args:
            input_constraints       : The constraints on the input. The first dimensions should be the same as the
                                      input to the neural network, the last dimension should contain the lower bound
                                      on axis 0 and the upper on axis 1.
            from_layer              : Updates this layer and all later layers

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds are invalid if the forced bounds
            make at least one upper bound smaller than a lower bound.
        """

        assert from_layer >= 1, "From layer should be >= 1"
        assert isinstance(input_constraints,
                          np.ndarray), "input_constraints should be a np array"

        self.input_constraints = input_constraints
        for layer_num in range(from_layer, self.num_layers):
            self._bounds_symbolic_cached[layer_num] = None
            self._bounds_concrete_cached[layer_num] = None
            if "relaxations" in self._mappings[layer_num].params:
                del self._mappings[layer_num].params["relaxations"]

        # Concrete bounds from previous layer might have to be recalculated due to new split-constraints
        if from_layer > 1:
            self._bounds_concrete_cached[from_layer - 1] = None

        self.node_impact = []
        for i in range(self.num_layers):
            self.node_impact.append(np.zeros(self.layer_sizes[i]))

        for layer_num in range(from_layer, self.num_layers):
            self._prop_bounds_and_errors(layer_num)

        return True

    def _prop_bounds_and_errors(self, layer_num: int):
        """
        Calculates the symbolic input bounds.

        This updates all bounds, relaxations and errors for the given layer, by propagating from the previous layer.

        Args:
            layer_num: The layer number
        """

        mapping = self._mappings[layer_num]

        if mapping.is_linear:
            self._split_impact_indicator[layer_num] = mapping.propagate(self._split_impact_indicator[layer_num - 1],
                                                                        add_bias=False)
            self._error_matrix_to_node_indices[layer_num] = self._error_matrix_to_node_indices[layer_num - 1].copy()
        else:
            relaxations = self._calc_relaxations(layer_num)
            self._mappings[layer_num].params["relaxations"] = relaxations

            self._split_impact_indicator[layer_num], self._error_matrix_to_node_indices[layer_num] = \
                self._prop_impact_through_relaxation(relaxations, layer_num)

        # We already test the validity of the bounds after each computation, no need to check again (and spend a lot
        # of time doing so)

    def _get_concrete_bounds(self, layer_num: int) -> tuple:
        """
        Calculates the concrete upper and lower bounds.

        Args:
            layer_num : The layer number
        Returns
            (2, node, 2), where the first axis is lower/upper bound, and the third axis are their respective concrete lower and upper values
        """

        if self._bounds_concrete_cached[layer_num] is None:
            combined_lower_symbolic_bounds, combined_upper_symbolic_bounds = self._get_symbolic_equation(
                layer_num)

            input_constraints = np.stack(
                [self.input_constraints, self.input_constraints], axis=0)
            concrete_lower_bounds = concretise_symbolic_bounds_jit(input_constraints,
                                                                   combined_lower_symbolic_bounds)[:]
            concrete_upper_bounds = concretise_symbolic_bounds_jit(input_constraints,
                                                                   combined_upper_symbolic_bounds)[:]

            concrete_bounds = np.stack(
                [concrete_lower_bounds, concrete_upper_bounds], axis=0)
            concrete_bounds = self._adjust_bounds_from_forced_bounds(concrete_bounds,
                                                                     self._forced_input_bounds[layer_num])

            self._bounds_concrete_cached[layer_num] = concrete_bounds

        assert not np.any(np.isnan(self._bounds_concrete_cached[layer_num]))
        return self._bounds_concrete_cached[layer_num]

    @staticmethod
    def _adjust_bounds_from_forced_bounds(bounds_concrete: np.array, forced_input_bounds: np.array) -> np.array:
        """
        Adjusts the concrete input bounds using the forced bounds.

        The method chooses the best bound from the stored concrete input bounds and the forced bounds as the new
        concrete input bound.

        Args:
            bounds_concrete     : A 2xNx2 array with the concrete lower and upper bounds, where N is the number of Nodes.
            forced_input_bounds : A 2xNx2 array with the forced input bounds used for adjustment, where N is the number
                                  of Nodes.
        Returns:
            A 2xNx2 array with the adjusted concrete bounds, where N is the number of Nodes.
        """

        bounds_concrete_new = bounds_concrete.copy()

        if forced_input_bounds is None:
            return bounds_concrete_new

        forced_lower = forced_input_bounds[:, 0:1]
        forced_upper = forced_input_bounds[:, 1:2]

        smaller_idx = bounds_concrete[0] <= forced_lower
        bounds_concrete_new[0][smaller_idx] = np.hstack(
            (forced_lower, forced_lower))[smaller_idx]
        smaller_idx = bounds_concrete[1] <= forced_lower
        bounds_concrete_new[1][smaller_idx] = np.hstack(
            (forced_lower, forced_lower))[smaller_idx]

        larger_idx = bounds_concrete[0] >= forced_upper
        bounds_concrete_new[0][larger_idx] = np.hstack(
            (forced_upper, forced_upper))[larger_idx]
        larger_idx = bounds_concrete[1] >= forced_upper
        bounds_concrete_new[1][larger_idx] = np.hstack(
            (forced_upper, forced_upper))[larger_idx]

        return bounds_concrete_new

    def _calc_relaxations(self, layer_num: int) -> np.array:
        """
        Calculates the linear relaxations for the given mapping and concrete bounds. .

        Args:
            layer_num : The layer number for which to compute the relaxations
        Returns:
            A 2xNx2 array where the first dimension indicates the lower and upper relaxation, the second dimension
            are the nodes in the current layer and the last dimension contains the parameters [a, b] in
            l(x) = ax + b.
        """
        mapping = self._mappings[layer_num]
        bounds_concrete = self._get_concrete_bounds(layer_num - 1)
        do_test = self.forced_input_bounds[0] is None
        lower_relaxation = mapping.linear_relaxation(
            bounds_concrete, False, do_test)
        upper_relaxation = mapping.linear_relaxation(
            bounds_concrete, True, do_test)

        return np.stack([lower_relaxation, upper_relaxation], axis=0)

    def _prop_impact_through_relaxation(self, relaxations: np.array, layer_num: int) -> tuple:
        """
        Updates the error matrix and the error_matrix_to_node_indices array.

        The old errors are propagated through the lower relaxations and the new errors due to the relaxations at this
        layer are concatenated the result.

        Args:
            relaxations                 : A 2xNx2 array where the first dimension indicates the lower and upper
                                          relaxation, the second dimension contains the nodes in the current layer
                                          and the last dimension contains the parameters [a, b] in l(x) = ax + b.
            layer_num                   : The number of the current layer, used to update error_matrix_to_node_indices.

        Returns:
            (error_matrix_new, error_matrix_to_node_indices_new), where both are on the same form as explained in the
            input arguments.
        """
        # Get the relaxation parameters
        a_low, a_up = relaxations[0, :, 0], relaxations[1, :, 0]
        b_low, b_up = relaxations[0, :, 1], relaxations[1, :, 1]

        zeroed_mask = np.all(
            [a_low == 0, b_low == 0, a_up == 0, b_up == 0], axis=0)
        change_mask = np.any(
            [a_low != 1, b_low != 0, a_up != 1, b_up != 0], axis=0)
        err_idx = np.argwhere(np.logical_and(
            change_mask, 1 - zeroed_mask))[:, 0]
        assert err_idx.ndim == 1
        num_err = err_idx.shape[0]

        split_impact_indicator = self._split_impact_indicator[layer_num - 1]
        num_old_err = split_impact_indicator.shape[1]
        layer_size = split_impact_indicator.shape[0]
        split_impact_indicator_new = np.zeros(
            (layer_size, num_old_err + num_err), np.float32)

        if num_old_err > 0:
            split_impact_indicator_new[:, :num_old_err] = a_up[:,
                                                               np.newaxis] * split_impact_indicator

        # Calculate the new errors.
        error_matrix_to_node_indices = self._error_matrix_to_node_indices[layer_num - 1]
        if num_err > 0:
            bounds_concrete = self._get_concrete_bounds(layer_num - 1)
            # Calculate the error at the lower and upper input bound.
            error_lower = (
                (bounds_concrete[0][:, 0] * a_up + b_up) - (bounds_concrete[0][:, 0] * a_low + b_low))
            error_upper = (
                (bounds_concrete[1][:, 1] * a_up + b_up) - (bounds_concrete[1][:, 1] * a_low + b_low))
            max_err = np.max((error_lower, error_upper), axis=0)
            split_impact_indicator_new[:, num_old_err:][err_idx, np.arange(
                num_err)] = max_err[err_idx]

            error_matrix_to_node_indices_new = np.hstack(
                (np.zeros(err_idx.shape, dtype=int)[:, np.newaxis] + layer_num, err_idx[:, np.newaxis]))
            error_matrix_to_node_indices_new = np.vstack(
                (error_matrix_to_node_indices, error_matrix_to_node_indices_new))
        else:
            error_matrix_to_node_indices_new = error_matrix_to_node_indices.copy()

        return split_impact_indicator_new, error_matrix_to_node_indices_new

    def merge_current_bounds_into_forced(self):
        """
        Sets forced input bounds to the best of current forced bounds and calculated bounds.
        """

        for i in range(self.num_layers):
            concrete_bounds = self._get_concrete_bounds(i)
            if concrete_bounds is None:
                continue

            elif self.forced_input_bounds[i] is None:
                self.forced_input_bounds[i] = np.stack(
                    [concrete_bounds[0][:, 0], concrete_bounds[1][:, 1]], axis=1)

            else:
                better_lower = self.forced_input_bounds[i][:,
                                                           0] < concrete_bounds[0][:, 0]
                self.forced_input_bounds[i][better_lower,
                                            0] = concrete_bounds[0][better_lower, 0]

                better_upper = self.forced_input_bounds[i][:,
                                                           1] > concrete_bounds[1][:, 1]
                self.forced_input_bounds[i][better_upper,
                                            1] = concrete_bounds[1][better_upper, 1]

    def largest_error_split_node(self, output_weights: np.array = None) -> Optional[tuple]:
        """
        Returns the node with the largest weighted error effect on the output

        The error from overestimation is calculated for each output node with respect to each hidden node.
        This value is weighted using the given output_weights and the index of the node with largest effect on the
        output is returned. Nodes from early layers are selected first.

        Args:
            output_weights  : A Nx2 array with the weights for the lower bounds in column 1 and the upper bounds
                              in column 2. All weights should be >= 0.
        Returns:
              (layer_num, node_num) of the node with largest error effect on the output
        """
        if self._split_impact_indicator[-1].shape[1] == 0:
            return None

        earliest_layer_num = np.min(
            self._error_matrix_to_node_indices[-1][:, 0])
        node_mask = self._error_matrix_to_node_indices[-1][:,
                                                           0] == earliest_layer_num
        selected_nodes = self._error_matrix_to_node_indices[-1][node_mask][:, 1]
        max_err_idx = np.argmax(
            self.node_impact[earliest_layer_num][selected_nodes])

        return self._error_matrix_to_node_indices[-1][max_err_idx]

    def _init_datastructure(self):
        """
        Initialises the data-structure.
        """

        num_layers = len(self.layer_sizes)

        self._bounds_concrete_cached = [None] * num_layers
        self._bounds_symbolic_cached = [None] * num_layers

        self._forced_input_bounds = [None] * num_layers

        self._split_impact_indicator = [None] * num_layers
        self._error_matrix_to_node_indices = [None] * num_layers
        self._error = [None] * num_layers

        # Set the error matrices of the input layer to zero
        self._split_impact_indicator[0] = np.zeros(
            (self.layer_sizes[0], 0), dtype=np.float32)
        self._error_matrix_to_node_indices[0] = np.zeros((0, 2), dtype=int)

    def _read_mappings_from_torch_model(self, torch_model: VeriNetNN):
        """
        Initializes the mappings from the torch model.

        Args:
            torch_model : The Neural Network
        """

        # Initialise with None for input layer
        self._mappings = [None]
        self._layer_shapes = [self._input_shape]

        for layer in torch_model.layers:
            self._process_layer(layer)

        self._layer_sizes = [int(np.prod(shape))
                             for shape in self._layer_shapes]

    def _process_layer(self, layer: nn.Module):
        """
        Processes the mappings (Activation function, FC, Conv, ...) for the given "layer".

        Reads the mappings from the given layer, adds the relevant abstraction to self._mappings and calculates the data
        shape after the mappings.

        Args:
            layer: The layer number
        """

        # Recursively process Sequential layers
        if isinstance(layer, nn.Sequential):
            for child in layer:  # type: ignore
                self._process_layer(child)
            return

        # Add the mapping
        try:
            self._mappings.append(
                AbstractMapping.get_activation_mapping_dict()[layer.__class__]())
        except KeyError as e:
            raise MappingNotImplementedException(
                f"Mapping: {layer} not implemented") from e

        # Add the necessary parameters (Weight, bias....)
        for param in self._mappings[-1].required_params:

            attr = getattr(layer, param)

            if isinstance(attr, torch.Tensor):
                self._mappings[-1].params[param] = attr.detach().numpy()
            else:
                self._mappings[-1].params[param] = attr

        # Calculate the output shape of the layer
        self._mappings[-1].params["in_shape"] = self._layer_shapes[-1]
        self._layer_shapes.append(
            self._mappings[-1].out_shape(self._layer_shapes[-1]))


class BoundsException(Exception):
    pass


class MappingNotImplementedException(BoundsException):
    pass
