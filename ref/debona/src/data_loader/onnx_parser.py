"""
A class for loading neural networks in onnx format and converting to torch.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import onnx
import onnx.numpy_helper
import numpy as np

from src.neural_networks.verinet_nn import VeriNetNN
from src.util.logger import get_logger
from src.util.config import *

logger = get_logger(LOGS_LEVEL, __name__, "../../logs/", "verifier_log")


class ONNXParser:
    def __init__(self, filepath: str):

        self.model = onnx.load(filepath)
        self.torch_model = None
        self.transpose_needed = False
        self.transpose_needed_tested = False

    def to_pytorch(self) -> VeriNetNN:
        """
        Converts the self.onnx model to a VeriNetNN(torch) model.

        Returns:
            The VeriNetNN model.
        """

        nodes = self.model.graph.node
        input_nodes = 1
        for dim in self.model.graph.input[0].type.tensor_type.shape.dim:
            input_nodes *= dim.dim_value
        output_nodes = 1
        for dim in self.model.graph.output[0].type.tensor_type.shape.dim:
            output_nodes *= dim.dim_value
        
        self._expect_unexpected_input = False
        for i in range(len(nodes)-1,-1,-1):
            if len(nodes[i].input) == 0:
                self._expect_unexpected_input = True
                del nodes[0]

        curr_input_idx = nodes[0].input[0]
        mappings = []

        i = 0
        while i < len(nodes):
            mapping, curr_input_idx, i, node_count = self._process_node(
                curr_input_idx, nodes, i, input_nodes)

            if mapping is not None:
                mappings.append(mapping)

        self.torch_model = VeriNetNN(mappings)

        return self.torch_model, input_nodes, output_nodes

    def _get_weights(self, node):
        return [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name in node.input]

    def _process_node(self, curr_input_idx: int, nodes: List[onnx.NodeProto],
            i: int, input_nodes: int) -> Tuple[torch.nn.Module, int, int, int]:
        """
        Processes a onnx node converting it to a corresponding torch node.

        Args:
            curr_input_idx:
                The expected onnx input index to the current node
            nodes:
                The onnx nodes
            i:
                The current node index
            input_nodes:
                The number of input nodes
        Returns:
                The corresponding torch.nn operation, current input id and next node id, number of output nodes
        """
        node = nodes[i]

        if curr_input_idx not in node.input and not self._expect_unexpected_input:
            logger.warning(
                f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
        self._expect_unexpected_input = False

        if node.op_type in ["Sub", "Div"]:
            print("Sub and Div layers are ignored, make sure that they don't influence the network output (this is true for the VNN2021 competition)")
            curr_input_idx = node.output[0]
            return None, curr_input_idx, i + 1, input_nodes
           
        elif node.op_type == "MatMul":
            [weights] = self._get_weights(node)
            if not self.transpose_needed_tested:
                if weights.shape[0] == input_nodes:
                    self.transpose_needed = False
                else:
                    self.transpose_needed = True
                self.transpose_needed_tested = True
            if self.transpose_needed:
                weights = weights.T

            if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                [bias] = self._get_weights(nodes[i + 1])
                next_i = i + 2
                curr_input_idx = nodes[i + 1].output[0]
            else:
                bias = np.zeros(weights.shape[1])
                next_i = i + 1
                curr_input_idx = node.output[0]
            layer = nn.Linear(weights.shape[0], weights.shape[1])
            layer.weight.data = torch.Tensor(weights.T.copy())
            layer.bias.data = torch.Tensor(bias.copy())
            return layer, curr_input_idx, next_i, weights.shape[1]

        elif node.op_type == "Relu":

            if len(node.input) != 1:
                logger.warning(
                    f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]
            return nn.ReLU(), curr_input_idx, i + 1, input_nodes

        elif node.op_type == "Sigmoid":

            if curr_input_idx != node.input[0]:
                logger.warning(
                    f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 1:
                logger.warning(
                    f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return nn.Sigmoid(), curr_input_idx, i + 1, input_nodes

        elif node.op_type == "Tanh":

            if curr_input_idx != node.input[0]:
                logger.warning(
                    f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 1:
                logger.warning(
                    f"Unexpected input length: \n{node}, expected {1}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return nn.Tanh(), curr_input_idx, i + 1, input_nodes

        elif node.op_type in ["Flatten", "Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:

            # Reshape operations are assumed to adhere to the standard used in VeriNetNN and thus skipped.
            curr_input_idx = node.output[0]

            logger.info(f"Skipped node of type: {node.op_type}")

            return None, curr_input_idx, i + 1, input_nodes

        elif node.op_type == "Gemm":

            if curr_input_idx != node.input[0]:
                logger.warning(
                    f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
            if len(node.input) != 3:
                logger.warning(
                    f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
            curr_input_idx = node.output[0]

            return self.gemm_to_torch(node), curr_input_idx, i + 1, None

        # Convolutions are not yet supported
        # elif node.op_type == "Conv":
        #     if curr_input_idx != node.input[0]:
        #         logger.warning(f"Unexpected input for node: \n{node}, expected {curr_input_idx}, got {node.input[0]}")
        #     if len(node.input) != 3:
        #         logger.warning(f"Unexpected input length: \n {node}, expected {3}, got {len(node.input)}")
        #     curr_input_idx = node.output[0]

        #     return self.conv_to_torch(node), curr_input_idx, i + 1

        else:
            logger.warning(f"Node not recognised: \n{node}")
            exit(1)
            return None, curr_input_idx, i + 1

    # noinspection PyArgumentList
    def gemm_to_torch(self, node) -> nn.Linear:
        """
        Converts a onnx 'gemm' node to a torch Linear.

        Args:
            node:
                The Gemm node.
        Returns:
            The torch Linear layer.
        """

        [weights] = [onnx.numpy_helper.to_array(
            t) for t in self.model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(
            t) for t in self.model.graph.initializer if t.name == node.input[2]]

        assert len(weights.shape) == 2
        affine = nn.Linear(weights.shape[0], weights.shape[1])
        affine.weight.data = torch.Tensor(weights.copy())
        affine.bias.data = torch.Tensor(bias.copy())

        return affine

    # noinspection PyArgumentList
    def conv_to_torch(self, node) -> nn.Linear:
        """
        Converts a onnx 'Conv' node to a torch Conv.

        Args:
            node:
                The Conv node.
        Returns:
            The torch Conv layer.
        """

        [weights] = [
            onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer
            if t.name == node.input[1]
        ]
        [bias] = [
            onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer
            if t.name == node.input[2]
        ]

        dilations = 1
        groups = 1
        pads = None
        strides = 1

        for att in node.attribute:
            if att.name == "dilations":
                dilations = [i for i in att.ints]
            elif att.name == "group":
                groups = att.i
            elif att.name == "pads":
                pads = [i for i in att.ints]
            elif att.name == "strides":
                strides = [i for i in att.ints]

        conv = nn.Conv2d(weights.shape[1],
                         weights.shape[0],
                         weights.shape[2:4],
                         stride=strides,
                         padding=pads[0:2],
                         groups=groups,
                         dilation=dilations)
        conv.weight.data = torch.Tensor(weights.copy())
        conv.bias.data = torch.Tensor(bias.copy())

        return conv
