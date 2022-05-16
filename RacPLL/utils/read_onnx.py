import onnxruntime as ort
import onnx.numpy_helper
import torch.nn as nn
import numpy as np
import torch
import onnx


class NetworkTorchFromONNX(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers = nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


class ONNXParser:

    def __init__(self, filename):
        self.model = onnx.load(filename)
        self.torch_model = None
        self.transpose_needed = False
        self.transpose_needed_tested = False

        self.pytorch_model = self.to_pytorch()
        self.model_str = self.model.SerializeToString()
        self.sess = ort.InferenceSession(self.model_str)

        self.initializers = [i.name for i in self.model.graph.initializer]
        self.inputs = [i for i in self.model.graph.input if i.name not in self.initializers]

        self.input_name = self.inputs[0].name
        self.input_shape = tuple(d.dim_value for d in self.inputs[0].type.tensor_type.shape.dim)

    def to_pytorch(self):
        nodes = self.model.graph.node

        self.input_nodes = 1
        for dim in self.model.graph.input[0].type.tensor_type.shape.dim:
            self.input_nodes *= dim.dim_value
            
        self.output_nodes = 1
        for dim in self.model.graph.output[0].type.tensor_type.shape.dim:
            self.output_nodes *= dim.dim_value

        self._expect_unexpected_input = False
        for i in range(len(nodes)-1,-1,-1):
            if len(nodes[i].input) == 0:
                self._expect_unexpected_input = True
                del nodes[0]

        curr_input_idx = nodes[0].input[0]
        layers = []
        node_counts = self.input_nodes

        i = 0
        while i < len(nodes):
            layer, curr_input_idx, i, node_counts = self._process_node(curr_input_idx, nodes, i, node_counts)
            if layer is not None:
                layers.append(layer)
        return NetworkTorchFromONNX(layers)

    def _get_weights(self, node):
        return [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name in node.input]


    def _process_node(self, curr_input_idx, nodes, i, input_nodes):
        node = nodes[i]

        if curr_input_idx not in node.input and not self._expect_unexpected_input:
            print('cac')

        self._expect_unexpected_input = False
        curr_input_idx = node.output[0]

        if node.op_type in ["Sub", "Div"]:
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
            return nn.ReLU(), curr_input_idx, i + 1, input_nodes

        elif node.op_type in ["Flatten", "Shape", "Constant", "Gather", "Unsqueeze", "Concat", "Reshape"]:
            return None, curr_input_idx, i + 1, input_nodes

        elif node.op_type == "Gemm":
            return self._gemm_to_torch(node), curr_input_idx, i + 1, None

        elif node.op_type == "Conv":
            return self.conv_to_torch(node), curr_input_idx, i + 1, None


    def _gemm_to_torch(self, node) -> nn.Linear:
        [weights] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[2]]

        affine = nn.Linear(weights.shape[0], weights.shape[1])
        affine.weight.data = torch.Tensor(weights.copy())
        affine.bias.data = torch.Tensor(bias.copy())

        return affine

    def _conv_to_torch(self, node) -> nn.Linear:
        [weights] = [onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t).astype(float) for t in self.model.graph.initializer if t.name == node.input[2]]

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


    def __call__(self, x):
        return self.sess.run(None, {self.input_name: x.numpy()})[0]


