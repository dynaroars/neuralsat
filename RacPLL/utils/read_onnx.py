from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs
from onnx.helper import ValueInfoProto, make_graph, make_model
import onnxruntime as ort
import onnx.numpy_helper
import torch.nn as nn
import numpy as np
import torch
import onnx

import settings

def make_model_with_graph(model, graph, check_model=True):
    'copy a model with a new graph'

    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # fix opset import
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if check_model:
        onnx.checker.check_model(onnx_model, full_check=True)

    return onnx_model


def get_io_shapes(model):
    """returns map io_name -> shape"""

    # model = remove_unused_initializers(model)

    rv = {}

    intermediate_outputs = list(enumerate_model_node_outputs(model))

    initializers = [i.name for i in model.graph.initializer]
    inputs = [i for i in model.graph.input if i.name not in initializers]
    assert len(inputs) == 1

    t = inputs[0].type.tensor_type.elem_type
    assert t == onnx.TensorProto.FLOAT
    dtype = np.float32

    if dtype == np.float32:
        elem_type = onnx.TensorProto.FLOAT
    else:
        assert dtype == np.float64
        elem_type = onnx.TensorProto.DOUBLE

    # create inputs as zero tensors
    input_map = {}

    for inp in inputs:            
        shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
        
        input_map[inp.name] = np.zeros(shape, dtype=dtype)

        # also save it's shape
        rv[inp.name] = shape

    new_out = []

    # add all old outputs
    for out in model.graph.output:
        new_out.append(out)
        
    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        value_info = ValueInfoProto()
        value_info.name = out_name
        new_out.append(value_info)

    # ok run once and get all outputs
    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                       new_out, model.graph.initializer)

    new_onnx_model = make_model_with_graph(model, graph, check_model=False)
    
    sess = ort.InferenceSession(new_onnx_model.SerializeToString())

    res = sess.run(None, input_map)
    names = [o.name for o in sess.get_outputs()]
    out_map = {name: output for name, output in zip(names, res)}

    for out_name in intermediate_outputs:
        if out_name in rv: # inputs were already added
            continue

        rv[out_name] = out_map[out_name].shape
        
    return rv


def extract_ordered_relus(model, start):
    relu_nodes = []
    marked_values = [start]
    marked_nodes = []

    modified = True

    while modified:
        modified = False

        for index, node in enumerate(model.graph.node):

            # node was already processed
            if index in marked_nodes:
                continue

            should_process = False

            for inp in node.input:
                if inp in marked_values:
                    should_process = True
                    break

            # none of the node's inputs were marked
            if not should_process:
                continue

            # process the node!
            modified = True
            marked_nodes.append(index)

            if node.op_type == 'Relu':
                relu_nodes.append(node)

            for out in node.output:
                if out in marked_values:
                    continue

                marked_values.append(out)

    return relu_nodes


class PyTorchModelWrapper(nn.Module):

    def __init__(self, layers):
        super().__init__()

        self.layers = nn.Sequential(*layers)

        self.layers_mapping = None
        self.input_shape = None

        self.n_input = None
        self.n_output = None
        

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


    @torch.no_grad()
    def get_assignment(self, x):
        idx = 0
        implication = {}
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                s = torch.zeros_like(x, dtype=int) 
                s[x > 0] = 1
                implication.update(dict(zip(self.layers_mapping[idx], s.numpy().astype(dtype=bool))))
                idx += 1
        return implication

class ONNXParser:

    def __init__(self, filename):
        self.model = onnx.load(filename)
        # self.transpose_needed = True

        self.model_str = self.model.SerializeToString()
        self.sess = ort.InferenceSession(self.model_str)

        self.initializers = [i.name for i in self.model.graph.initializer]
        self.inputs = [i for i in self.model.graph.input if i.name not in self.initializers]

        self.input_name = self.inputs[0].name
        self.input_shape = tuple(d.dim_value for d in self.inputs[0].type.tensor_type.shape.dim)

        self.pytorch_model = self.to_pytorch()

    def to_pytorch(self):
        nodes = self.model.graph.node

        # for i, node in enumerate(nodes):
        #     print(i, '------->\n', node.op_type)

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

        i = 0
        while i < len(nodes):
            layer, curr_input_idx, i = self._process_node(curr_input_idx, nodes, i)
            if layer is not None:
                layers.append(layer)
                # print(i, layer)

        model = PyTorchModelWrapper(layers)
        model.n_input = self.input_nodes
        model.n_output = layers[-1].weight.shape[0]
        model.input_shape = self.input_shape
        return model

    def _get_weights(self, node):
        return [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name in node.input]


    def _process_node(self, curr_input_idx, nodes, i):
        node = nodes[i]

        self._expect_unexpected_input = False
        curr_input_idx = node.output[0]

        if node.op_type == "MatMul":

            [weights] = self._get_weights(node)

            # if self.transpose_needed:
            #     weights = weights.T

            if i + 1 < len(nodes) and nodes[i + 1].op_type == "Add":
                [bias] = self._get_weights(nodes[i + 1])
                next_i = i + 2
                curr_input_idx = nodes[i + 1].output[0]
            else:
                bias = np.zeros(weights.shape[0])
                next_i = i + 1
                curr_input_idx = node.output[0]
            # print(weights.shape)
            layer = nn.Linear(weights.shape[1], weights.shape[0])
            layer.weight.data = torch.Tensor(weights.copy()).to(settings.DTYPE)
            layer.bias.data = torch.Tensor(bias.copy()).to(settings.DTYPE)
            return layer, curr_input_idx, next_i

        elif node.op_type == "Relu":
            return nn.ReLU(), curr_input_idx, i + 1

        elif node.op_type == "Flatten":
            layer = nn.Flatten()
            return layer, curr_input_idx, i + 1

        elif node.op_type == "Gemm":
            return self._gemm_to_torch(node), curr_input_idx, i + 1

        elif node.op_type == "Conv":
            return self._conv_to_torch(node), curr_input_idx, i + 1

        else:
            print('Discard:', node.op_type)
            return None, curr_input_idx, i + 1



    def _gemm_to_torch(self, node) -> nn.Linear:
        [weights] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[1]]
        [bias] = [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name == node.input[2]]

        layer = nn.Linear(weights.shape[1], weights.shape[0])
        layer.weight.data = torch.Tensor(weights.copy()).to(settings.DTYPE)
        layer.bias.data = torch.Tensor(bias.copy()).to(settings.DTYPE)

        return layer

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
        conv.weight.data = torch.Tensor(weights.copy()).to(settings.DTYPE)
        conv.bias.data = torch.Tensor(bias.copy()).to(settings.DTYPE)

        return conv


    def __call__(self, x):
        return self.sess.run(None, {self.input_name: x.numpy()})[0]


    def extract_ordered_relu_shapes(self):
        relu_nodes = extract_ordered_relus(self.model, self.input_name)
        io_shapes = get_io_shapes(self.model)
        return [io_shapes[r.input[0]] for r in relu_nodes]
