from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs, select_model_inputs_outputs
from onnx.helper import ValueInfoProto, make_graph, make_model
from onnx2pytorch.convert.operations import convert_operations
import onnxruntime as ort
import onnx.numpy_helper
import torch.nn as nn
import onnx2pytorch
import numpy as np
import torch
import onnx
import math

from abstract.crown.utils import *
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

    # t = inputs[0].type.tensor_type.elem_type
    # print(inputs[0], onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE)
    # assert t == onnx.TensorProto.FLOAT
    # dtype = np.float32

    # if dtype == np.float32:
    #     elem_type = onnx.TensorProto.FLOAT
    # else:
    #     assert dtype == np.float64
    #     elem_type = onnx.TensorProto.DOUBLE
    elem_type = inputs[0].type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.FLOAT:
        dtype = np.float32
    elif elem_type == onnx.TensorProto.DOUBLE:
        dtype = np.float64
    else:
        raise elem_type
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
            # print(index, node)
            # print(marked_nodes)

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


def remove_unused_initializers(model):
    new_init = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            new_init.append(init)

    graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
                        model.graph.output, new_init)

    onnx_model = make_model_with_graph(model, graph)

    return onnx_model


class PyTorchModelWrapper(nn.Module):

    def __init__(self, layers):
        super().__init__()

        if isinstance(layers, list):
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = layers

        self.layers_mapping = None
        self.input_shape = None

        self.n_input = None
        self.n_output = None
        

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)


    def forward_grad(self, x):
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
                implication.update(dict(zip(self.layers_mapping[idx], s.flatten().numpy().astype(dtype=bool))))
                idx += 1
        return implication

    @torch.no_grad()
    def get_concrete(self, x):
        x = x.view(self.input_shape)
        idx = 0
        implication = {}
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                implication.update(dict(zip(self.layers_mapping[idx], x.view(-1))))
                idx += 1
            x = layer(x)
        return implication

    @torch.no_grad()
    def forward_layer(self, x, lid):
        relu_idx = 0
        # print(lid)
        for layer in self.layers:
            if isinstance(layer, nn.ReLU):
                relu_idx += 1
            if relu_idx <= lid:
                continue
            # print(layer)
            x = layer(x)
        return x


class Sub(nn.Module):

    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, x, y=None):
        if y is None:
            return x - self.constant
        return x - y



class Div(nn.Module):

    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, x, y=None):
        if y is None:
            return x / self.constant
        return x / y


class Add(nn.Module):

    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, x, y=None):
        if y is None:
            return x + self.constant
        return x + y


class ONNXParser:

    def __init__(self, filename, transpose_weight=False):
        self.model = remove_unused_initializers(onnx.load(filename))
        self.transpose_weight = transpose_weight
        # self.transpose_needed = True

        self.model_str = self.model.SerializeToString()
        self.sess = ort.InferenceSession(self.model_str)

        self.initializers = [i.name for i in self.model.graph.initializer]
        self.inputs = [i for i in self.model.graph.input if i.name not in self.initializers]

        self.input_name = self.inputs[0].name
        self.input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in self.inputs[0].type.tensor_type.shape.dim)
        if len(self.input_shape) == 1:
            self.input_shape = (1, self.input_shape[0])

        self.pytorch_model = self.to_pytorch()

        # elem_type = self.inputs[0].type.tensor_type.elem_type
        # if elem_type == onnx.TensorProto.FLOAT:
        #     dtype = np.float32
        # elif elem_type == onnx.TensorProto.DOUBLE:
        #     dtype = np.float64
        # else:
        #     raise elem_type

        # self.dtype = dtype

    def to_pytorch(self):

        # for i, node in enumerate(nodes):
        #     print(i, '------->\n', node.op_type)

        self.input_nodes = 1
        for dim in self.model.graph.input[0].type.tensor_type.shape.dim:
            self.input_nodes *= dim.dim_value if dim.dim_value != 0 else 1

        self.output_nodes = 1
        for dim in self.model.graph.output[0].type.tensor_type.shape.dim:
            self.output_nodes *= dim.dim_value if dim.dim_value != 0 else 1

        # nodes = self.model.graph.node
        # for i in range(len(nodes)-1,-1,-1):
        #     if len(nodes[i].input) == 0:
        #         del nodes[0]

        # curr_input_idx = nodes[0].input[0]

        # i = 0
        # while i < len(nodes):
        #     layer, curr_input_idx, i = self._process_node(curr_input_idx, nodes, i)
        #     if layer is not None:
        #         layers.append(layer)
        #         # print(i, layer)
        layers = []
        n_output = None
        should_transpose = True


        for op_id, op_name, op in convert_operations(self.model.graph, self.model.opset_import[0].version, 0, False):
            if isinstance(op, onnx2pytorch.operations.Constant):
                last_constant = op.constant
            elif isinstance(op, onnx2pytorch.operations.base.OperatorWrapper):
                if op.__class__.__name__ == 'sub':
                    if not torch.equal(last_constant, torch.zeros(last_constant.shape)):
                        layers.append(Sub(constant=last_constant))
            elif isinstance(op, onnx2pytorch.operations.Div):
                if not torch.equal(last_constant, torch.ones(last_constant.shape)):
                    layers.append(Div(constant=last_constant))
            elif isinstance(op, onnx2pytorch.operations.Flatten):
                layers.append(nn.Flatten())
            elif isinstance(op, onnx2pytorch.operations.Transpose):
                layers.append(op)
            elif isinstance(op, onnx2pytorch.operations.Reshape):
                layers.append(op)
            elif isinstance(op, nn.ReLU):
                layers.append(op)
            elif isinstance(op, nn.Linear):
                weight = op.weight.data

                if self.transpose_weight:
                    weight = weight.t()

                layer = nn.Linear(weight.shape[1], weight.shape[0])
                layer.weight.data = weight.to(settings.DTYPE)
                layer.bias.data = op.bias.data.to(settings.DTYPE)
                layers.append(layer)

                n_output = weight.shape[0]
            elif isinstance(op, nn.Conv2d):
                op.weight.data = op.weight.data.to(settings.DTYPE)
                op.bias.data = op.bias.data.to(settings.DTYPE)
                layers.append(op)
            else:
                print(op)
                raise

        # print('n_output', n_output)
        model = PyTorchModelWrapper(layers)
        model.n_input = self.input_nodes
        model.n_output = n_output
        model.input_shape = self.input_shape
        return model

    def _get_weights(self, node):
        return [onnx.numpy_helper.to_array(t) for t in self.model.graph.initializer if t.name in node.input]


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
                         padding=pads[0:2] if pads else None,
                         groups=groups,
                         dilation=dilations)
        conv.weight.data = torch.Tensor(weights.copy()).to(settings.DTYPE)
        conv.bias.data = torch.Tensor(bias.copy()).to(settings.DTYPE)

        return conv


    def __call__(self, x):
        return self.sess.run(None, {self.input_name: x.numpy()})[0]


    def extract_ordered_relu_shapes(self):
        relu_nodes = extract_ordered_relus(self.model, self.input_name)
        assert relu_nodes, "expected at least one relu layer in network"
        io_shapes = get_io_shapes(self.model)
        return [io_shapes[r.input[0]] for r in relu_nodes]




class ONNXParser2:

    def __init__(self, filename, dataset):

        if dataset == 'mnist':
            input_shape = (1, 1, 28, 28)
            n_output = 10
        elif dataset == 'cifar':
            input_shape = (1, 3, 32, 32)
            n_output = 10
        else:
            raise 

        model, is_channel_last = load_model_onnx(filename, input_shape=input_shape[1:])

        if is_channel_last:
            input_shape = input_shape[:1] + input_shape[2:] + input_shape[1:2]
            print(f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {input_shape}')

        self.pytorch_model = PyTorchModelWrapper(model)
        self.pytorch_model.n_input = math.prod(input_shape)
        self.pytorch_model.n_output = n_output
        self.pytorch_model.input_shape = input_shape
