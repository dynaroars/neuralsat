import struct
import torch
import torch.nn as nn
import warnings
from tools.custom_torch_modules import Flatten, Transpose, View, Reshape, Add, Mul

"""
Adapted from: https://gist.github.com/qinjian623/6aa777037534c1c1dccbb66f832e93b8
"""

# TODO more types maybe?
data_type_tab = {
    1: ['f', 4],
    2: ['B', 1],
    3: ['b', 1],
    4: ['H', 2],
    5: ['h', 2],
    6: ['i', 4],
    7: ['q', 8],
    10: ['e', 2],
    11: ['d', 8],
    12: ['I', 4],
    13: ['Q', 8]
}

torch_type_tab = {
    1: torch.float32,
    7: torch.int64,
    11: torch.float64,
}


class ResNetError(Exception):
    def __init__(self, message):
        super().__init__(message)


def empty(x):
    return x


# TODO pytorch only accepts 2-value list for padding.
def _slim422(l4):
    assert len(l4) == 4

    p0, p1 = l4[::2]
    if l4[0] == 0:  # TODO bad code
        p0 = l4[2] // 2
        if l4[2] == 1:
            p0 = 1
    if l4[1] == 0:  # TODO bad code
        p1 = l4[3] // 2
        if l4[3] == 1:
            p1 = 1
    return p0, p1


def _check_attr(attrs, map):
    for attr in attrs:
        if attr.name not in map:
            warnings.warn("Missing {} in parser's attr_map.".format(attr.name))


def unpack_weights(initializer, nodes):
    ret = {}
    for i in initializer:
        name = i.name
        dtype = i.data_type
        shape = list(i.dims)
        if dtype not in data_type_tab:
            warnings.warn("This data type {} is not supported yet.".format(dtype))
        fmt, size = data_type_tab[dtype]
        if len(i.raw_data) == 0:
            if dtype == 1:
                data_list = i.float_data
            elif dtype == 7:
                data_list = i.int64_data
            else:
                warnings.warn("No-raw-data type {} not supported yet.".format(dtype))
        else:
            data_list = struct.unpack('<' + fmt * (len(i.raw_data) // size), i.raw_data)
        t = torch.tensor(data_list, dtype=torch_type_tab[dtype])
        if len(shape) != 0:
            t = t.view(*shape)
        ret[name] = t
    for i in nodes:
        if i.op_type.lower() != "constant":
            continue
        dtype = i.attribute[0].t.data_type
        shape = list(i.attribute[0].t.dims)
        if dtype not in data_type_tab:
            warnings.warn("This data type {} is not supported yet.".format(dtype))
        fmt, size = data_type_tab[dtype]
        if len(i.attribute[0].t.raw_data) == 0:
            if dtype == 1:
                data_list = i.attribute[0].t.float_data
            elif dtype == 7:
                data_list = i.attribute[0].t.int64_data
            else:
                warnings.warn("No-raw-data type {} not supported yet.".format(dtype))
        else:
            data_list = struct.unpack('<' + fmt * (len(i.attribute[0].t.raw_data) // size), i.attribute[0].t.raw_data)
        t = torch.tensor(data_list, dtype=torch_type_tab[dtype])
        if len(shape) != 0:
            t = t.view(*shape)
        ret[i.output[0]] = t
    return ret


def rebuild_conv(node, weights):
    rebuild_conv.conv_attr_map = {
        "pads": "padding",
        "strides": "stride",
        "kernel_shape": "kernel_size",
        "group": "groups",
        "dilations": "dilation"
    }
    assert len(node.output) == 1
    with_bias = False
    if len(node.input) == 3:
        with_bias = True
        bias_name = node.input[2]
        bias = weights[bias_name]

    weight_name = node.input[1]
    weight = weights[weight_name]
    in_channels = weight.shape[1]
    out_channels = weight.shape[0]
    kwargs = {}

    if len(list(filter(lambda x: x.name == "auto_pad", node.attribute))) > 0:
        assert list(filter(lambda x: x.name == "auto_pad", node.attribute))[0].s == b"NOTSET", "auto_pad not supported"
    # Ignore spurious auto_pad attributes that are not doing anything ("NOTSET").
    attributes = filter(lambda x: x.name != "auto_pad", node.attribute)

    for att in attributes:
        kwargs[rebuild_conv.conv_attr_map[att.name]] = list(att.ints) if att.name != 'group' else att.i
    if 'padding' in kwargs:
        kwargs["padding"] = _slim422(kwargs["padding"])
    groups = 1 if 'groups' not in kwargs else kwargs['groups']
    in_channels *= groups
    conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=with_bias)
    conv.weight.data = weight
    if with_bias:
        conv.bias.data = bias
    return conv, node.input[:1], node.output


def rebuild_dropout(node, weights):
    ratio = node.attribute[0].f
    return nn.Dropout2d(p=ratio), node.input, node.output


def rebuild_batchnormalization(node, weights):
    rebuild_batchnormalization.bn_attr_map = {
        "epsilon": "eps",
        "momentum": "momentum"
    }
    assert len(node.input) == 5
    assert len(node.output) == 1
    weight = weights[node.input[1]]
    bias = weights[node.input[2]]
    running_mean = weights[node.input[3]]
    running_var = weights[node.input[4]]
    dim = weight.shape[0]
    kwargs = {}
    _check_attr(node.attribute, rebuild_batchnormalization.bn_attr_map)
    for att in node.attribute:
        if att.name in rebuild_batchnormalization.bn_attr_map:
            kwargs[rebuild_batchnormalization.bn_attr_map[att.name]] = att.f

    bn = nn.BatchNorm2d(num_features=dim)
    bn.weight.data = weight
    bn.bias.data = bias
    bn.running_mean.data = running_mean
    bn.running_var.data = running_var
    return bn, node.input[:1], node.output


def rebuild_relu(node, weights):
    return nn.ReLU(), node.input, node.output


def rebuild_max(node, weights):
    # Only support max operators that incode ReLUs.
    assert (node.input[1] in weights) or (node.input[0] in weights)
    const_input = next(filter(lambda x: x in weights, node.input))
    assert weights[const_input].sum() == 0
    nonconst_input = next(filter(lambda x: x not in weights, node.input))
    return nn.ReLU(), [nonconst_input], node.output


def rebuild_sigmoid(node, weights):
    return nn.Sigmoid(), node.input, node.output


def rebuild_maxpool(node, weights):
    rebuild_maxpool.mp_attr_map = {
        "pads": "padding",
        "strides": "stride",
        "kernel_shape": "kernel_size",
    }
    kwargs = {}
    for att in node.attribute:
        kwargs[rebuild_maxpool.mp_attr_map[att.name]] = list(att.ints)
    if 'padding' in kwargs:
        kwargs["padding"] = _slim422(kwargs["padding"])
    mp = nn.MaxPool2d(**kwargs)
    return mp, node.input, node.output


def rebuild_add(node, weights):
    is_residual = not (node.input[0] in weights or node.input[1] in weights)
    if is_residual:
        raise ResNetError("ResNet architectures not supported yet.")
    const = weights[node.input[0]] if node.input[0] in weights else weights[node.input[1]]
    return Add(const), node.input, node.output


def rebuild_sub(node, weights):
    const = weights[node.input[0]] if node.input[0] in weights else weights[node.input[1]]
    return Add(-const), node.input, node.output


def rebuild_div(node, weights):
    const = weights[node.input[0]] if node.input[0] in weights else weights[node.input[1]]
    return Mul(1/const), node.input, node.output


def rebuild_globalaveragepool(node, weights):
    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    return avg_pool, node.input, node.output


def rebuild_transpose(node, weights):
    return Transpose(tuple(node.attribute[0].ints)), node.input, node.output


def rebuild_flatten(node, weights):
    return Flatten(), node.input, node.output


def rebuild_gemm(node, weights):
    weight = weights[node.input[1]]
    bias = weights[node.input[2]]
    trans_b = list(filter(lambda x: x.name == "transB", node.attribute))[0].i
    if not trans_b:
        in_feats = weight.shape[0]
        out_feats = weight.shape[1]
        weight = weight.t()
    else:
        in_feats = weight.shape[1]
        out_feats = weight.shape[0]
    linear = nn.Linear(in_features=in_feats, out_features=out_feats)
    linear.weight.data = weight
    linear.bias.data = bias
    return linear, node.input[:1], node.output


def rebuild_matmul(node, weights):

    if node.input[1] in weights:
        weight = weights[node.input[1]]
        in_feats = weight.shape[0]
        out_feats = weight.shape[1]
        linear = nn.Linear(in_features=in_feats, out_features=out_feats, bias=True)
        linear.weight.data = weight.permute((1, 0))
    else:
        weight = weights[node.input[0]]
        in_feats = weight.shape[1]
        out_feats = weight.shape[0]
        linear = nn.Linear(in_features=in_feats, out_features=out_feats, bias=True)
        linear.weight.data = weight
    linear.weight.bias = torch.zeros_like(weight.select(1, 0))
    return linear, node.input[:1], node.output


def rebuild_concat(node, weights):
    dim = node.attribute[0].i

    def concat(*inputs):
        # for i in inputs:
        #     print(i.shape)
        ret = torch.cat(inputs, dim)
        # print(ret.shape)
        # exit()
        return ret
    return concat, node.input, node.output


def rebuild_pad(node, weights):
    mode = node.attribute[0].s
    pads = list(node.attribute[1].ints)
    value = node.attribute[2].f
    assert mode == b'constant'  # TODO constant only
    assert sum(pads[:4]) == 0  # TODO pad2d only
    pad = nn.ConstantPad2d(pads[4:], value)
    return pad, node.input, node.output


def rebuild_constant(node, weights):
    raw_data = node.attribute[0].t.raw_data
    data_type = node.attribute[0].t.data_type
    fmt, size = data_type_tab[data_type]
    data = struct.unpack('<' + fmt * (len(raw_data) // size), raw_data)
    if len(data) == 1:
        data = data[0]

    def constant():
        return torch.tensor(data)
    return constant, [], node.output


def rebuild_sum(node, weights):
    def sum(*inputs):
        ret = inputs[0]
        for i in inputs[1:]:
            ret += i
        return ret
    return sum, node.input, node.output


def rebuild_shape(node, weights):
    def shape(x):
        return torch.tensor(list(x.shape))
    return shape, node.input, node.output


def rebuild_gather(node, weights):
    axis = node.attribute[0].i

    def gather(x, idx):
        return torch.gather(x, axis, idx)
    return gather, node.input, node.output


def _nd_unsqueeze(x, dims):
    dims = sorted(dims)
    for d in dims:
        x = torch.unsqueeze(x, dim=d)
    return x


def rebuild_unsqueeze(node, weights):
    axes = node.attribute[0].ints

    def unsqueeze(x):
        return _nd_unsqueeze(x, axes)

    return unsqueeze, node.input, node.output


def rebuild_mul(node, weights):
    const = weights[node.input[0]] if node.input[0] in weights else weights[node.input[1]]
    return Mul(const), node.input, node.output


def rebuild_softmax(node, weights):
    def f_softmax(x):
        return x.softmax(dim=1, dtype=torch.double).float()
    return f_softmax, node.input, node.output


def rebuild_reshape(node, weights):
    shape = tuple(weights[node.input[1]].tolist())
    return Reshape(shape), node.input, node.output


def rebuild_averagepool(node, weights):
    rebuild_averagepool.avg_attr_map = {
        "pads": "padding",
        "strides": "stride",
        "kernel_shape": "kernel_size",
    }
    kwargs = {}

    for att in node.attribute:
        kwargs[rebuild_averagepool.avg_attr_map[att.name]] = list(att.ints)
    if 'padding' in kwargs:
        kwargs["padding"] = _slim422(kwargs["padding"])
    ap = nn.AvgPool2d(**kwargs)
    return ap, node.input, node.output


def rebuild_op(node, weights):
    op_type = node.op_type
    return globals()['rebuild_'+op_type.lower()](node, weights)


def construct_pytorch_nodes(graph, weights):
    # NOTE: assumes absence of residual connections, that the bias changes only according to output channels
    graph_dict = {}
    for cnode in graph.node:
        for out_name in cnode.output:
            graph_dict[out_name] = cnode
    ret = {}
    missing_biases = []
    for idx, single_node in enumerate(graph.node):
        matmul_wo_bias = None
        if single_node.op_type.lower() == "add":
            for cinput in single_node.input:
                if cinput in graph_dict and graph_dict[cinput].op_type.lower() in ["matmul", "conv"]:
                    matmul_wo_bias = cinput
                    break
            if matmul_wo_bias is not None:
                missing_biases.append((single_node, graph_dict[matmul_wo_bias].name + graph_dict[matmul_wo_bias].output[0]))
        if matmul_wo_bias is None:
            key = single_node.name + single_node.output[0]
            ret[key] = rebuild_op(single_node, weights)

    for missing_bias, matmul_name in missing_biases:
        if ret[matmul_name][0].bias is None:
            # Add add as bias of Conv2d operator -- PyTorch only supports 1D biases so the other dims are averaged
            bias = weights[missing_bias.input[1]]
            if bias.shape != (ret[matmul_name][0].weight.shape[0],):
                if len(bias.shape) == 4:
                    # average batch dimension (assumes it's a conversion artifact)
                    bias = bias.mean(dim=0)
                # NOTE: assumes the two trailing dimensions are conversion artifacts
                assert len(bias.shape) == 3
                bias = bias.mean(dim=(-1, -2))
            ret[matmul_name][0].bias = nn.Parameter(bias)
        else:
            # Add add as bias of Linear operator
            ret[matmul_name][0].bias.data = weights[missing_bias.input[1]]
        ret[matmul_name] = (ret[matmul_name][0], ret[matmul_name][1], missing_bias.output)

    return ret


def resolve_deps(name, deps, inter_tensors):
    if name in inter_tensors:
        return
    else:
        op, deps_names = deps[name]
        args = []
        for deps_name in deps_names:
            resolve_deps(deps_name, deps, inter_tensors)
            args.append(inter_tensors[deps_name])
        result = op(*args)
        inter_tensors[name] = result


class OnnxConverter(nn.Module):
    def __init__(self, onnx_model, input_name=None):
        super(OnnxConverter, self).__init__()
        self.deps = {}
        self.inter_tensors = dict()
        self.weights = unpack_weights(onnx_model.graph.initializer, onnx_model.graph.node)
        nodes = construct_pytorch_nodes(onnx_model.graph, self.weights)
        for idx, key in enumerate(nodes):
            node, inputs, outputs = nodes[key]
            if isinstance(node, nn.Module):
                # before adding the module, check whether we know the inputs -- pytorch conversion lacking otherwise
                for cin in inputs:
                    # input_unknown means that it's not the overall input, it's not a torch module,
                    # it's not in the initializers
                    input_unknown = cin not in [cinp.name for cinp in onnx_model.graph.input] \
                                    and (not cin in self.weights) and (not isinstance(self.deps[cin][0], nn.Module))
                    if input_unknown:
                        print("Missing input to pytorch module while converting onnx -> pytorch")
                self.add_module(str(idx), node)
            for output_name in outputs:
                self.deps[output_name] = (node, inputs)

        self.input_name = onnx_model.graph.input[0].name
        self.output_name = onnx_model.graph.output[0].name
        if input_name is not None:
            self.input_name = input_name

    def forward(self, input):
        self.inter_tensors = self.weights.copy()
        self.inter_tensors[self.input_name] = input
        resolve_deps(self.output_name, self.deps, self.inter_tensors)
        return self.inter_tensors[self.output_name]
