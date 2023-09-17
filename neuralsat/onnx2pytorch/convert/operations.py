from collections import defaultdict
from functools import partial

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from onnx import numpy_helper
from torch.nn.modules.linear import Identity

from onnx2pytorch.convert.attribute import extract_attributes
from onnx2pytorch.convert.layer import (
    convert_layer,
    convert_linear_layer,
    convert_batch_norm_layer,
    convert_instance_norm_layer,
    convert_lstm_layer,
)
from onnx2pytorch.operations import *
from onnx2pytorch.operations.base import OperatorWrapper
from onnx2pytorch.operations import Resize, Upsample
from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
    value_wrapper,
)


def get_buffer_name(param_name):
    """
    Convert name of initializer to valid name for nn.Module attribute.
    """
    return "_initializer_{}".format(param_name.replace(".", "_"))


def get_init_parameter(modules, item, default):
    """
    Look in modules for the item, and if not found return default.

    Parameters
    ----------
    modules: list of nn.Modules
        Modules whose attributes to search.
    item: str
        Name of initializer in ONNX model.
    default: torch.Tensor
        Tensor to return if item not found.
    """
    item_name = get_buffer_name(item)
    for mod in modules:
        if hasattr(mod, item_name):
            return getattr(mod, item_name)
    return default


def convert_operations(onnx_graph, opset_version, batch_dim=0, enable_pruning=True, quirks=None):
    """
    Convert onnx model operations. Yields onnx's operator_id, operator_name and
    converted pytorch operator.

    Parameters
    ----------
    onnx_graph: onnx.GraphProto
        Loaded onnx model's GraphProto.
    opset_version: int
        ONNX model's opset version.
    batch_dim: int
        Usually 0 for computer vision models and 1 for NLP models.
    enable_pruning: bool
        Track kept/pruned indices between different calls to forward pass.

    Returns
    -------
    iterator: (op_id, op_name, op)
    """
    weights = {tensor.name: tensor for tensor in onnx_graph.initializer}
    
    onnx_inputs = [node.name for node in onnx_graph.input]
    onnx_initializers = [node.name for node in onnx_graph.initializer]
    inputs = list(set(onnx_inputs) - set(onnx_initializers))
    inputs = [node for node in onnx_graph.input if node.name in inputs]
    
    for i, node in enumerate(onnx_graph.node):
        # extract only useful inputs
        is_nhwc = False
        is_last_removed = False
        params = [weights[par_name] for par_name in node.input if par_name in weights]

        if node.op_type == "Add":
            op = Add(feature_dim=batch_dim + 1)  # 0 for CV models and 1 for NLP
        elif node.op_type == "And":
            op = OperatorWrapper(torch.logical_and)
        elif node.op_type == "AveragePool":
            op = convert_layer(node, "AvgPool")
        elif node.op_type == "BatchNormalization":
            op = convert_batch_norm_layer(node, params=params)
        elif node.op_type == "Cast":
            op = Cast(**extract_attributes(node))
        elif node.op_type == "Ceil":
            op = OperatorWrapper(torch.ceil)
        elif node.op_type == "Clip":
            op = Clip(**extract_attributes(node))
        elif node.op_type == "Concat":
            op = partial(torch.cat, **extract_attributes(node))
        elif node.op_type == "Constant":
            constant = extract_attributes(node)['constant']
            next_node = onnx_graph.node[i + 1]
            if len(constant) == 2 and (-1 in constant) and next_node.op_type == 'Reshape' \
                    and len(node.input) == 0 and len(node.output) == 1:
                op = Flatten()
                node.input.extend([n_i for n_i in next_node.input if n_i != node.output[0]])
                node.output.pop()
                node.output.extend(next_node.output)
                onnx_graph.node.pop(i + 1)  # remove next node
            else:
                op = Constant(**extract_attributes(node))
        elif node.op_type == "ConstantOfShape":
            op = ConstantOfShape(**extract_attributes(node))
        elif node.op_type == "Conv":
            op = convert_layer(node, "Conv", params)
            if (i < len(onnx_graph.node) + 1) and (onnx_graph.node[i + 1].op_type == "BatchNormalization") and quirks.get(node.op_type, {}).get('merge_batch_norm', False):
                next_node = onnx_graph.node[i+1]
                next_params = [weights[par_name] for par_name in next_node.input if par_name in weights]
                next_layer = convert_batch_norm_layer(next_node, params=next_params)
                
                merge_bn_weight = (next_layer.bnu.weight / (next_layer.bnu.eps + next_layer.bnu.running_var).sqrt()).diag()
                merge_bn_bias = next_layer.bnu.bias - next_layer.bnu.weight * next_layer.bnu.running_mean / (next_layer.bnu.eps + next_layer.bnu.running_var).sqrt()
                
                merge_weight = torch.matmul(merge_bn_weight, op.weight.flatten(1)).view(op.weight.shape)
                if op.bias is None:
                    op.bias = nn.Parameter(torch.zeros(op.weight.shape[0]).float())

                merge_bias = torch.matmul(merge_bn_weight, op.bias) + merge_bn_bias

                with torch.no_grad():
                    op.weight.copy_(merge_weight)
                    op.bias.copy_(merge_bias)
                    
                node.output.pop()
                node.output.extend(next_node.output)
                onnx_graph.node.pop(i + 1)  # remove next node

                
        elif node.op_type == "ConvTranspose":
            op = convert_layer(node, "ConvTranspose", params)
        elif node.op_type == "Div":
            op = Div()
        elif node.op_type == "Elu":
            op = nn.ELU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Equal":
            op = OperatorWrapper(torch.eq)
        elif node.op_type == "Erf":
            op = OperatorWrapper(torch.erf)
        elif node.op_type == "Exp":
            op = OperatorWrapper(torch.exp)
        elif node.op_type == "Expand":
            op = Expand()
        elif node.op_type == "Flatten":
            op = Flatten(**extract_attributes(node))
            op.feature_dim = batch_dim + 1  # Necessary for transformers
        elif node.op_type == "Floor":
            op = OperatorWrapper(torch.floor)
        elif node.op_type == "Gather":
            op = Gather(**extract_attributes(node))
        elif node.op_type == "GatherND":
            op = GatherND(**extract_attributes(node))
        elif node.op_type == "Gemm":
            op = convert_linear_layer(node, params)
        elif node.op_type == "GlobalAveragePool":
            op = GlobalAveragePool()
        elif node.op_type == "Greater":
            op = OperatorWrapper(torch.greater)
        elif node.op_type == "Identity":
            op = nn.Identity()
        elif node.op_type == "InstanceNormalization":
            op = convert_instance_norm_layer(node, params=params)
        elif node.op_type == "LeakyRelu":
            op = nn.LeakyReLU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Less":
            op = OperatorWrapper(torch.less)
        elif node.op_type == "Log":
            op = OperatorWrapper(torch.log)
        elif node.op_type == "Loop":
            op = Loop(
                opset_version=opset_version,
                batch_dim=batch_dim,
                **extract_attributes(node),
            )
        elif node.op_type == "LSTM":
            op = convert_lstm_layer(node, weights)
        elif node.op_type == "MatMul":
            if params:
                weight = torch.tensor(numpy_helper.to_array(params[0]))
                # print(weight.ndim)
                # print(node.input)
                # print(list(weights.keys()))
                if node.input[0] in weights:
                    # op = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                    # op.weight.data = weight
                    if weight.ndim == 2:
                        op = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                        op.weight.data = weight
                    else:
                        op = MatMul()
                else:
                    op = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
                    op.weight.data = weight.t()
                    
                # check if next node Add to add bias
                if i < len(onnx_graph.node) - 1:
                    next_node = onnx_graph.node[i + 1]
                    # print(next_node.op_type)
                    # print('weights:', list(weights.keys()))
                    # print('next input[0]', next_node.input[0])
                    # print('next input[1]', next_node.input[1])
                    # print('current input[0]:', node.input[0])
                    # print('current input[1]:', node.input[1])
                    # print('current output[0]:', node.output[0])
                    if len(next_node.input) == 2 and next_node.op_type == "Add":
                        use_bias = False
                        if ((next_node.input[0] in weights) or (next_node.input[1] in weights)) and (node.output[0] in next_node.input):
                            if   (next_node.input[1] in weights):
                                bias = torch.tensor(numpy_helper.to_array(weights[next_node.input[1]]))
                                use_bias = True
                            elif (next_node.input[0] in weights):
                                bias = torch.tensor(numpy_helper.to_array(weights[next_node.input[0]]))
                                use_bias = True
                        if use_bias:
                            op.bias = nn.Parameter(bias)
                            node.output.pop()
                            node.output.extend(next_node.output)
                            onnx_graph.node.pop(i + 1)  # remove next node
                #         print('bias:', use_bias)
                # print(i, op, len(onnx_graph.node))
                # print()
            else:
                op = MatMul()
        elif node.op_type == "Max":
            op = OperatorWrapper(torch.max)
        elif node.op_type == "MaxPool":
            op = convert_layer(node, "MaxPool")
        elif node.op_type == "Min":
            op = OperatorWrapper(torch.min)
        elif node.op_type == "Mul":
            op = OperatorWrapper(torch.mul)
        elif node.op_type == "NonMaxSuppression":
            op = NonMaxSuppression(**extract_attributes(node))
        elif node.op_type == "Not":
            op = OperatorWrapper(torch.logical_not)
        elif node.op_type == "OneHot":
            op = OneHot(**extract_attributes(node))
        elif node.op_type == "Or":
            op = OperatorWrapper(torch.logical_or)
        elif node.op_type == "Pad":
            op = Pad(**extract_attributes(node))
        elif node.op_type == "Pow":
            op = OperatorWrapper(torch.pow)
        elif node.op_type == "PRelu":
            op = PRelu()
        elif node.op_type == "Range":
            op = Range()
        elif node.op_type == "Reciprocal":
            op = OperatorWrapper(torch.reciprocal)
        elif node.op_type == "ReduceMax":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.max, **kwargs)
        elif node.op_type == "ReduceMean":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.mean, **kwargs)
        elif node.op_type == "ReduceMin":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.min, **kwargs)
        elif node.op_type == "ReduceProd":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.prod, **kwargs)
        elif node.op_type == "ReduceSum":
            op = ReduceSum(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Relu":
            op = nn.ReLU(inplace=True)
        elif node.op_type == "Reshape":
            shape = list(
                filter(lambda x: x.name == node.input[1], onnx_graph.initializer)
            )
            shape = np.copy(numpy_helper.to_array(shape[0])) if shape else None
            if all(shape == (1, -1)):
                op = Flatten()
                for n_idx, n_name in enumerate(node.input):
                    if n_name in onnx_initializers:
                        node.input.pop(n_idx)
            else:
                op = Reshape(enable_pruning, shape, quirks=quirks.get("Reshape"))
            # exit()
        elif node.op_type == "Resize":
            op = Resize(**extract_attributes(node))
        elif node.op_type == "Scatter":
            op = Scatter(**extract_attributes(node))
        elif node.op_type == "ScatterElements":
            op = ScatterElements(**extract_attributes(node))
        elif node.op_type == "ScatterND":
            op = ScatterND()
        elif node.op_type == "Shape":
            op = Shape()
        elif node.op_type == "Sigmoid":
            op = nn.Sigmoid()
        elif node.op_type == "Slice":
            op = Slice(**extract_attributes(node))
        elif node.op_type == "Softmax":
            if node == onnx_graph.node[-1] and quirks.get(node.op_type, {}).get('skip_last_layer', False):
                op = nn.Identity()
                is_last_removed = True
            else:
                kwargs = dict(dim=-1)
                kwargs.update(extract_attributes(node))
                op = nn.Softmax(**kwargs)
        elif node.op_type == "Softplus":
            op = nn.Softplus(beta=1)
        elif node.op_type == "Softsign":
            op = nn.Softsign()
        elif node.op_type == "Split":
            kwargs = extract_attributes(node)
            # if the split_size_or_sections is not in node attributes,
            # the number_of_splits becomes the number of node outputs
            if "split_size_or_sections" not in kwargs:
                kwargs["number_of_splits"] = len(node.output)
            op = Split(enable_pruning, **kwargs)
        elif node.op_type == "Sqrt":
            op = OperatorWrapper(torch.sqrt)
        elif node.op_type == "Squeeze":
            op = Squeeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Sub":
            op = OperatorWrapper(torch.sub)
        elif node.op_type == "Tanh":
            op = OperatorWrapper(torch.tanh)
        elif node.op_type == "ThresholdedRelu":
            op = ThresholdedRelu(**extract_attributes(node))
        elif node.op_type == "Tile":
            op = Tile()
        elif node.op_type == "TopK":
            op = TopK()
        elif node.op_type == "Transpose":
            # op = Transpose(**extract_attributes(node))
            if i == 0 and extract_attributes(node)['dims'] == (0, 3, 1, 2): 
                # nhwc
                next_node = onnx_graph.node[i + 1]
                assert next_node.op_type == 'Conv'
                for i_, inp in enumerate(next_node.input):
                    if inp == node.output[0]:
                        next_node.input[i_] = node.input[0]
                        break
                op = None
                is_nhwc = True
            if i == 0 and extract_attributes(node)['dims'] == (0, 2, 3, 1) \
                    and inputs[0].type.tensor_type.shape.dim[1].dim_value == 1 and quirks.get(node.op_type, {}).get('remove_gdvb_transpose', False):
                next_node = onnx_graph.node[i + 1]
                for i_, inp in enumerate(next_node.input):
                    if inp == node.output[0]:
                        next_node.input[i_] = node.input[0]
                        break
                op = None
                 
            elif quirks.get(node.op_type, {}).get('remove_spare_permute', False):
                dims = extract_attributes(node)['dims']
                if dims == (0, 2, 3, 1):
                    remove_transpose = False
                    for ii in range(i, len(onnx_graph.node)):
                        node_tmp = onnx_graph.node[ii]
                        # print(node_tmp.op_type)
                        if node_tmp.op_type == node.op_type and extract_attributes(node_tmp)['dims'] == (0, 3, 1, 2):
                            remove_transpose = True
                            break
                        
                        if node_tmp.op_type in ['Reshape', 'Conv', 'ConvTranspose']:
                            break
                    
                    if remove_transpose:
                        op = nn.Identity()
                    else:
                        op = Transpose(**{**extract_attributes(node), **{"quirks": quirks.get("Transpose", {})}})
                        
                elif dims == (0, 3, 1, 2):
                    if remove_transpose:
                        op = nn.Identity()
                        remove_transpose = False
                    else:
                        op = Transpose(**{**extract_attributes(node), **{"quirks": quirks.get("Transpose", {})}})
                else:
                    raise NotImplementedError()
                
                
                # exit()
            
            else:
                # https://github.com/KaidiXu/onnx2pytorch/commit/b96e9f9591a53367cd302301fcd0d6695f924f21
                op = Transpose(**{**extract_attributes(node), **{"quirks": quirks.get("Transpose", {})}})
        elif node.op_type == "Unsqueeze":
            op = Unsqueeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Upsample":
            op = Upsample(**extract_attributes(node))
        elif node.op_type == "Where":
            op = Where()
        elif node.op_type == "Dropout":
            op = nn.Identity()
        elif node.op_type == "ArgMax":
            op = Argmax(node.attribute[0].i)
        else:
            op = getattr(torch, node.op_type.lower(), None)
            if op is None:
                raise NotImplementedError(
                    "Conversion not implemented for op_type={}.".format(node.op_type)
                )
            else:
                print(
                    "Automatic inference of operator: {}".format(node.op_type.lower())
                )

        op_name = "{}_{}".format(node.op_type, node.output[0])
        op_id = node.output[0]
        yield op_id, op_name, op, is_nhwc, is_last_removed
