import torch.nn as nn
import numpy as np
import torch
import math
import onnx
import gzip

import onnx2pytorch

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


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride)
    if type(pad) is not tuple:
        pad = (pad, pad)
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    return h, w


def load_onnx(path):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model

def load_model_onnx(path):
    onnx_model = load_onnx(path)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    input_shape = tuple(d.dim_value for d in onnx_input_dims) if len(onnx_input_dims) > 1 else (1, onnx_input_dims[0].dim_value)
    output_shape = tuple(d.dim_value for d in onnx_output_dims) if len(onnx_output_dims) > 1 else (1, onnx_output_dims[0].dim_value)
    # input_shape = tuple(input_shape)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    # if force_convert or len(input_shape) <= 2:
    #     new_modules = []
    #     modules = list(pytorch_model.modules())[1:]
    #     for mi, m in enumerate(modules):
    #         if isinstance(m, torch.nn.Linear):
    #             new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
    #             new_m.weight.data.copy_(m.weight.data)
    #             new_m.bias.data.copy_(m.bias)
    #             new_modules.append(new_m)
    #         elif isinstance(m, torch.nn.ReLU):
    #             new_modules.append(torch.nn.ReLU())
    #         elif isinstance(m, operations.flatten.Flatten):
    #             new_modules.append(torch.nn.Flatten())
    #         else:
    #             raise NotImplementedError

    #     seq_model = nn.Sequential(*new_modules)
    #     return seq_model, input_shape, output_shape, False

    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    for mi, m in enumerate(modules):
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
        elif isinstance(m, onnx2pytorch.operations.constant.Constant):
            pass
        else:
            # print('add', m, type(m))
            new_modules.append(m)

    seq_model = nn.Sequential(*new_modules)
    seq_model.eval()
    return seq_model, input_shape, output_shape


class ONNXParser:

    def __init__(self, filename):

        model, input_shape, output_shape = load_model_onnx(filename)

        self.pytorch_model = PyTorchModelWrapper(model)
        self.pytorch_model.n_input = np.prod(input_shape)
        self.pytorch_model.n_output = np.prod(output_shape)
        self.pytorch_model.input_shape = input_shape
