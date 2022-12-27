import torch.nn as nn
import numpy as np
import torch
import math
import onnx2pytorch
import onnx
import gzip


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


def load_onnx(path):
    if path.endswith('.gz'):
        onnx_model = onnx.load(gzip.GzipFile(path))
    else:
        onnx_model = onnx.load(path)
    return onnx_model

def load_model_onnx(path, is_channel_last=False, force_convert=False):
    # pip install onnx2pytorch
    onnx_model = load_onnx(path)

    onnx_input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
    input_shape = tuple(d.dim_value for d in onnx_input_dims) if len(onnx_input_dims) > 1 else (1, onnx_input_dims[0].dim_value)
    output_shape = tuple(d.dim_value for d in onnx_output_dims) if len(onnx_output_dims) > 1 else (1, onnx_output_dims[0].dim_value)
    # input_shape = tuple(input_shape)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)

    if force_convert or len(input_shape) <= 2:
        new_modules = []
        modules = list(pytorch_model.modules())[1:]
        for mi, m in enumerate(modules):
            if isinstance(m, torch.nn.Linear):
                new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
                new_m.weight.data.copy_(m.weight.data)
                new_m.bias.data.copy_(m.bias)
                new_modules.append(new_m)
            elif isinstance(m, torch.nn.ReLU):
                new_modules.append(torch.nn.ReLU())
            elif isinstance(m, onnx2pytorch.operations.flatten.Flatten):
                new_modules.append(torch.nn.Flatten())
            else:
                raise NotImplementedError

        seq_model = nn.Sequential(*new_modules)
        return seq_model, input_shape, output_shape, False


    # # Check model input shape.
    # is_channel_last = False
    # if onnx_shape != input_shape:
    #     # Change channel location.
    #     onnx_shape = onnx_shape[2:] + onnx_shape[:2]
    #     if onnx_shape == input_shape:
    #         is_channel_last = True
    #     else:
    #         print(f"Unexpected input shape in onnx: {onnx_shape}, given {input_shape}")

    # Fixup converted ONNX model. For ResNet we directly return; for other models, we convert them to a Sequential model.
    # We also need to handle NCHW and NHWC formats here.
    conv_c, conv_h, conv_w = input_shape
    modules = list(pytorch_model.modules())[1:]
    new_modules = []
    need_permute = False
    for mi, m in enumerate(modules):
        if isinstance(m, onnx2pytorch.operations.add.Add):
            # ResNet model. No need to convert to sequential.
            return pytorch_model, is_channel_last
        if isinstance(m, torch.nn.Conv2d):
            # Infer the output size of conv.
            conv_h, conv_w = conv_output_shape((conv_h, conv_w), m.kernel_size, m.stride, m.padding)
            conv_c = m.weight.size(0)
        if isinstance(m, onnx2pytorch.operations.reshape.Reshape):
            # Replace reshape with flatten.
            new_modules.append(nn.Flatten())
            # May need to permute the next linear layer if the model was in NHWC format.
            need_permute = True and is_channel_last
        elif isinstance(m, torch.nn.Linear) and need_permute:
            # The original model is in NHWC format and we now have NCHW format, so the dense layer's weight must be adjusted.
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.view(m.weight.size(0), conv_h, conv_w, conv_c).permute(0, 3, 1, 2).contiguous().view(m.weight.size(0), -1))
            new_m.bias.data.copy_(m.bias)
            need_permute = False
            new_modules.append(new_m)
        elif isinstance(m, torch.nn.ReLU) and mi == (len(modules)-1):
            # not add relu if last layer is relu
            pass
        else:
            new_modules.append(m)

    seq_model = nn.Sequential(*new_modules)

    return seq_model, input_shape, output_shape, is_channel_last


class ONNXParser:

    def __init__(self, filename, force_convert=False, is_channel_last=False):

        model, input_shape, output_shape, is_channel_last = load_model_onnx(filename, force_convert=force_convert, is_channel_last=is_channel_last)
        model = model.eval()

        # TODO: create input to check flag `is_channel_last`
 
        if is_channel_last:
            input_shape = input_shape[:1] + input_shape[2:] + input_shape[1:2]
            print(f'Notice: this ONNX file has NHWC order. We assume the X in vnnlib is also flattend in in NHWC order {input_shape}')

        self.pytorch_model = PyTorchModelWrapper(model)
        self.pytorch_model.n_input = np.prod(input_shape)
        self.pytorch_model.n_output = np.prod(output_shape)
        self.pytorch_model.input_shape = input_shape
