from pprint import pprint
import sortedcontainers
import torch.nn as nn
import numpy as np
import torch
import math

from utils.read_nnet import NetworkNNET
from utils.read_onnx import ONNXParser
import settings

class DNNParser:

    def parse(filename):
        if filename.lower().endswith('.nnet'):
            return DNNParser.parse_nnet(filename)
        if filename.lower().endswith('.onnx'):
            return DNNParser.parse_onnx(filename)

        print(f'Error extention: {filename}')
        raise NotImplementedError

    def parse_nnet(filename):
        model = NetworkNNET(filename)

        layers_mapping = {}
        idx = 1
        lid = 0
        for layer in model.layers[1:]: # discard input layer
            if isinstance(layer, nn.Linear):
                layers_mapping[lid] = sortedcontainers.SortedList()
                for i in range(layer.weight.shape[1]): # #nodes in layer
                    layers_mapping[lid].add(idx)
                    idx += 1
                lid += 1

        model.layers_mapping = layers_mapping
        return model

    def parse_onnx(filename):

        model = ONNXParser(filename)
        pytorch_model = model.pytorch_model

        # test to make transpose weights:
        x = torch.randn(pytorch_model.input_shape, dtype=settings.DTYPE)
        try:
            pytorch_model(x)
        except RuntimeError:
            model = ONNXParser(filename, transpose_weight=True)
            pytorch_model = model.pytorch_model

        relus = model.extract_ordered_relu_shapes()
        shapes = [0] + [math.prod(s) for s in relus]
        shapes = np.cumsum(shapes)
        res = list(zip(shapes, shapes[1:]))

        layers_mapping = {
            i: sortedcontainers.SortedList(
                range(s[0]+1, s[1]+1)
            ) for i, s in enumerate(res)
        }

        pytorch_model.layers_mapping = layers_mapping
        return pytorch_model
