from pprint import pprint
import sortedcontainers
import torch.nn as nn
import onnx2pytorch
import numpy as np
import torch
import math

from utils.read_onnx import ONNXParser
from utils.read_nnet import NetworkNNET
import settings

class DNNParser:

    def parse(filename, dataset, device=torch.device('cpu')):
        if filename.lower().endswith('.nnet'):
            assert dataset in ['acasxu', 'test']
            net = DNNParser.parse_nnet(filename)
        elif filename.lower().endswith('.onnx'):
            net = DNNParser.parse_onnx(filename, dataset)
        else:
            print(f'Error extention: {filename}')
            raise NotImplementedError
        net.device = device
        net.dataset = dataset
        return net.to(device)

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

    def parse_onnx(filename, dataset):

        if dataset in ['test']:
            raise
        else:
            model = ONNXParser(filename, dataset)
            pytorch_model = model.pytorch_model
            x = torch.randn(pytorch_model.input_shape, dtype=settings.DTYPE)

            count = 1
            layers_mapping = {}
            idx = 0
            
            for layer in pytorch_model.layers:
                if isinstance(layer, nn.ReLU):
                    layers_mapping[idx] = sortedcontainers.SortedList(range(count, count+x.numel()))
                    idx += 1
                    count += x.numel()
                x = layer(x)

        pytorch_model.layers_mapping = layers_mapping
        return pytorch_model
