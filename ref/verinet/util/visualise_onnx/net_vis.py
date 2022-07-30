"""
A simple script for visualising onnx networks.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import netron
import torch
import onnx

from verinet.sip_torch.rsip import RSIP
from verinet.sip_torch.operations.piecewise_linear import Relu
from verinet.sip_torch.operations.s_shaped import Sigmoid, Tanh
from verinet.parsers.onnx_parser import ONNXParser


def count_nodes(model, input_shape=(3, 32, 32)):

    rsip = RSIP(model, torch.LongTensor(input_shape))
    count = 0

    for node in rsip.nodes:
        if isinstance(node.op, Relu) or isinstance(node.op, Sigmoid) or isinstance(node.op, Tanh):
            count += node.in_size

    return count


if __name__ == '__main__':

    onnx_path = "../../resources/benchmarks/nn4sys/nets/normal_100.onnx"
    in_shape = (1,)

    onnx_parser = ONNXParser(onnx_path, transpose_fc_weights=False)
    torch_model = onnx_parser.to_pytorch()
    print(f"Number of nodes: {count_nodes(torch_model, input_shape=in_shape)}")
    for n in torch_model.nodes:
        print(n)

    # torch_model.save(torch.zeros((1, *in_shape)), "tmp.onnx")
    netron.start(onnx_path)
    # netron.start("tmp.onnx")

