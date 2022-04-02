import torch.nn as nn
import torch

from utils.read_nnet import ReLU, Linear
import utils


def _pos(x):
    return torch.clamp(x, 0, torch.inf)


def _neg(x):
    return torch.clamp(x, -torch.inf, 0)


def _evaluate(eq_lower, eq_upper, input_lower, input_upper):
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    o_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    return o_l.sum(0) + eq_lower[-1], o_u.sum(0) + eq_upper[-1]


def relu_transform(eq_lower, eq_upper, input_lower, input_upper, output_lower=None, output_upper=None):
    # evaluate output ranges
    output_eq_lower = eq_lower.clone()
    output_eq_upper = eq_upper.clone()
    if output_lower is None or output_upper is None:
        output_lower, output_upper = _evaluate(eq_lower, eq_upper, input_lower, input_upper)

    for i, (lb, ub) in enumerate(zip(output_lower, output_upper)):
        if ub <= 0:
            output_eq_lower[:, i] = 0
        elif lb >= 0:
            pass
        else:
            output_eq_lower[:, i] = 0
            output_eq_upper[:-1, i] = 0
            output_eq_upper[-1, i] = ub
    return output_eq_lower, output_eq_upper


def linear_transform(layer, eq_lower, eq_upper):
    weight = layer.weight
    pos_weight, neg_weight = _pos(weight), _neg(weight)
    out_eq_upper = eq_upper @ pos_weight.T + eq_lower @ neg_weight.T
    out_eq_lower = eq_lower @ pos_weight.T + eq_upper @ neg_weight.T
    if bias := layer.bias is not None:
        out_eq_lower[-1] += bias
        out_eq_upper[-1] += bias
    return out_eq_lower, out_eq_upper


def flatten_transform(eq_lower, eq_upper):
    # TODO: ?
    return eq_lower, eq_upper



def forward(net, lower, upper):
    input_features = lower.numel()

    # initialize lower and upper equation
    eq_lower = torch.concat([torch.eye(input_features), torch.zeros(1, input_features)], dim=0)
    eq_upper = eq_lower.clone()

    output_lower = lower.clone()
    output_upper = upper.clone()

    for layer in net.layers:
        if isinstance(layer, nn.modules.linear.Linear):
            eq_lower, eq_upper = linear_transform(layer, eq_lower, eq_upper)
        elif isinstance(layer, nn.modules.activation.ReLU):
            eq_lower, eq_upper = relu_transform(eq_lower, eq_upper, lower, upper, output_lower, output_upper)
        else:
            raise NotImplementedError
        output_lower, output_upper = _evaluate(eq_lower, eq_upper, lower, upper)

    return output_lower, output_upper


def forward_nnet(net, lower, upper):
    input_features = lower.numel()

    # initialize lower and upper equation
    eq_lower = torch.concat([torch.eye(input_features), torch.zeros(1, input_features)], dim=0)
    eq_upper = eq_lower.clone()

    output_lower = lower.clone()
    output_upper = upper.clone()

    for layer in net.layers:
        if isinstance(layer, Linear):
            eq_lower, eq_upper = linear_transform(layer, eq_lower, eq_upper)
        elif isinstance(layer, ReLU):
            eq_lower, eq_upper = relu_transform(eq_lower, eq_upper, lower, upper, output_lower, output_upper)
        else:
            raise NotImplementedError
        output_lower, output_upper = _evaluate(eq_lower, eq_upper, lower, upper)


    return output_lower, output_upper
