import torch.nn as nn
import torch

from utils.read_nnet import ReLU, Linear
import utils


def _pos(x):
    return torch.clamp(x, 0, torch.inf)


def _neg(x):
    return torch.clamp(x, -torch.inf, 0)


def _evaluate(eq_lower, eq_upper,
              input_lower, input_upper):
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    o_l_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_u_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    o_l_l = o_l_l.sum(0) + eq_lower[-1]
    o_u_u = o_u_u.sum(0) + eq_upper[-1]
    return o_l_l, o_u_u


def _evaluate_inner_bounds(eq_lower, eq_upper,
                           input_lower, input_upper):
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    o_l_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    o_u_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_l_u = o_l_u.sum(0) + eq_lower[-1]
    o_u_l = o_u_l.sum(0) + eq_upper[-1]
    return o_l_u, o_u_l


def relu_transform(eq_lower, eq_upper,
                   input_lower, input_upper,
                   input_bounds=None):
    # evaluate output ranges
    output_eq_lower = eq_lower.clone()
    output_eq_upper = eq_upper.clone()
    grad_mask = torch.zeros(input_lower.size(0))

    if input_bounds is not None:
        o_l_l, o_u_u = input_bounds
    else:
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper,
                                 input_lower, input_upper)
    o_l_u, o_u_l = _evaluate_inner_bounds(eq_lower, eq_upper,
                                          input_lower, input_upper)

    for i, (ll, lu, uu, ul) in enumerate(zip(o_l_l, o_l_u, o_u_u, o_u_l)):
        if uu <= 0:
            grad_mask[i] = 0
            output_eq_lower[:, i] = 0
            output_eq_upper[:, i] = 0
        elif ll >= 0:
            grad_mask[i] = 2
        else:
            grad_mask[i] = 1
            if ul < 0:
                output_eq_upper[:, i] = eq_upper[:, i] * uu / (uu - ul)
                output_eq_upper[-1, i] -= uu * ul / (uu - ul)
            if lu < 0:
                output_eq_lower[:, i] = 0
            else:
                output_eq_lower[:, i] = eq_lower[:, i] * lu / (lu - ll)
    return (output_eq_lower, output_eq_upper), grad_mask


def backward_relu_transform(grad_lower, grad_upper, grad_mask):
    # always negative
    zero_inds = grad_mask.eq(0)
    grad_lower[zero_inds] = 0
    grad_upper[zero_inds] = 0

    # lower < 0 and upper > 0
    one_inds = grad_mask.eq(1)
    grad_upper[one_inds] = grad_upper[one_inds].clamp(0, torch.inf)
    grad_lower[one_inds] = grad_lower[one_inds].clamp(-torch.inf, 0)
    return grad_lower, grad_upper


def linear_transform(layer, eq_lower, eq_upper):
    pos_weight, neg_weight = _pos(layer.weight), _neg(layer.weight)
    out_eq_upper = eq_upper @ pos_weight.T + eq_lower @ neg_weight.T
    out_eq_lower = eq_lower @ pos_weight.T + eq_upper @ neg_weight.T
    if layer.bias is not None:
        out_eq_lower[-1] += layer.bias
        out_eq_upper[-1] += layer.bias
    return out_eq_lower, out_eq_upper


def backward_linear_transform(layer, grad_lower, grad_upper):
    pos_weight, neg_weight = _pos(layer.weight), _neg(layer.weight)
    input_grad_lower = grad_lower @ pos_weight + grad_upper @ neg_weight
    input_grad_upper = grad_upper @ pos_weight + grad_lower @ neg_weight
    return input_grad_lower, input_grad_upper


def flatten_transform(eq_lower, eq_upper):
    raise NotImplementedError


@torch.no_grad()
def forward(net, lower, upper, return_grad_mask=False):
    input_features = lower.numel()

    # initialize lower and upper equation
    eq_lower = torch.concat([torch.eye(input_features), torch.zeros(1, input_features)], dim=0)
    eq_upper = eq_lower.clone()

    o_l_l = lower.clone()
    o_u_u = upper.clone()
    grad_mask = {}

    for layer_id, layer in enumerate(net.layers):
        if isinstance(layer, nn.Linear):
            eq_lower, eq_upper = linear_transform(layer, eq_lower, eq_upper)
        elif isinstance(layer, nn.ReLU):
            (eq_lower, eq_upper), grad_mask_l = relu_transform(eq_lower, eq_upper,
                                                               lower, upper,
                                                               input_bounds=(o_l_l, o_u_u))
            grad_mask[layer_id] = grad_mask_l
        else:
            raise NotImplementedError
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper, lower, upper)

    if return_grad_mask:
        return (o_l_l, o_u_u), grad_mask
    return o_l_l, o_u_u


@torch.no_grad()
def backward(net, output_grad, grad_mask):
    grad_lower = grad_upper = output_grad

    for layer_id in reversed(range(len(net.layers))):
        layer = net.layers[layer_id]
        if isinstance(layer, nn.Linear):
            grad_lower, grad_upper = backward_linear_transform(layer, grad_lower, grad_upper)
        elif isinstance(layer, nn.ReLU):
            grad_lower, grad_upper = backward_relu_transform(grad_lower, grad_upper, grad_mask[layer_id])
        else:
            raise NotImplementedError

    return grad_lower, grad_upper
