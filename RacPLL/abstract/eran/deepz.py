import torch.nn as nn
import torch
import os

import settings

def relu_transform(center, error):

    center_orig_shape = center.shape
    error_orig_shape = error.shape

    center, error = flatten_transform(center, error)

    # bounds
    error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
    ub = center + error_apt
    lb = center - error_apt

    # processing index
    case_idx = torch.where(((ub > 0) & (lb < 0)))

    # new error
    n_error, d = error.shape
    n_new_error = case_idx[0].shape[0]
    new_error = torch.zeros((n_error + n_new_error, d), dtype=settings.DTYPE)
    new_error[:n_error] = error
    new_error[:, ub[0] <= 0] = 0

    # new center
    new_center = torch.zeros(center.size(), dtype=settings.DTYPE)
    new_center[lb[0] >= 0] = center[lb[0] >= 0]

    # process 
    ub_select = ub[case_idx]
    lb_select = lb[case_idx]
    error_select = error[:, case_idx[1]]
    center_select = center[case_idx[1]]

    # deepzono
    slopes = ub_select / (ub_select - lb_select)
    mu = -slopes * lb_select / 2

    new_center[case_idx[1]] = slopes * center_select + mu
    new_error[:n_error, case_idx[1]] = error_select * slopes.unsqueeze(0)

    for e in range(n_new_error):
        new_error[n_error + e, case_idx[1][e]] = mu[e]

    new_center = new_center.view(center_orig_shape)
    new_error = new_error.view((new_error.shape[0],) + error_orig_shape[1:])

    return new_center, new_error


def linear_transform(layer, center, error):
    center = layer.weight @ center.squeeze()
    center = center.squeeze()
    if layer.bias is not None:
        if len(layer.bias) == 1:
            center = center.unsqueeze(0)
        center += layer.bias
    error = error @ layer.weight.permute(1, 0)
    return center, error


def conv_transform(layer, center, error):
    center = torch.nn.functional.conv2d(center, 
                                        layer.weight.to(settings.DTYPE), 
                                        layer.bias.to(settings.DTYPE), 
                                        stride=layer.stride,
                                        padding=layer.padding)
    error = torch.nn.functional.conv2d(error, 
                                       layer.weight.to(settings.DTYPE), 
                                       stride=layer.stride,
                                       padding=layer.padding)
    return center, error


def flatten_transform(center, error):
    center = center.flatten()
    error = error.view(error.shape[0], -1)
    return center, error



def get_bound(center, error):
    center, error = flatten_transform(center, error)
    error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
    ub = center + error_apt
    lb = center - error_apt
    lb, ub = lb.squeeze(), ub.squeeze()
    if len(lb.squeeze().shape) == 0:
        return lb.unsqueeze(0), ub.unsqueeze(0)
    return lb, ub


@torch.no_grad()
def forward(net, lower, upper):

    lbs = lower.view(net.input_shape)
    ubs = upper.view(net.input_shape)

    center = (ubs + lbs) / 2
    error = (ubs - lbs) / 2

    error = torch.diag(torch.ones(net.n_input) * error.flatten())
    error = error.view((net.n_input, ) + net.input_shape[1:])

    center = center.to(settings.DTYPE)
    error = error.to(settings.DTYPE)

    hidden_bounds = []

    for layer in net.layers:
        if isinstance(layer, nn.Linear):
            center, error = linear_transform(layer, center, error)
        elif isinstance(layer, nn.ReLU):
            hidden_bounds.append(get_bound(center, error))
            center, error = relu_transform(center, error)
        elif isinstance(layer, nn.Conv2d):
            center, error = conv_transform(layer, center, error)
        elif isinstance(layer, nn.Flatten):
            center, error = flatten_transform(center, error)
        else:
            raise NotImplementedError
        
    return get_bound(center, error), hidden_bounds


