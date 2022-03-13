import torch.nn as nn
import torch

def relu_transform(center, error):
    n_error, d = error.shape
    error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
    
    # bounds
    ub = center + error_apt
    lb = center - error_apt

    # processing index
    case_idx = torch.where(((ub > 0) & (lb < 0)))

    # new error
    n_new_error = case_idx[0].shape[0]
    new_error = torch.zeros((n_error + n_new_error, d))
    new_error[:n_error] = error
    new_error[:, ub[0] <= 0] = 0

    # new center
    new_center = torch.zeros(center.size())
    new_center[lb >= 0] = center[lb >= 0]

    # process 
    ub_select = ub[case_idx]
    lb_select = lb[case_idx]
    error_select = error[:, case_idx[1]]
    center_select = center[case_idx]

    slopes = ub_select / (ub_select - lb_select)
    mu = -slopes * lb_select / 2
    new_center[case_idx] = slopes * center_select + mu
    new_error[:n_error, case_idx[1]] = error_select * slopes.unsqueeze(0)
    new_error[n_error:, case_idx[1]] = mu

    return new_center, new_error


def linear_transform(layer, center, error):
    center = layer.weight.mm(center[0].unsqueeze(-1)).squeeze(-1) + layer.bias
    center = center.unsqueeze(0)
    error = error.mm(layer.weight.permute(1, 0))
    return center, error

def flatten_transform(center, error):
    center = center.view(1, -1)
    error = error.view(error.shape[0], -1)
    return center, error

def forward(net, lower, upper):
    center = (upper + lower) / 2
    error = (upper - lower) / 2

    _, h, w = lower.shape

    error = torch.diag(torch.ones(h * w) * error.flatten())
    error = error.reshape((h * w, h, w))

    for layer in net.layers:
        if type(layer) is nn.modules.flatten.Flatten:
            center, error = flatten_transform(center, error)

        elif type(layer) is nn.modules.linear.Linear:
            center, error = linear_transform(layer, center, error)
            # print(center, center.shape)
            # print(error, error.shape)
        elif type(layer) is nn.modules.activation.ReLU:
            center, error = relu_transform(center, error)
        else:
            raise NotImplementedError

    return center, error