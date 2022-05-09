import torch.nn as nn
import torch
import os


def relu_transform(center, error):
    # bounds
    error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
    ub = center + error_apt
    lb = center - error_apt

    # processing index
    case_idx = torch.where(((ub > 0) & (lb < 0)))

    # new error
    n_error, d = error.shape
    n_new_error = case_idx[0].shape[0]
    new_error = torch.zeros((n_error + n_new_error, d))
    new_error[:n_error] = error
    new_error[:, ub[0] <= 0] = 0

    # new center
    new_center = torch.zeros(center.size())
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

    return new_center, new_error


def linear_transform(layer, center, error):
    center = layer.weight.mm(center.unsqueeze(-1))
    center = center.squeeze()
    if layer.bias is not None:
        if len(layer.bias) == 1:
            center = center.unsqueeze(0)
        center += layer.bias
    error = error.mm(layer.weight.permute(1, 0))
    return center, error


def flatten_transform(center, error):
    center = center.view(1, -1)
    error = error.view(error.shape[0], -1)
    return center, error



def get_bound(center, error):
    error_apt = torch.sum(error.abs(), dim=0, keepdim=True)
    ub = center + error_apt
    lb = center - error_apt
    lb, ub = lb.squeeze(), ub.squeeze()
    if len(lb.squeeze().shape) == 0:
        return lb.unsqueeze(0), ub.unsqueeze(0)
    return lb, ub


@torch.no_grad()
def forward(net, lower, upper):
    center = (upper + lower) / 2
    error = (upper - lower) / 2

    h = lower.shape[0]

    error = torch.diag(torch.ones(h) * error.flatten())
    error = error.reshape((h, h))

    hidden_bounds = []

    # print('------------------------------------------')
    # print('input')
    # print('center', center.numpy().tolist())
    # print('error', error.numpy().tolist())
    # l, u = get_bound(center, error)
    # print('l', l.numpy().tolist())
    # print('u', u.numpy().tolist())
    # print()
    
    for layer in net.layers:
        if isinstance(layer, nn.Linear):
            center, error = linear_transform(layer, center, error)
        elif isinstance(layer, nn.ReLU):
            hidden_bounds.append(get_bound(center, error))
            center, error = relu_transform(center, error)
        else:
            raise NotImplementedError
        
    #     print(layer)
    #     print('center', center.numpy().tolist())
    #     print('error', error.numpy().tolist())
    #     l, u = get_bound(center, error)
    #     print('l', l.numpy().tolist())
    #     print('u', u.numpy().tolist())
    #     print()

    # print('------------------------------------------')
    return get_bound(center, error), hidden_bounds




# @torch.no_grad()
# def forward2(net, lower, upper, steps=2):
#     bounds = [(l, u) for l, u in zip(lower, upper)]
#     bounds = [torch.linspace(b[0], b[1], steps=steps) for b in bounds]
#     bounds = [[torch.Tensor([b[i], b[i+1]]) for i in range(b.shape[0] - 1)] for b in bounds]
#     bounds = itertools.product(*bounds)
#     splits = [(torch.Tensor([_[0] for _ in b]), torch.Tensor([_[1] for _ in b])) for b in bounds]

#     bounds = Parallel(n_jobs=os.cpu_count())(delayed(forward)(net, l, u) for l,u in splits)
#     lbs = torch.stack([b[0] for b in bounds]).squeeze()
#     ubs = torch.stack([b[1] for b in bounds]).squeeze()
#     return lbs.min(0).values, ubs.max(0).values

