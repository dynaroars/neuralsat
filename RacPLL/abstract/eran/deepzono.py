import torch.nn as nn
import torch
import os

import settings


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


class DeepZono:

    def __init__(self, net):
        self.net = net
        self._build_network_transformer()


        self.from_layer = {k: [] for k in net.layers_mapping.keys()}

        for k in self.from_layer:
            idx = 0
            for layer in net.layers:
                if isinstance(layer, nn.ReLU):
                    idx += 1
                if k < idx:
                    if isinstance(layer, nn.Linear):
                        l = LinearTransformer(layer)
                    elif isinstance(layer, nn.ReLU):
                        l = ReLUTransformer()
                    else:
                        raise NotImplementedError
                    self.from_layer[k] += [l]

        # for k in self.from_layer:
        #     print(k, self.from_layer[k])
        # exit()

    @torch.no_grad()
    def forward_layer(self, lower, upper, layer_id):

        # lbs = lower.view(self.net.input_shape)
        # ubs = upper.view(self.net.input_shape)
        # print(lower.shape)
        # exit()
        lbs = lower.unsqueeze(0)
        ubs = upper.unsqueeze(0)

        center = (ubs + lbs) / 2
        error = (ubs - lbs) / 2

        error = torch.diag(torch.ones(len(lower)) * error.flatten())
        error = error.view((len(lower), len(lower)))

        center = center.to(settings.DTYPE)
        error = error.to(settings.DTYPE)

        for layer in self.from_layer[layer_id]:
            center, error = layer(center, error)
        return get_bound(center, error)

    def _build_network_transformer(self):
        self.layers = []
        for layer in self.net.layers:
            if isinstance(layer, nn.Linear):
                self.layers.append(LinearTransformer(layer))
            elif isinstance(layer, nn.ReLU):
                self.layers.append(ReLUTransformer())
            elif isinstance(layer, nn.Conv2d):
                self.layers.append(Conv2dTransformer(layer))
            elif isinstance(layer, nn.Flatten):
                self.layers.append(FlattenTransformer())
            else:
                raise NotImplementedError


    @torch.no_grad()
    def __call__(self, lower, upper):

        lbs = lower.view(self.net.input_shape)
        ubs = upper.view(self.net.input_shape)

        center = (ubs + lbs) / 2
        error = (ubs - lbs) / 2

        error = torch.diag(torch.ones(self.net.n_input) * error.flatten())
        error = error.view((self.net.n_input, ) + self.net.input_shape[1:])

        center = center.to(settings.DTYPE)
        error = error.to(settings.DTYPE)

        hidden_bounds = []

        for layer in self.layers:
            if isinstance(layer, ReLUTransformer):
                hidden_bounds.append(get_bound(center, error))
            center, error = layer(center, error)

        return get_bound(center, error), hidden_bounds


class LinearTransformer(nn.Module):

    def __init__(self, layer):
        super(LinearTransformer, self).__init__()
        self.weight = layer.weight.to(settings.DTYPE)
        self.bias = layer.bias.to(settings.DTYPE)

    def forward(self, center, error):
        center = self.weight @ center.squeeze()
        center = center.squeeze()
        if self.bias is not None:
            if len(self.bias) == 1:
                center = center.unsqueeze(0)
            center += self.bias
        error = error @ self.weight.permute(1, 0)
        return center, error



class ReLUTransformer(nn.Module):

    def __init__(self):
        super(ReLUTransformer, self).__init__()

    def forward(self, center, error):
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


class FlattenTransformer(nn.Module):

    def __init__(self):
        super(FlattenTransformer, self).__init__()

    def forward(self, center, error):
        center, error = flatten_transform(center, error)
        return center, error


class Conv2dTransformer(nn.Module):

    def __init__(self, layer):
        super(Conv2dTransformer, self).__init__()
        self.weight = layer.weight.to(settings.DTYPE)
        self.bias = layer.bias.to(settings.DTYPE)
        self.stride = layer.stride
        self.padding = layer.padding

    def forward(self, center, error):
        center = torch.nn.functional.conv2d(center, 
                                            self.weight, 
                                            self.bias, 
                                            stride=self.stride,
                                            padding=self.padding)
        error = torch.nn.functional.conv2d(error, 
                                           self.weight, 
                                           stride=self.stride,
                                           padding=self.padding)
        return center, error







