from typing import Tuple
from torch import nn
import numpy as np
# from numpy.core.defchararray import upper
import torch
from torch.nn.modules.activation import ReLU

from collections import namedtuple
from itertools import product


DTYPE = np.float64

class Box():
    '''Represents the constraints for a single layer in a network.'''
    
    l: np.ndarray
    u: np.ndarray

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.l = np.array(lower_bounds)
        self.u = np.array(upper_bounds)

    def flatten(self):
        return self.l.flatten(), self.u.flatten()

    def __repr__(self):
        return ( f"Box l = {self.l}\n" +
                f"Box u = {self.u}" )


def transform_box(box: Box, layer: nn.Module) -> Box:

    if isinstance(layer, nn.Linear):
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        lower_bounds = b + np.sum(np.minimum(W * box.l, W * box.u), axis = 1)
        upper_bounds = b + np.sum(np.maximum(W * box.l, W * box.u), axis = 1)
        
    elif isinstance(layer, nn.ReLU):
        lower_bounds = np.maximum(box.l, 0)
        upper_bounds = np.maximum(box.u, 0)

    else:
        raise NotImplementedError

    return Box(lower_bounds, upper_bounds)






class DeepPoly():
    l_bias: np.ndarray
    l_weights: np.ndarray
    u_bias: np.ndarray
    u_weights: np.ndarray
    box: Box
    # in_dpoly: DeepPoly # TODO 
    name: str # useful for debugging
    layer_shape: Tuple[int]
    in_shape: Tuple[int]

    def __init__(self, in_dpoly, l_bias, l_weights, u_bias, u_weights, box = None,
            layer_shape = None, name = ""):
        self.l_bias = np.array(l_bias, dtype = DTYPE)
        self.l_weights = np.array(l_weights, dtype = DTYPE)
        self.u_bias = np.array(u_bias, dtype = DTYPE)
        self.u_weights = np.array(u_weights, dtype = DTYPE)
        self.in_dpoly = in_dpoly
        self.name = name

        # infers layer_shape
        if layer_shape is not None:
            self.layer_shape = layer_shape
        elif l_bias is not None:
            self.layer_shape = self.l_bias.shape
        elif box is not None:
            self.layer_shape = box.l.shape
        else:
            raise Exception("Layer's shape can't be infered.")

        # infers in_shape
        if l_weights is not None:
            layer_ndim = len(self.layer_shape)
            self.in_shape = self.l_weights.shape[layer_ndim:]
        elif in_dpoly is not None:
            self.in_shape = in_dpoly.layer_shape
        else:
            print("Warning: Input's shape can't be infered.")

        if box is None:
            self.calculate_box()
        else:
            self.box = box

    def calculate_box(self):
        if self.in_dpoly is None:
            self.box = None
            return

        lb, lW, ub, uW = self.biflatten()
        prev_box_l, prev_box_u = self.in_dpoly.box.flatten()
        box_l = lb + np.sum(np.minimum(lW * prev_box_l, lW * prev_box_u), 
            axis = 1)
        box_u = ub + np.sum(np.maximum(uW * prev_box_l, uW * prev_box_u), 
            axis = 1)
        self.box = Box(box_l.reshape(self.layer_shape), box_u.reshape(self.layer_shape))

    def biflatten(self):
        lb = self.l_bias.flatten()
        lW = self.l_weights.reshape(self.layer_size(), self.in_size())
        ub = self.u_bias.flatten()
        uW = self.u_weights.reshape(self.layer_size(), self.in_size())
        return lb, lW, ub, uW

    def l_combined(self) -> np.ndarray:
        """Merges weights and bias."""
        lb, lW, _, _ = self.biflatten()
        return np.hstack([np.expand_dims(lb, 1), lW])

    def u_combined(self) -> np.ndarray:
        _, _, ub, uW = self.biflatten()
        return np.hstack([np.expand_dims(ub, 1), uW])

    def l_combined_ones(self) -> np.ndarray:
        l_combined = self.l_combined()
        return np.vstack([np.ones(l_combined.shape[1]), l_combined])

    def u_combined_ones(self) -> np.ndarray:
        u_combined = self.u_combined()
        return np.vstack([np.ones(u_combined.shape[1]), u_combined])

    def get_neur(self, idx):
        AbstractNeuron = namedtuple('AbstractNeuron', 'l u')
        return AbstractNeuron(
            l = self.l_combined()[idx],
            u = self.u_combined()[idx]
        )

    def layer_size(self):
        return int(np.product(self.layer_shape))
        
    def in_size(self):
        return np.product(self.in_shape)

    def update_neur(self, idx, new_lb, new_lw, new_ub, new_uw):
        self.l_bias[idx] = new_lb
        self.l_weights[idx] = new_lw
        self.u_bias[idx] = new_ub
        self.u_weights[idx] = new_uw

    def layer_iterator(self):
        return product(*map(range, self.layer_shape))

    def __repr__(self):
        lines = [f"DPoly {self.name} | shape {self.layer_shape}"]
        for idx in self.layer_iterator():
            lines.append(f"neur{list(idx)}: l = {self.box.l[idx]}, u = {self.box.u[idx]}")

        return "\n".join(lines)








def anull_neg(f):
    return f * (f >= 0)


def anull_nonneg(f):
    return f * (f < 0)


def affine_expand(mat):
    exp_mat = np.vstack([np.zeros(mat.shape[1]), mat])
    exp_mat[0, 0] = 1
    return exp_mat


def affine_substitute_eq(f, sub_mat):
    return f @ affine_expand(sub_mat)

def affine_substitute_lt(f, sub_mat_l, sub_mat_u):
    return (anull_neg(f) @ affine_expand(sub_mat_u) + anull_nonneg(f) @ affine_expand(sub_mat_l))

def affine_substitute_gt(f, sub_mat_l, sub_mat_u):
    return (anull_neg(f) @ affine_expand(sub_mat_l) + anull_nonneg(f) @ affine_expand(sub_mat_u))

def split_wb(combined: np.ndarray):
    return combined[0], combined[1:]

def backsub_transform(dpoly: DeepPoly):
    in_dpoly: DeepPoly = dpoly.in_dpoly
    new_l_w = []
    new_l_b = []
    new_u_w = []
    new_u_b = []
    for i in range(dpoly.layer_size()):
        neur = dpoly.get_neur(i)
        tmp_l_b, tmp_l_w = split_wb(affine_substitute_gt(
            neur.l, in_dpoly.l_combined(), in_dpoly.u_combined()))
        new_l_b.append(tmp_l_b)
        new_l_w.append(tmp_l_w)

        tmp_u_b, tmp_u_w = split_wb(affine_substitute_lt(
            neur.u, in_dpoly.l_combined(), in_dpoly.u_combined()))
        new_u_b.append(tmp_u_b)
        new_u_w.append(tmp_u_w)

    new_l_b = np.array(new_l_b).reshape(dpoly.layer_shape)
    new_l_w = np.array(new_l_w).reshape((*dpoly.layer_shape, *in_dpoly.in_shape))
    new_u_b = np.array(new_u_b).reshape(dpoly.layer_shape)
    new_u_w = np.array(new_u_w).reshape((*dpoly.layer_shape, *in_dpoly.in_shape))

    return DeepPoly(in_dpoly.in_dpoly, new_l_b, new_l_w, new_u_b, new_u_w)


def layer_transform(in_dpoly: DeepPoly, layer: nn.Module):
    if isinstance(layer, nn.Linear):
       return linear_transform(in_dpoly, layer)   
    elif isinstance(layer, nn.ReLU):
        return relu_transform(in_dpoly)
    else:
        raise NotImplementedError



def linear_transform(in_dpoly: DeepPoly, layer: nn.Linear):
    W = layer.weight.detach().numpy()
    if layer.bias is not None:
        b = layer.bias.detach().numpy()
    else:
        b = np.zeros(W.shape[0])
    l_bias = b.copy()
    l_weights = W.copy()
    u_bias = b.copy()
    u_weights = W.copy()

    return DeepPoly(in_dpoly, l_bias, l_weights, u_bias, u_weights)


def relu_transform(in_dpoly: DeepPoly):
    n_neur = in_dpoly.layer_size()
    l_bias = np.zeros(n_neur, dtype = DTYPE)
    l_weights = np.zeros((n_neur, n_neur), dtype = DTYPE)
    u_bias = l_bias.copy()
    u_weights = l_weights.copy()

    neg_idx = in_dpoly.box.u <= 0
    # all values already set to 0

    pos_idx = in_dpoly.box.l >= 0
    # l_bias[pos_idx] = in_dpoly.l_bias[pos_idx]
    # l_bias already set to 0
    l_weights[pos_idx] = np.eye(n_neur)[pos_idx]
    # u_bias[pos_idx] = in_dpoly.u_bias[pos_idx]
    # u_bias already set to 0
    u_weights[pos_idx] = np.eye(n_neur)[pos_idx]

    crossing_idx = ~(neg_idx | pos_idx)
    slope = (in_dpoly.box.u) / (in_dpoly.box.u - in_dpoly.box.l)
    y_intercept = - slope * in_dpoly.box.l
    # l_bias already set to 0
    # l_weights already set to 0
    u_bias[crossing_idx] = y_intercept[crossing_idx]
    u_weights[crossing_idx] = np.diag(slope)[crossing_idx]

    return DeepPoly(in_dpoly, l_bias, l_weights, u_bias, u_weights)



def _create_input_dpoly(lower, upper):
    # box_l = np.maximum(input_range[0], inputs - eps)
    # box_u = np.minimum(input_range[1], inputs + eps)
    # Create a deep_poly for inputs
    dpoly_shape = (*lower.shape, 0)
    return DeepPoly(
            None,
            lower,
            np.zeros(dpoly_shape),
            upper,
            np.zeros(dpoly_shape),
            Box(lower, upper)
    )

def backsubstitute(start_dpoly):
    if start_dpoly.in_dpoly is None:
        return

    curr_dp = start_dpoly
    while curr_dp.in_dpoly.in_dpoly is not None:
        curr_dp = backsub_transform(curr_dp)
    start_dpoly.box = curr_dp.box


import torch.nn.functional as F
import torch.nn as nn
import torch


class CorinaNet(nn.Module):

    def __init__(self):
        super().__init__()

        fc1 = nn.Linear(2, 2, bias=False)
        fc1.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [1, 1]]))

        fc2 = nn.Linear(2, 2, bias=False)
        fc2.weight = torch.nn.Parameter(torch.Tensor([[0.5, -0.2], [-0.5, 0.1]]))

        fc3 = nn.Linear(2, 2, bias=False)
        fc3.weight = torch.nn.Parameter(torch.Tensor([[1, -1], [-1, 1]]))

        self.layers = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)


    def forward(self, x):
        return self.layers(x)


class FC(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        layers = []
        prev_size = input_size
        for idx, size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, size))
            if idx < len(hidden_sizes) - 1:
                layers.append(nn.ReLU())
            prev_size = size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def forward(net, lower, upper):
    dpolys = [_create_input_dpoly(lower, upper)]

    for layer in net.layers:
        dpolys.append(layer_transform(dpolys[-1], layer))
        if isinstance(layer, nn.Linear):
            backsubstitute(dpolys[-1])

    print(dpolys[-1])

if __name__ == '__main__':
    
    net = CorinaNet().eval()
    lower = np.array([-5, -4])
    upper = np.array([-1, -2])

    net = FC(5, [2, 2, 5]).eval()
    lower = torch.Tensor([-5, -4, 1, 2, 3])
    upper = torch.Tensor([-1, -2, 4, 5, 6])

    forward(net, lower, upper)