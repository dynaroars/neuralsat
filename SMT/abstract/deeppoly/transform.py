import torch
import torch.nn as nn

def relu_transform(inputs, error_term):
    """
    Relu transformer
    :param inputs: torch.Tensor - size: (1, C, H, W)
    :param error_term: torch.Tensor - size: (error_term_num, C, H, W)
    :return:
    """
    error_apt = torch.sum(torch.abs(error_term), dim=0, keepdim=True).view(inputs.size())
    ux = inputs + error_apt
    lx = inputs - error_apt

    c = error_term.shape[1]
    h = error_term.shape[2]
    w = error_term.shape[3]
    assert c == inputs.shape[1], 'Relu transformer channel num error'
    assert h == inputs.shape[2], 'Relu transformer height num error'
    assert w == inputs.shape[3], 'Relu transformer width num error'

    err_num_ori = error_term.shape[0]
    case_idx = torch.where(((ux[0, :, :, :] > 0) & (lx[0, :, :, :] < 0)))   # where to compute lambda
    point_num = case_idx[0].shape[0]                                        # how many new error term to add
    error_term_new = torch.zeros((err_num_ori + point_num, c, h, w))        # new error term with new size
    outputs = torch.zeros(inputs.size())
    outputs[lx >= 0] = inputs[lx >= 0]                                      # lower bound >=0
    error_term_new[:err_num_ori, :, :, :] = error_term                      # lower bound >=0 error terms stay unchanged
    error_term_new[:, ux[0, :, :, :] <= 0] = 0                              # upper bound <= 0
    ux_select = ux[0][case_idx]
    lx_select = lx[0][case_idx]
    error_term_select = error_term[:, case_idx[0], case_idx[1], case_idx[2]]
    inputs_select = inputs[0][case_idx]
    slopes = ux_select / (ux_select - lx_select)    #lambda
    outputs[0][case_idx] = slopes * inputs_select - slopes * lx_select / 2
    error_term_new[:err_num_ori, case_idx[0], case_idx[1], case_idx[2]] = slopes.view((1, -1)) * error_term_select
    new_error_terms = -slopes * lx_select / 2
    for i in range(point_num):
        c_idx, h_idx, w_idx = case_idx[0][i], case_idx[1][i], case_idx[2][i]
        error_term_new[err_num_ori + i, c_idx, h_idx, w_idx] = new_error_terms[i]
    return outputs, error_term_new


def affine_transform(layer, inputs, error_term):
    """
    Affine transformer
    :param layer: torch.nn.module - layer to computer convex relaxation
    :param inputs:  torch.Tensor - size: (1, 1, feature_length, 1)
    :param error_term:  torch.Tensor - size: (error_term_num, 1, feature_length, 1)
    :return:
    """
    assert inputs.size()[2] == error_term.size()[2], 'Affine transformer error_term dimension error'
    outputs = (layer.weight.mm(inputs.detach()[0, 0, :, :])).view((layer.weight.shape[0])) + layer.bias
    outputs = outputs.view(1, 1, outputs.size()[0], 1)
    error = error_term[:, 0, :, 0].mm(layer.weight.permute(1, 0))  # transpose to do matrix mut
    error = error.view(error.size()[0], 1, error.size()[1], 1)
    return outputs, error


def conv_transform(layer, inputs, error_term):
    """
    Convolution transformer
    :param layer: torch.nn.module - layer to computer convex relaxation
    :param inputs: torch.Tensor - size: (1, C, H, W)
    :param error_term: torch.Tensor - size: (error_term_num, C, H, W)
    :return:
    """
    padding_1, padding_2 = layer.padding[0], layer.padding[1]
    stride_1, stride_2 = layer.stride[0], layer.stride[1]
    kernel_size_1, kernel_size_2 = layer.weight.size()[2], layer.weight.size()[3]
    assert kernel_size_1 == kernel_size_2, 'Convolution kernel sizes in 2 dimension are not equal!'
    assert padding_1 == padding_2, 'padding sizes not equal'
    assert stride_1 == stride_2, 'stride not equal'
    (error_term_num, c, h, w) = error_term.size()
    assert c == inputs.shape[1], 'Conv transformer channel num error'
    assert h == inputs.shape[2], 'Conv transformer height num error'
    assert w == inputs.shape[3], 'Conv transformer width num error'
    assert h == w, 'Conv: h and w not equal'

    outputs = torch.nn.functional.conv2d(inputs, layer.weight, layer.bias, stride=layer.stride,
                                         padding=layer.padding)
    error_term = error_term.view((error_term_num, c, h, w))
    error = torch.nn.functional.conv2d(error_term, layer.weight, stride=layer.stride,
                                       padding=layer.padding)
    return outputs, error