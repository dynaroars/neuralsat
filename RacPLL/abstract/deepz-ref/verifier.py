import torch.nn as nn
import numpy as np
import torch
import time
import os

from transform import *
import network


DEVICE = 'cpu'
INPUT_SIZE = 2


def analyze(net, inputs, eps, true_label):
    start_pred = time.time()

    inputs_ux = torch.clamp(inputs.data + eps, max=1)
    inputs_lx = torch.clamp(inputs.data - eps, min=0)    # 
    inputs = (inputs_ux + inputs_lx) / 2

    error_term = (inputs_ux - inputs_lx) / 2
    # print('ub', inputs_ux)
    # print('lb', inputs_lx)
    error_term = torch.diag(torch.ones(INPUT_SIZE * INPUT_SIZE) * error_term.flatten())
    error_term = error_term.reshape((INPUT_SIZE * INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE))



    for layer in net.layers:
        if type(layer) is torch.nn.modules.linear.Linear:
            # print('Linear layer')
            inputs, error_term = affine_transform(layer, inputs, error_term)
            # print(inputs, error_term)
            # exit()

        elif type(layer) is torch.nn.modules.activation.ReLU:
            # print('Relu layer')
            inputs, error_term = relu_transform(inputs, error_term)

        elif type(layer) is torch.nn.modules.flatten.Flatten:
            # print('Flatten layer')
            inputs = inputs.view(1, 1, inputs.size()[1] * inputs.size()[2] * inputs.size()[3], 1)
            error_term = error_term.view(error_term.size()[0], 1,
                                         error_term.size()[1] * error_term.size()[2] * error_term.size()[3], 1)

            # print(inputs, inputs.shape)
            # print(error_term, error_term.shape)
            # exit()

        elif type(layer) is torch.nn.modules.conv.Conv2d:
            # print('Conv layer')
            inputs, error_term = conv_transform(layer, inputs, error_term)

        else:
            # print('Norm layer')
            mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
            sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
            inputs = (inputs - mean) / sigma
            error_term = error_term / sigma


    print(inputs)
    print(error_term)
    exit()

    # error_term_view = error_term.view((error_term.shape[0], 10)).detach().numpy().copy()
    # old version without this line:
    error_term = error_term - error_term[:, :, true_label, :].view((error_term.shape[0], 1, 1, 1))
    # error_term_view_2 = error_term.view((error_term.shape[0], 10)).detach().numpy()
    error_apt = torch.sum(torch.abs(error_term), dim=0, keepdim=True).view(inputs.size())
    inputs_ux = inputs + error_apt  # upper bound
    inputs_lx = inputs - error_apt  # lower bound
    true_label_lx = inputs_lx[0, 0, true_label, 0].detach().numpy()
    labels_ux = inputs_ux.detach().numpy()
    labels_ux = np.delete(labels_ux, [true_label])
    end_pred = time.time()
    # print("prediction time: {}".format(end_pred - start_pred))
    if (true_label_lx > labels_ux).all():
        return True
    else:
        return False

if __name__ == '__main__':
    torch.manual_seed(0)
    net = network.FullyConnected(2, [22, 21, 40, 5])
    # net.load_state_dict(torch.load("fc1.pt", map_location=torch.device('cpu')))
    x = torch.rand([1, 1, 2, 2])
    x = x / x.abs().max()
    # print(x)
    y = net(x)

    pred_label = y.max(dim=1)[1].item()
    # print('x', x)
    # print('y', y)
    # print(pred_label)

    res = analyze(net, x, 1e-2, 1)
    print(res)

    # model = zonotope.Model(net, eps=0.05, x=x, true_label=pred_label)
