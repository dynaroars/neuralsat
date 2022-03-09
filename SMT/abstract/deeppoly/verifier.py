import torch.nn as nn
import numpy as np
import torch
import time
import os

import transform
import network


DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
    start_pred = time.time()
    inputs_lx = inputs.detach() - eps * 1  # lower bound
    inputs_ux = inputs.detach() + eps * 1  # upper bound
    inputs_lx[inputs_lx < 0] = 0
    inputs_ux[inputs_ux > 1] = 1
    inputs = (inputs_ux - inputs_lx) / 2 + inputs_lx
    error_term_apt = (inputs_ux - inputs_lx) / 2
    error_term = torch.zeros((inputs.shape[1] * inputs.shape[2] * inputs.shape[3],
                              inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    k = 0
    for i in range(INPUT_SIZE):
        for j in range(INPUT_SIZE):
            error_term[k, 0, i, j] = error_term_apt[0, 0, i, j]
            k += 1

    for layer in net.layers:
        if type(layer) is torch.nn.modules.linear.Linear:
            # print('Linear layer')
            inputs, error_term = affine_transform(layer, inputs, error_term)

        elif type(layer) is torch.nn.modules.activation.ReLU:
            # print('Relu layer')
            inputs, error_term = relu_transform(inputs, error_term)

        elif type(layer) is torch.nn.modules.flatten.Flatten:
            # print('Flatten layer')
            inputs = inputs.view(1, 1, inputs.size()[1] * inputs.size()[2] * inputs.size()[3], 1)
            error_term = error_term.view(error_term.size()[0], 1,
                                         error_term.size()[1] * error_term.size()[2] * error_term.size()[3], 1)

        elif type(layer) is torch.nn.modules.conv.Conv2d:
            # print('Conv layer')
            inputs, error_term = conv_transform(layer, inputs, error_term)

        else:
            # print('Norm layer')
            mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
            sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
            inputs = (inputs - mean) / sigma
            error_term = error_term / sigma

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
    net = network.FullyConnected(2, [2, 2, 2])
    print(net)
    # net.load_state_dict(torch.load("fc1.pt", map_location=torch.device('cpu')))
    x = torch.randn([1, 1, 2, 2])
    y = net(x)

    pred_label = y.max(dim=1)[1].item()
    print('x', x)
    print('y', y)
    print(pred_label)

    model = zonotope.Model(net, eps=0.05, x=x, true_label=pred_label)
