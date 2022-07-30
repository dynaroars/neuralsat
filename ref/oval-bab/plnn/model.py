import math
import torch
from tools.custom_torch_modules import Flatten, View, Add, supported_transforms
from plnn.naive_approximation import NaiveNetwork
from torch import nn


def cifar_model_large():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model


def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model


GE='>='
LE='<='
COMPS = [GE, LE]


def simplify_network(all_layers):
    '''
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    '''
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias = nn.Parameter(joint_bias, requires_grad=False)
            joint_layer.weight = nn.Parameter(joint_weight, requires_grad=False)
            new_all_layers.append(joint_layer)
        elif (type(layer) is nn.MaxPool1d) and (layer.kernel_size == 1) and (layer.stride == 1):
            # This is just a spurious Maxpooling because the kernel_size is 1
            # We will do nothing
            pass
        elif (type(layer) is View) and (type(new_all_layers[-1]) is View):
            # No point in viewing twice in a row
            del new_all_layers[-1]

            # Figure out what was the last thing that imposed a shape
            # and if this shape was the proper one.
            prev_layer_idx = -1
            lay_nb_dim_inp = 0
            while True:
                parent_lay = new_all_layers[prev_layer_idx]
                prev_layer_idx -= 1
                if type(parent_lay) is nn.ReLU:
                    # Can't say anything, ReLU is flexible in dimension
                    continue
                elif type(parent_lay) is nn.Linear:
                    lay_nb_dim_inp = 1
                    break
                elif type(parent_lay) is nn.MaxPool1d:
                    lay_nb_dim_inp = 2
                    break
                else:
                    raise NotImplementedError
            if len(layer.out_shape) != lay_nb_dim_inp:
                # If the View is actually necessary, add the change
                new_all_layers.append(layer)
                # Otherwise do nothing
        else:
            new_all_layers.append(layer)
    return new_all_layers


def remove_maxpools(layers, domain, dtype=torch.float):
    """
    Given the list of layers and the domain, transform maxpools into equivalent relu-based layers.
    """
    if any(map(lambda x: isinstance(x, (nn.MaxPool1d, nn.MaxPool2d)), layers)):
        new_layers = simplify_network(reluify_maxpool(layers, domain, dtype=dtype))
        return new_layers
    else:
        return layers


def reluify_maxpool(layers, domain, dtype=torch.float32):
    '''
    Remove all the Maxpool units of a feedforward network represented by
    `layers` and replace them by an equivalent combination of ReLU + Linear

    This is only valid over the domain `domain` because we use some knowledge
    about upper and lower bounds of certain neurons
    '''
    naive_net = NaiveNetwork(layers)
    naive_net.do_interval_analysis(domain)
    lbs = naive_net.lower_bounds
    layers = layers[:]

    def prod(lst):
        out = 1
        for el in lst:
            out *= el
        return out

    new_all_layers = []

    idx_of_inp_lbs = 0
    layer_idx = 0
    while layer_idx < len(layers):
        layer = layers[layer_idx]
        if type(layer) is nn.MaxPool1d:
            # We need to decompose this MaxPool until it only has a size of 2
            assert layer.padding == 0
            assert layer.dilation == 1
            if layer.kernel_size > 2:
                assert layer.kernel_size % 2 == 0, "Not supported yet"
                assert layer.stride % 2 == 0, "Not supported yet"
                # We're going to decompose this maxpooling into two maxpooling
                # max(     in_1, in_2 ,      in_3, in_4)
                # will become
                # max( max(in_1, in_2),  max(in_3, in_4))
                first_mp = nn.MaxPool1d(2, stride=2)
                second_mp = nn.MaxPool1d(layer.kernel_size // 2,
                                         stride=layer.stride // 2)
                # We will replace the Maxpooling that was originally there with
                # those two layers
                # We need to add a corresponding layer of lower bounds
                first_lbs = lbs[idx_of_inp_lbs]
                intermediate_lbs = []
                for pair_idx in range(len(first_lbs) // 2):
                    intermediate_lbs.append(max(first_lbs[2*pair_idx],
                                                first_lbs[2*pair_idx+1]))
                # Do the replacement
                del layers[layer_idx]
                layers.insert(layer_idx, first_mp)
                layers.insert(layer_idx+1, second_mp)
                lbs.insert(idx_of_inp_lbs+1, intermediate_lbs)

                # Now continue so that we re-go through the loop with the now
                # simplified maxpool
                continue
            elif layer.kernel_size == 2:
                # Each pair need two in the intermediate layers that is going
                # to be Relu-ified
                pre_nb_inp_lin = len(lbs[idx_of_inp_lbs])
                # How many starting position can we fit in?
                # 1 + how many stride we can fit before we're too late in the array to fit a kernel_size
                pre_nb_out_lin = (1 + ((pre_nb_inp_lin - layer.kernel_size) // layer.stride)) * 2
                pre_relu_lin = nn.Linear(pre_nb_inp_lin, pre_nb_out_lin, bias=True)
                pre_relu_lin.weight.data = pre_relu_lin.weight.data.type(dtype)
                pre_relu_lin.bias.data = pre_relu_lin.bias.data.type(dtype)
                pre_relu_weight = pre_relu_lin.weight.data
                pre_relu_bias = pre_relu_lin.bias.data
                pre_relu_weight.zero_()
                pre_relu_bias.zero_()
                # For each of (x, y) that needs to be transformed to max(x, y)
                # We create (x-y, y-y_lb)
                first_in_index = 0
                first_out_index = 0
                while first_in_index + 1 < pre_nb_inp_lin:
                    pre_relu_weight[first_out_index, first_in_index] = 1
                    pre_relu_weight[first_out_index, first_in_index+1] = -1

                    pre_relu_weight[first_out_index+1, first_in_index+1] = 1
                    pre_relu_bias[first_out_index+1] = -lbs[idx_of_inp_lbs][first_in_index + 1]

                    # Now shift
                    first_in_index += layer.stride
                    first_out_index += 2
                new_all_layers.append(pre_relu_lin)
                new_all_layers.append(nn.ReLU())

                # We now need to create the second layer
                # It will sum [max(x-y, 0)], [max(y - y_lb, 0)] and y_lb
                post_nb_inp_lin = pre_nb_out_lin
                post_nb_out_lin = post_nb_inp_lin // 2
                post_relu_lin = nn.Linear(post_nb_inp_lin, post_nb_out_lin)
                post_relu_lin.weight.data = post_relu_lin.weight.data.type(dtype)
                post_relu_lin.bias.data = post_relu_lin.bias.data.type(dtype)
                post_relu_weight = post_relu_lin.weight.data
                post_relu_bias = post_relu_lin.bias.data
                post_relu_weight.zero_()
                post_relu_bias.zero_()
                first_in_index = 0
                out_index = 0
                while first_in_index + 1 < post_nb_inp_lin:
                    post_relu_weight[out_index, first_in_index] = 1
                    post_relu_weight[out_index, first_in_index+1] = 1
                    post_relu_bias[out_index] = lbs[idx_of_inp_lbs][layer.stride*out_index+1]
                    first_in_index += 2
                    out_index += 1
                new_all_layers.append(post_relu_lin)
                idx_of_inp_lbs += 1
            else:
                # This should have been cleaned up in one of the simplify passes
                raise NotImplementedError
        elif type(layer) is nn.MaxPool2d:
            # We need to decompose this MaxPool until it only has a size of 2
            assert sum(layer.padding) == 0 if isinstance(layer.padding, tuple) else layer.padding == 0
            assert layer.dilation == 1
            if prod(layer.kernel_size) > 2:
                for ck in layer.kernel_size:
                    assert ck % 2 == 0, "Not supported yet"
                for cs in layer.stride:
                    assert cs % 2 == 0, "Not supported yet"
                # We're going to decompose this maxpooling into two maxpooling
                # max(     in_1, in_2 ,      in_3, in_4)
                # will become
                # max( max(in_1, in_2),  max(in_3, in_4))

                if prod(layer.kernel_size) > 4:
                    first_mp = nn.MaxPool2d((2, 2), stride=(2,2))
                    second_mp = nn.MaxPool2d([ck // 2 for ck in layer.kernel_size],
                                             stride=[cs // 2 for cs in layer.stride])
                else:
                    first_mp = nn.MaxPool2d((2, 1), stride=(2, 1))
                    second_mp = nn.MaxPool2d((1, 2), stride=(1, 2))

                # We will replace the Maxpooling that was originally there with
                # those two layers
                # We need to add a corresponding layer of lower bounds
                first_lbs = lbs[idx_of_inp_lbs]
                # Do the replacement
                del layers[layer_idx]
                layers.insert(layer_idx, first_mp)
                layers.insert(layer_idx+1, second_mp)
                lbs.insert(idx_of_inp_lbs+1, first_mp(first_lbs.unsqueeze(0)).squeeze(0))

                # Now continue so that we re-go through the loop with the now
                # simplified maxpool
                continue
            elif prod(layer.kernel_size) == 2:
                # Each pair need two in the intermediate layers that is going
                # to be Relu-ified

                # For each of (x, y) that needs to be transformed to max(x, y)
                # We create (x-y, y-y_lb) in two separate channels, which will be summed after the ReLU.
                in_channels = lbs[idx_of_inp_lbs].shape[0]
                out_channels = lbs[idx_of_inp_lbs].shape[0] * 2
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=layer.kernel_size, stride=layer.stride)
                conv_layer.weight.data = conv_layer.weight.data.type(dtype)
                conv_layer.bias.data = conv_layer.bias.data.type(dtype)
                conv_layer_w = conv_layer.weight.data
                conv_layer_b = conv_layer.bias.data
                conv_layer_w.zero_()
                conv_layer_b.zero_()
                for inchan in range(in_channels):
                    conv_layer_w[inchan, inchan] = torch.tensor([1, -1]).view_as(conv_layer_w[inchan, inchan])
                    conv_layer_w[in_channels + inchan, inchan] = torch.tensor([0, 1]).view_as(conv_layer_w[inchan, inchan])

                # As one can't pass -y_lb as bias, need it as a post layer transform
                out_ex = conv_layer(lbs[idx_of_inp_lbs].unsqueeze(0)).squeeze(0)
                lb_bias = torch.zeros_like(out_ex)
                lb_bias[in_channels:] = out_ex[in_channels:].clone()
                bias_op = Add(-lb_bias)
                new_all_layers.extend([conv_layer, bias_op, nn.ReLU()])

                # We now need to create the second layer
                # It will sum [max(x-y, 0)], [max(y - y_lb, 0)] and y_lb
                conv_layer = nn.Conv2d(out_channels, in_channels, kernel_size=(1, 1), stride=(1, 1))
                conv_layer.weight.data = conv_layer.weight.data.type(dtype)
                conv_layer.bias.data = conv_layer.bias.data.type(dtype)
                conv_layer_w = conv_layer.weight.data
                conv_layer_b = conv_layer.bias.data
                conv_layer_w.zero_()
                conv_layer_b.zero_()
                for inchan in range(in_channels):
                    conv_layer_w[inchan, inchan] = torch.tensor([1]).view_as(conv_layer_w[inchan, inchan])
                    conv_layer_w[inchan, in_channels + inchan] = torch.tensor([1]).view_as(conv_layer_w[inchan, inchan])
                bias_op = Add(out_ex[in_channels:])

                new_all_layers.extend([conv_layer, bias_op])
                idx_of_inp_lbs += 1
            else:
                # This should have been cleaned up in one of the simplify passes
                raise NotImplementedError
        elif isinstance(layer, (nn.Linear, nn.Conv2d)):
            new_all_layers.append(layer)
            idx_of_inp_lbs += 1
        elif isinstance(layer, (nn.ReLU, *supported_transforms, nn.AvgPool2d, nn.ConstantPad2d)):
            new_all_layers.append(layer)
        elif type(layer) is View:
            # We shouldn't add the view as we are getting rid of them
            pass
        layer_idx += 1
    return new_all_layers
