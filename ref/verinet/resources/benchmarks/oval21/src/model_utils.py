from torch import nn 
import torch
from torch.nn.parameter import Parameter
import copy
import numpy as np
from src.propagation import Propagation
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#### CIFAR
# 16*16*16 (4096) --> 32*8*8 (2048) --> 100 
# 6244 ReLUs
# wide model
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
    return model


# 16*16*8 (2048) -->  16*16*8 (2048) --> 16*16*8 (2048) --> 512 --> 100
# 6756 ReLUs
#deep model
def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


# 16*16*8 (2048) --> 16*8*8 (1024) --> 100 
# 3172 ReLUs (small model)
def cifar_model_m2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


# 16*16*4 (1024) --> 8*8*8 (512) --> 100 
def cifar_model_m1(): 
    model = nn.Sequential(
        nn.Conv2d(3, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_kw_net_loader(model):
    # Load the KW nets common in the OVAL CIFAR benchmarks, along with the correct input normalization.
    if model=='cifar_base_kw':
        model_name = './nets/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_wide_kw':
        model_name = './nets/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_deep_kw':
        model_name = './nets/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    return model, normalizer


def load_cifar_oval_kw_1_vs_all(model, idx, epsilon=2/255, max_solver_batch=10000, no_verif_layers=False,
                                cifar_test=None, use_cpu=False):
    """
    Version of the OVAL verification datasets consistent with the common NN verification practices: the epsilon is
    expressed before normalization, the adversarial robustness property is defined against all possible misclassified
    classes, the image is clipped to [0,1] before normalization.
    """

    model, normalizer = cifar_kw_net_loader(model)

    if cifar_test is None:
        cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))

    x, y = cifar_test[idx]

    # first check the model is correct at the input
    y_pred = torch.max(model(normalizer(x).unsqueeze(0))[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if y_pred != y:
        print('model prediction is incorrect for the given model')
        return None, None, None, None

    # Add epsilon in original space, clamp, normalize.
    domain = torch.stack(
        [normalizer((x - epsilon).clamp(0, 1)),
         normalizer((x + epsilon).clamp(0, 1))], dim=-1)

    if no_verif_layers:
        # Avoid computing the network model in canonical form for adversarial robustness, return the original net.
        return x, y, model, domain

    print('One vs all property')
    verification_layers = one_vs_all_from_model(model, y, domain=domain, max_solver_batch=max_solver_batch,
                                                use_cpu=use_cpu)
    return x, y, verification_layers, domain


def one_vs_all_from_model(model, true_label, domain=None, max_solver_batch=10000, use_cpu=False):
    """
        Given a pre-trained PyTorch network given by model, the true_label (ground truth) and the input domain for the
        property, create a network encoding a 1 vs. all adversarial verification task.
        The one-vs-all property is encoded exploiting a max-pool layer.
    """

    for p in model.parameters():
        p.requires_grad = False
    layers = list(model.children())

    last_layer = layers[-1]
    diff_in = last_layer.out_features
    diff_out = last_layer.out_features - 1
    diff_layer = nn.Linear(diff_in, diff_out, bias=True)
    temp_weight_diff = torch.eye(10)
    temp_weight_diff[:, true_label] -= 1
    all_indices = list(range(10))
    all_indices.remove(true_label)
    weight_diff = temp_weight_diff[all_indices]
    bias_diff = torch.zeros(9)

    diff_layer.weight = Parameter(weight_diff, requires_grad=False)
    diff_layer.bias = Parameter(bias_diff, requires_grad=False)
    layers.append(diff_layer)
    layers = simplify_network(layers)

    if not use_cpu:
        dev_verif_layers = [copy.deepcopy(lay).cuda() for lay in layers]
        dev_domain = domain.cuda().unsqueeze(0)
    else:
        dev_verif_layers = layers
        dev_domain = domain.unsqueeze(0)
    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = Propagation(dev_verif_layers, max_batch=max_solver_batch, type="best_prop",
                                   params={"best_among": ["KW", "crown"]})
    intermediate_net.define_linear_approximation(dev_domain)
    lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()

    candi_tot = diff_out
    # since what we are actually interested in is the minium of gt-cls,
    # we revert all the signs of the last layer
    max_pool_layers = max_pool(candi_tot, lbs, change_sign=True)

    # simplify linear layers
    simp_required_layers = layers[-1:] + max_pool_layers
    simplified_layers = simplify_network(simp_required_layers)

    final_layers = layers[:-1] + simplified_layers
    return final_layers


def max_pool(candi_tot, lb_abs, change_sign=False):
    '''
    diff layer is provided when simplify linear layers are required
    by providing linear layers, we reduce consecutive linear layers
    to one
    '''
    layers = []
    # perform max-pooling
    # max-pooling is performed in terms of paris.
    # Each loop iteration reduces the number of candidates by two
    while candi_tot > 1:
        temp = list(range(0, candi_tot//2))
        even = [2*i for i in temp]
        odd = [i+1 for i in even]
        max_pool_layer1 = nn.Linear(candi_tot, candi_tot, bias=True)
        weight_mp_1 = torch.eye(candi_tot)
        ####### replaced this
        # weight_mp_1[even,odd] = -1
        ####### with this
        for idl in even:
            weight_mp_1[idl, idl+1] = -1
        #######
        bias_mp_1 = torch.zeros(candi_tot)
        for idl in odd:
            bias_mp_1[idl] = -lb_abs[idl]
        bias_mp_1[-1] = -lb_abs[-1]
        #import pdb; pdb.set_trace()
        max_pool_layer1.weight = Parameter(weight_mp_1, requires_grad=False)
        max_pool_layer1.bias = Parameter(bias_mp_1, requires_grad=False)
        layers.append(max_pool_layer1)
        layers.append(nn.ReLU())
        new_candi_tot = (candi_tot+1)//2
        sum_layer = nn.Linear(candi_tot, new_candi_tot, bias=True)
        sum_layer_weight = torch.zeros([new_candi_tot, candi_tot])
        ####### replaced this
        # sum_layer_weight[temp,even]=1; sum_layer_weight[temp,odd]=1
        ####### with this
        for idl in temp:
            sum_layer_weight[idl, 2*idl] = 1; sum_layer_weight[idl, 2*idl+1]=1
        #######
        sum_layer_weight[-1][-1] = 1
        sum_layer_bias = torch.zeros(new_candi_tot)
        for idl in temp:
            sum_layer_bias[idl]= lb_abs[2*idl+1]
        sum_layer_bias[-1] = lb_abs[-1]
        if change_sign is True and new_candi_tot==1:
            sum_layer.weight = Parameter(-1*sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(-1*sum_layer_bias, requires_grad=False)
        else:
            sum_layer.weight = Parameter(sum_layer_weight, requires_grad=False)
            sum_layer.bias = Parameter(sum_layer_bias, requires_grad=False)
        layers.append(sum_layer)

        pre_candi_tot = candi_tot
        candi_tot = new_candi_tot
        pre_lb_abs = lb_abs
        lb_abs = np.zeros(new_candi_tot)
        for idl in temp:
            lb_abs[idl]= min(pre_lb_abs[2*idl], pre_lb_abs[2*idl+1])
        lb_abs[-1] = pre_lb_abs[-1]

    return layers


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
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
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


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class View(nn.Module):
    '''
    This is necessary in order to reshape "flat activations" such as used by
    nn.Linear with those that comes from MaxPooling
    '''
    def __init__(self, out_shape):
        super(View, self).__init__()
        self.out_shape = out_shape

    def forward(self, inp):
        # We make the assumption that all the elements in the tuple have
        # the same batchsize and need to be brought to the same size

        # We assume that the first dimension is the batch size
        batch_size = inp.size(0)
        out_size = (batch_size, ) + self.out_shape
        out = inp.view(out_size)
        return out