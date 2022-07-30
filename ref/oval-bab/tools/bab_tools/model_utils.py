from torch import nn 
import torch
from plnn.model import simplify_network
from tools.custom_torch_modules import View, Flatten
from torch.nn.parameter import Parameter
import random
import copy
from tools.colt_layers import Conv2d, Normalization, ReLU, Linear, Sequential
from tools.colt_layers import Flatten as flatten
import numpy as np
from plnn.proxlp_solver.propagation import Propagation
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# TODO: remove OVAL+ from verification_properties and code? Same with OVAL COLT?
'''
Code adapted from GNN_branching (author: Jingyue Lu)
This file contains all model structures we have considered
'''

## original kw small model
## 14x14x16 (3136) --> 7x7x32 (1568) --> 100 --> 10 ----(4804 ReLUs)
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

## 14*14*8 (1568) --> 14*14*8 (1568) --> 14*14*8 (1568) --> 392 --> 100 (5196 ReLUs)
def mnist_model_deep():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# first medium size model 14x14x4 (784) --> 7x7x8 (392) --> 50 --> 10 ----(1226 ReLUs)
# robust error 0.068
def mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*7*7,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# increase the mini model by increasing the number of channels
## 8x8x8 (512) --> 4x4x16 (256) --> 50 (50) --> 10 (818)
def mini_mnist_model_m1():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(4*4*16,50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    return model


# without the extra 50-10 layer (originally, directly 128-10, robust error is around 0.221)
## 8x8x4 (256) --> 4x4x8 (128) --> 50 --> 10 ---- (434 ReLUs)
def mini_mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*4*4,50),
        nn.ReLU(),
        nn.Linear(50,10),
    )
    return model

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

def add_single_prop(layers, gt, cls):
    '''
    gt: ground truth lable
    cls: class we want to verify against
    '''
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt] = 1

    #verif_layers2 = flatten_layers(verif_layers1,[1,14,14])
    final_layers = [layers[-1], additional_lin_layer]
    final_layer  = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


def cifar_kw_net_loader(model):
    # Load the KW nets common in the OVAL CIFAR benchmarks, along with the correct input normalization.
    if model=='cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_wide_kw':
        model_name = './models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_deep_kw':
        model_name = './models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    elif model=='cifar_madry':
        model_name = './models/cifar_madry_8px.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
    return model, normalizer


# OVAL cifar dataset -- the epsilon is expressed in the normalized space, without clamping.
def load_cifar_1to1_exp(model, idx, test = None, cifar_test = None, printing=False, return_true_class=False):

    model, normalizer = cifar_kw_net_loader(model)

    if cifar_test is None:
        cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalizer]))

        # for local usage:
        # cifar_test = datasets.CIFAR10('./cifardata.nosync/', train=False,
        #                               transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)

    x, y = cifar_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    if printing:
        print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        if return_true_class:
            return None, None, None, None, None
        else:
            return None, None, None
    else: 
        if test ==None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        if printing:
            print('tested against ',test)
        for p in model.parameters():
            p.requires_grad =False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        if return_true_class:
            return x, added_prop_layers, test, y_pred, model
        else:
            return x, added_prop_layers, test


def load_cifar_oval_kw_1_vs_all(model, idx, epsilon=2/255, max_solver_batch=10000, no_verif_layers=False,
                                cifar_test=None):
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
    verification_layers = one_vs_all_from_model(model, y, domain=domain, max_solver_batch=max_solver_batch)
    return x, y, verification_layers, domain


##########################################################################################
########################## Colt Networks for CAV workshop ################################
##########################################################################################
mnist_mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
mnist_sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
cifar_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
cifar_sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))

def get_mean_sigma(dataset):
    if dataset == "cifar_colt":
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    elif dataset == "mnist_colt":
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean, sigma

def reluified_max_pool(candi_tot, lb_abs, flip_out_sign=False, dtype=torch.float32):
    '''
    diff layer is provided when simplify linear layers are required
    by providing linear layers, we reduce consecutive linear layers
    to one
    '''
    layers = []
    # perform max-pooling
    # max-pooling is performed in terms of pairs.
    # Each loop iteration reduces the number of candidates by two
    while candi_tot > 1:
        temp = list(range(0, candi_tot//2))
        even = [2*i for i in temp]
        odd = [i+1 for i in even]
        max_pool_layer1 = nn.Linear(candi_tot, candi_tot, bias=True)
        weight_mp_1 = torch.eye(candi_tot, dtype=dtype)
        ####### replaced this
        # weight_mp_1[even,odd] = -1
        ####### with this
        for idl in even:
            weight_mp_1[idl, idl+1] = -1
        #######
        bias_mp_1 = torch.zeros(candi_tot, dtype=dtype)
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
        sum_layer_weight = torch.zeros([new_candi_tot, candi_tot], dtype=dtype)
        ####### replaced this
        # sum_layer_weight[temp,even]=1; sum_layer_weight[temp,odd]=1
        ####### with this
        for idl in temp:
            sum_layer_weight[idl, 2*idl] = 1; sum_layer_weight[idl, 2*idl+1]=1
        #######
        sum_layer_weight[-1][-1] = 1
        sum_layer_bias = torch.zeros(new_candi_tot, dtype=dtype)
        for idl in temp:
            sum_layer_bias[idl]= lb_abs[2*idl+1]
        sum_layer_bias[-1] = lb_abs[-1]
        if flip_out_sign is True and new_candi_tot==1:
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


def one_vs_all_from_model(model, true_label, domain=None, max_solver_batch=10000, use_ib=False, gpu=True):
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

    verif_layers = [copy.deepcopy(lay).cuda() for lay in layers] if gpu else layers
    if not use_ib:
        # use best of CROWN and KW as intermediate bounds
        intermediate_net = Propagation(verif_layers, max_batch=max_solver_batch, type="best_prop",
                                       params={"best_among": ["KW", "crown"]})
    else:
        # use IBP for intermediate bounds
        intermediate_net = Propagation(verif_layers, max_batch=max_solver_batch, type="naive")
    verif_domain = domain.cuda().unsqueeze(0) if gpu else domain.unsqueeze(0)
    intermediate_net.define_linear_approximation(verif_domain, override_numerical_errors=True)
    lbs = intermediate_net.lower_bounds[-1].squeeze(0).cpu()

    candi_tot = diff_out
    # since what we are actually interested in is the minium of gt-cls,
    # we revert all the signs of the last layer
    max_pool_layers = reluified_max_pool(candi_tot, lbs, flip_out_sign=True)

    # simplify linear layers
    simp_required_layers = layers[-1:] + max_pool_layers
    simplified_layers = simplify_network(simp_required_layers)

    final_layers = layers[:-1] + simplified_layers
    return final_layers


def normalize(image, dataset):
    mean, sigma = get_mean_sigma(dataset)
    return (image - mean) / sigma


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 5, stride=2, padding=2),
            ReLU((16*width1, input_size//2, input_size//2)),
            Conv2d(16*width1, 32*width2, 4, stride=2, padding=1),
            ReLU((32*width2, input_size//4, input_size//4)),
            flatten(),
            Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 3, stride=1, padding=1, dim=input_size),
            ReLU((16*width1, input_size, input_size)),
            Conv2d(16*width1, 16*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((16*width2, input_size//2, input_size//2)),
            Conv2d(16*width2, 32*width3, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width3, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

def convmed_colt_to_pytorch(n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
    model = nn.Sequential(
        nn.Conv2d(input_channel, 16*width1, 5, 2, 2, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width1, 32*width2, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        Flatten(),#maybe flatten(),
        nn.Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size, bias=True),
        nn.ReLU(),
        nn.Linear(linear_size, n_class, bias=True)
    )
    return model

def convmedbig_colt_to_pytorch(n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
    model = nn.Sequential(
        nn.Conv2d(input_channel, 16*width1, 3, 1, 1, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width1, 16*width2, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        nn.Conv2d(16*width2, 32*width3, 4, 2, 1, 1, 1, True),
        nn.ReLU(),
        Flatten(),#maybe flatten(),
        nn.Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size, bias=True),
        nn.ReLU(),
        nn.Linear(linear_size, n_class, bias=True)
    )
    return model

def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)
    for name, param in params_src:
        split_name = name.split('.')
        if len(split_name) == 5:
            changed_name = (str(int(split_name[2])-1) + '.' + split_name[4])
            if changed_name in dict_dest:
                dict_dest[changed_name].data.copy_(param.data)

def get_network(dataset, device, net_name, net_loc, input_size, input_channel, n_class):
    if net_name.startswith('convmed_'):
        tokens = net_name.split('_')
        obj = ConvMed
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        linear_size = int(tokens[4])
        net = obj(device, dataset, n_class, input_size, input_channel, width1=width1, width2=width2, linear_size=linear_size)
        net = net.to(device)
        net.load_state_dict(torch.load(net_loc))
        new_net = convmed_colt_to_pytorch(n_class, input_size, input_channel, width1=width1, width2=width2, linear_size=linear_size)
        copyParams(net, new_net)
    elif net_name.startswith('convmedbig_'):
        tokens = net_name.split('_')
        assert tokens[0] == 'convmedbig'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = ConvMedBig(device, dataset, n_class, input_size, input_channel, width1, width2, width3, linear_size=linear_size)
        net = net.to(device)
        net.load_state_dict(torch.load(net_loc))

        new_net = convmedbig_colt_to_pytorch(n_class, input_size, input_channel, width1=width1, width2=width2, width3=width3, linear_size=linear_size)
        copyParams(net, new_net)
    else:
        assert False, 'Unknown network!'

    return new_net


def load_1toall_eth(dataset, model, idx = None, test = None, eps_temp=None, max_solver_batch=10000,
                   no_verif_layers=False):
    device = 'cpu'
    if model=='mnist_0.1':
        net_name = 'convmed_flat_2_2_100'
        net_loc = './models/mnist_0.1_convmed_flat_2_2_100.pt'
        model = get_network(dataset, device, net_name, net_loc, 28, 1, 10)
    elif model=='mnist_0.3':
        net_name = 'convmed_flat_2_4_250'
        net_loc = './models/mnist_0.3_convmed_flat_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 28, 1, 10)
    elif model=='cifar10_8_255':
        net_name = 'convmed_flat_2_4_250'
        net_loc = './models/cifar10_8_255_convmed_flat_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 32, 3, 10)
    elif model=='cifar10_2_255':
        net_name = 'convmedbig_flat_2_2_4_250'
        net_loc = './models/cifar10_2_255_convmedbig_flat_2_2_4_250.pt'
        model = get_network(dataset, device, net_name, net_loc, 32, 3, 10)
    elif 'mnist_convMedGRELU' in model:
        net_name = model
        # net_loc = './models/cifar10_convSmallRELU__Point.pyt'
        model = torch.load('./models/'+ model + '.pkl')
        # model = torch.load('./models/'+ model + '.pkl')
        eps_temp = 0.12
    else:
        raise NotImplementedError

    # print(model)
    # print(idx)
    current_test = test[idx]
    image= np.float64(current_test[1:len(current_test)])/np.float64(255)
    y=np.int(current_test[0])
    if dataset == "cifar_colt":
        # x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 3, 32, 32])).float())
        x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0,3,1,2)), dataset)
    elif dataset == "mnist_colt":
        x = normalize(torch.from_numpy(np.array(image, dtype=np.float32).reshape([1, 1, 28, 28])).float(), dataset)
    # import torchvision.datasets as datasets
    # import torchvision.transforms as transforms
    # normalize_d = transforms.Normalize(mean=[0.1307],
    #                                      std=[0.3081])
    # mnist_test = datasets.MNIST('./data/', train=False, download=True,
    #                               transform=transforms.Compose([transforms.ToTensor(), normalize_d]))
    # x,y = mnist_test[3]
    # x = x.unsqueeze(0)

    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if  y_pred != y: 
        print('model prediction is incorrect for the given model')
        return None, None, None, None
    else:
        # layers = list(model.children())
        # added_prop_layers = add_single_prop(layers, y_pred, test)
        # return x, added_prop_layers, test
        if dataset == "cifar_colt":
            x_m_eps = normalize(torch.from_numpy(np.array(
                image - eps_temp, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0, 3, 1, 2)).clamp(0,1), dataset)
            x_p_eps = normalize(torch.from_numpy(np.array(
                image + eps_temp, dtype=np.float32).reshape([1, 32, 32, 3]).transpose(0, 3, 1, 2)).clamp(0,1), dataset)
        elif dataset == "mnist_colt":
            x_m_eps = normalize(torch.from_numpy(
                np.array(image - eps_temp, dtype=np.float32).reshape([1, 1, 28, 28])).clamp(0,1).float(), dataset)
            x_p_eps = normalize(torch.from_numpy(
                np.array(image + eps_temp, dtype=np.float32).reshape([1, 1, 28, 28])).clamp(0,1).float(), dataset)

        domain = torch.stack([x_m_eps.squeeze(0), x_p_eps.squeeze(0)], dim=-1)

        if no_verif_layers:
            # Avoid computing the network model in canonical form for adversarial robustness, return the original net.
            return x, y, model, domain

        added_prop_layers = one_vs_all_from_model(model, y, domain=domain, max_solver_batch=max_solver_batch)
        for layer in added_prop_layers:
            for p in layer.parameters():
                p.requires_grad = False

        return x, y_pred, added_prop_layers, domain


def load_cifar_model_and_dataset(model, train_or_test='train'):
    if model == 'cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_wide_kw':
        model_name = './models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_deep_kw':
        model_name = './models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    if train_or_test == 'train':
        cifar_test = datasets.CIFAR10('./cifardata/', train=True, download=False,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))
    elif train_or_test == 'test':
        cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=False,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))

    return model, cifar_test
