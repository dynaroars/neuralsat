import torch
import copy
from torch import nn
from plnn_bounds.model import cifar_model, cifar_model_large


def load_network(filename):
    dump = torch.load(filename)
    state_dict = dump['state_dict'][0]
    if len(state_dict) == 8:
        model = cifar_model()
    elif len(state_dict) == 14:
        model = cifar_model_large()
    else:
        raise NotImplementedError
    # [0] because it's the dumb cascade training, which we don't deal with
    model.load_state_dict(state_dict)
    return model


def make_elided_models(model, return_error=False):
    """
    Default is to return GT - other
    Set `return_error` to True to get instead something that returns a loss
    (other - GT)

    mono_output=False is an argument I removed
    """
    elided_models = []
    layers = [lay for lay in model]
    assert isinstance(layers[-1], nn.Linear)

    net = layers[:-1]
    last_layer = layers[-1]
    nb_classes = last_layer.out_features

    for gt in range(nb_classes):
        new_layer = nn.Linear(last_layer.in_features,
                              last_layer.out_features-1)

        wrong_weights = last_layer.weight[[f for f in range(last_layer.out_features) if f != gt], :]
        wrong_biases = last_layer.bias[[f for f in range(last_layer.out_features) if f != gt]]

        if return_error:
            new_layer.weight.data.copy_(wrong_weights - last_layer.weight[gt])
            new_layer.bias.data.copy_(wrong_biases - last_layer.bias[gt])
        else:
            new_layer.weight.data.copy_(last_layer.weight[gt] - wrong_weights)
            new_layer.bias.data.copy_(last_layer.bias[gt] - wrong_biases)

        layers = copy.deepcopy(net) + [new_layer]
        # if mono_output and new_layer.out_features != 1:
        #     layers.append(View((1, new_layer.out_features)))
        #     layers.append(nn.MaxPool1d(new_layer.out_features,
        #                                stride=1))
        #     layers.append(View((1,)))
        new_elided_model = nn.Sequential(*layers)
        elided_models.append(new_elided_model)
    return elided_models


def cifar_loaders(batch_size):
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    return train_loader, test_loader


def dump_bounds(target_file, time, upper_bounds, to_ignore=None):
    bounds_list = upper_bounds.squeeze().numpy().tolist()
    if to_ignore is not None:
        # There is one of the optimization that is unnecessary: the one with
        # robustness to the ground truth.
        del bounds_list[to_ignore]
    bound_str = "\t".join(map(str, bounds_list))
    with open(target_file, 'w') as res_file:
        res_file.write(f"{time}\n{bound_str}\n")
