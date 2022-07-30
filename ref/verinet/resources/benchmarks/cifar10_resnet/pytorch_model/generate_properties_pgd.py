############################################################
#    CIFAR10-ResNet benchmark (for VNN Comp 2021)          #
#                                                          #
# Copyright (C) 2021  Shiqi Wang (sw3215@columbia.edu)     #
# Copyright (C) 2021  Huan Zhang (huan@huan-zhang.com)     #
# Copyright (C) 2021  Kaidi Xu (xu.kaid@northeastern.edu)  #
#                                                          #
# This program is licenced under the BSD 2-Clause License  #
############################################################

import os
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from resnet import resnet2b, resnet4b
from attack_pgd import attack_pgd

cifar10_mean = (0.4914, 0.4822, 0.4465)  # np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # np.std(train_set.train_data, axis=(0,1,2))/255


def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = False) -> tuple:

    """
    Loads the cifar10 data.

    Args:
        data_dir:
            The directory to store the full CIFAR10 dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))


# noinspection PyShadowingNames
def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: tuple = (0.4914, 0.4822, 0.4465),
                        std: tuple = (0.2471, 0.2435, 0.2616)) -> torch.Tensor:

    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
        mean:
            The channel-wise means.
        std:
            The channel-wise standard deviation.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """

    mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.tensor(std, device=img.device).view(-1, 1, 1)

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clip((img - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((img + eps), 0, 1) - mean) / std
    # print(bounds[..., 0].abs().sum(), bounds[..., 1].abs().sum())

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; CIFAR10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        # orignal separate version:
        # for i in range(total_output_class):
        #     if i != label:
        #         f.write(f"(assert (>= Y_{label} Y_{i}))\n")
        # f.write("\n")

        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")

def create_csv():
    name = ["model_name", "property_name", "timeout"]
    instance_list = []

    # 48 properties for resnet2b
    model_name = "resnet_2b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    assert os.path.exists("../vnnlib_properties_pgd_filtered/")
    for i in range(48):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet2b_pgd_filtered/prop_{i}_eps_0.008.vnnlib", "300"])

    # 24 properties for resnet2b
    model_name = "resnet_4b"
    assert os.path.exists(f"../onnx/{model_name}.onnx")
    for i in range(24):
        instance_list.append([f"onnx/{model_name}.onnx", f"vnnlib_properties_pgd_filtered/resnet4b_pgd_filtered/prop_{i}_eps_0.004.vnnlib", "300"])

    with open('../cifar10_resnet_instances.csv', 'w') as f:
        write = csv.writer(f)
        # write.writerow(fields)
        write.writerows(instance_list)


def create_vnnlib(args):
    num_imgs = args.num_images
    print(f"===== model: {args.model} epsilons: {args.epsilons} total images: {args.num_images} =====")
    print("deterministic", args.deterministic, "seed:", args.seed)
    epsilons = [eval(eps) for eps in args.epsilons.split(" ")]

    result_dir = "../vnnlib_properties_pgd_filtered/"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    result_dir = os.path.join("../vnnlib_properties_pgd_filtered/", args.model+"_pgd_filtered/")
    model_path = os.path.join(args.model, "model_best.pth")

    print("loading model {} and properties saved in {}".format(model_path, result_dir))

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    mu = torch.tensor(cifar10_mean).view(3,1,1)
    std = torch.tensor(cifar10_std).view(3,1,1)

    model = eval(args.model)()
    model.load_state_dict(torch.load(model_path, map_location='cpu')["state_dict"])
    if args.device == 'gpu':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.cuda()
        mu = mu.cuda()
        std = std.cuda()
        
    normalize = lambda X: (X - mu)/std

    if args.seed is not None:
        if args.device == 'gpu':
            torch.cuda.manual_seed_all(args.seed)
        torch.random.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    images, labels = load_data(num_imgs=10000, random=not args.deterministic)

    for eps in epsilons:
        acc, pgd_acc = 0, 0
        cnt = 0
        for i in range(images.shape[0]):
            if cnt>=num_imgs:
                break

            # Load image and label.
            image, label = images[i], labels[i]
            image = image.unsqueeze(0)
            y = torch.tensor([label], dtype=torch.int64)
            if args.device == 'gpu':
                image = image.cuda()
                y = y.cuda()

            output = model(normalize(image))
            # Skip incorrect examples. 
            if output.max(1)[1] != label: 
                print("incorrect image {}".format(i))
                continue

            acc += 1
            # Skip attacked examples.
            perturbation = attack_pgd(model, X=image, y=y, epsilon=eps, alpha=eps / 2.0,
                    attack_iters=100, num_restarts=5, upper_limit=1.0, lower_limit=0.0, normalize=normalize)

            attack_image = image + perturbation
            assert (attack_image >= 0.).all()
            assert (attack_image <= 1.).all()
            assert perturbation.abs().max() <= eps
            attack_output = model(normalize((image + perturbation))).squeeze(0)
            attack_label = attack_output.argmax()

            if attack_label != label:
                print("pgd succeed image {}, label {}, against label {}".format(i, label, attack_label))
                continue

            pgd_acc += 1

            print("scanned images: {}, selected: {}, label {}".format(i, cnt, label))

            input_bounds = create_input_bounds(image, eps)
            spec_path = os.path.join(result_dir, f"prop_{cnt}_eps_{eps:.3f}.vnnlib")
            save_vnnlib(input_bounds, label, spec_path)
            cnt += 1

    print("acc:", acc, "pgd_acc:", pgd_acc, "out of", i, "samples")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default="resnet2b", choices=["resnet2b", "resnet4b"])
    # parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--deterministic', action='store_true', help='Do not generate random examples; use dataset order instead.')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Choose device to generate adversarial examples.')
    # parser.add_argument('--epsilons', type=str, default="2/255")
    args = parser.parse_args()

    # Example: $python generate_properties_pgd.py --num_images 100 --random True --epsilons '2/255' --seed 0

    args.model = "resnet2b"
    args.epsilons = "2/255"
    args.num_images = 48
    create_vnnlib(args)

    args.model = "resnet4b"
    args.epsilons = "1/255"
    args.num_images = 24
    create_vnnlib(args)

    create_csv()


