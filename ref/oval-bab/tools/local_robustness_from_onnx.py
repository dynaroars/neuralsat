import argparse
import time

from tools.bab_tools.model_utils import one_vs_all_from_model
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.bab_tools.bab_runner import bab_from_json, bab_output_from_return_dict

import torch, copy, json

"""
    Example on how to run OVAL BaB for a 1 vs. all local robustness property using .json configuration files.
    
    Supported network structure (pre-layer transforms are optional):
    "pre-layer transforms" (this includes additions, multiplications --this covers normalization--, reshapings, flattenings, etc) -> 
    linear (nn.Linear or nn.Conv2d) -> ReLU() -> "pre-layer transforms" -> linear -> ReLU() -> "pre-layer transforms" -> [linear]
"""

def parse_input():
    # Parse the input specifications: return network, domain, args
    torch.manual_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_filename', type=str, help='onnx file to load.',
                        default="./models/onnx/cifar_base_kw.onnx")
    parser.add_argument('--gpu', action='store_true', help="run BaB on gpu rather than cpu -- will speed up execution")
    parser.add_argument('--cifar_oval', action='store_true',
                        help="test on the first CIFAR10 image, normalized for the networks in ./models/onnx/")
    parser.add_argument('--json', help='OVAL BaB json config file', default="./bab_configs/cifar2020_vnncomp21.json")

    args = parser.parse_args()

    # Test on loaded ONNX net.
    # For instance: ./models/onnx/cifar_base_kw.onnx
    assert args.network_filename.endswith('.onnx')

    model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)

    if not model_correctness:
        raise ValueError(
            "The parsed network does not match the onnx runtime output: please edit tools/bab_tools/onnx_reader.py")

    # Assert that the model specification is currently supported.
    supported = vnnlib_utils.is_supported_model(model)
    assert supported

    if not args.cifar_oval:
        eps = 0.065
        # Define input bounds as being a l_inf ball of eps around a randomly sampled point in [0, 1]. Specified as a
        # stack of [lower bounds, upper bounds] along the last tensor dimension.
        # NOTE: input domains different from l_inf balls (but still representable via element-wise lower/upper bounds)
        # require the modification of this code block
        input_point = torch.rand(in_shape)
        input_bounds = torch.stack([(input_point - eps).clamp(0, 1), (input_point + eps).clamp(0, 1)], dim=-1)
        y = 2
    else:
        # Performs verification on the first CIFAR10 image, assuming the network expects the same input normalization
        # as the networks in ./models/onnx/
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        idx = 0

        # NOTE: the input normalization and format must match the one employed at training time
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
        test_data = datasets.CIFAR10('./cifardata/', train=False, download=True, transform=transforms.ToTensor())
        input_point, y = test_data[idx]
        eps = 2 / 255
        input_bounds = torch.stack([normalizer((input_point - eps).clamp(0, 1)),
                                    normalizer((input_point + eps).clamp(0, 1))], dim=-1)

    # ReLUify any maxpool...
    with torch.no_grad():
        layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype=dtype)

    # Define 1vsall local robustness property in canonical form, assuming the correct class is at index y
    # NOTE: in case of different output specifications, this code block should be replaced by their representation as
    # a series of Linear/Conv2d layers and ReLU activations
    verif_layers = one_vs_all_from_model(
        torch.nn.Sequential(*layers), y, domain=input_bounds, max_solver_batch=1000, use_ib=True, gpu=args.gpu)

    return verif_layers, input_bounds, args


def run_bab_from_json():

    # A network is expressed as list of torch and custom layers (custom layers defined in tools/custom_torch_modules.py)
    layers, domain, args = parse_input()

    # Guide for complete verification. Includes an interface of BaB via .json configuration files.
    timeout = 300  # time out BaB after 300s

    return_dict = dict()
    with open(args.json) as json_file:
        json_params = json.load(json_file)

    # Run OVAL BaB on the (1vsall) local robustness property specified by "layers", which represent a network in
    # canonical form (see https://arxiv.org/abs/2104.06718 section 2.1). The input specification is represented by
    # "domain" (see input_bounds in the parse_input function above)
    bab_from_json(json_params, layers, domain, return_dict, None, instance_timeout=timeout, gpu=args.gpu)
    del json_params
    bab_out, bab_nb_states = bab_output_from_return_dict(return_dict)

    print(f"BaB output state: {bab_out}, number of visited nodes: {bab_nb_states}")


if __name__ == '__main__':

    run_bab_from_json()
