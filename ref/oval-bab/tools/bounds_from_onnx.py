import argparse
import time

from plnn.proxlp_solver.propagation import Propagation
from plnn.explp_solver.solver import ExpLP
from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
from plnn.branch_and_bound.branching_scores import BranchingChoice
import tools.bab_tools.vnnlib_utils as vnnlib_utils
from tools.custom_torch_modules import Flatten

import torch, copy

"""

    Contains an introduction to the functionalities of the codebase: how to load an ONNX network, how to compute 
    intermediate and output bounds, how to run a basic branch-and-bound instance without resorting to .json 
    configuration files.

"""


def generate_tiny_random_cnn():
    # Generate a very small CNN with random weight for testing purposes.
    # Input dimensions.
    in_chan = 3
    in_row = 2
    in_col = 2
    #Generate input domain.
    input_domain = torch.zeros((in_chan, in_row, in_col, 2))
    in_lower = (20 - -10) * torch.rand((in_chan, in_row, in_col)) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand((in_chan, in_row, in_col)) + (in_lower + 1)
    input_domain[:, :, :, 0] = in_lower
    input_domain[:, :, :, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_chan_c1 = 7
    ker_size = 2
    conv1 = torch.nn.Conv2d(in_chan, out_chan_c1, ker_size, stride=2, padding=1)
    conv1.weight = torch.nn.Parameter(torch.randn((out_chan_c1, in_chan, ker_size, ker_size)), requires_grad=False)
    conv1.bias = torch.nn.Parameter(torch.randn(out_chan_c1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    ker_size = 2
    out_chan_c2 = 5
    conv2 = torch.nn.Conv2d(out_chan_c1, out_chan_c2, ker_size, stride=5, padding=0)
    conv2.weight = torch.nn.Parameter(torch.randn((out_chan_c2, out_chan_c1, ker_size, ker_size)), requires_grad=False)
    conv2.bias = torch.nn.Parameter(torch.randn(out_chan_c2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_chan_c2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_chan_c2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)
    layers = [conv1, relu1, conv2, relu2, Flatten(), final]

    return layers, input_domain


def generate_tiny_random_linear(precision):
    # Generate a very small fully connected network with random weight for testing purposes.
    # Input dimensions.
    input_size = 2
    #Generate input domain.
    input_domain = torch.zeros((input_size, 2))
    in_lower = (20 - -10) * torch.rand(input_size) + -10
    in_upper = (50 - (in_lower + 1)) * torch.rand(input_size) + (in_lower + 1)
    input_domain[:, 0] = in_lower
    input_domain[:, 1] = in_upper

    # Generate layers: 2 convolutional (followed by one ReLU each), one final linear.
    out_size1 = 3
    lin1 = torch.nn.Linear(input_size, out_size1)
    lin1.weight = torch.nn.Parameter(torch.randn((out_size1, input_size)), requires_grad=False)
    lin1.bias = torch.nn.Parameter(torch.randn(out_size1), requires_grad=False)
    relu1 = torch.nn.ReLU()
    out_size2 = 3
    lin2 = torch.nn.Linear(out_size1, out_size2)
    lin2.weight = torch.nn.Parameter(torch.randn((out_size2, out_size1)), requires_grad=False)
    lin2.bias = torch.nn.Parameter(torch.randn(out_size2), requires_grad=False)
    relu2 = torch.nn.ReLU()
    final = torch.nn.Linear(out_size2, 1)
    final.weight = torch.nn.Parameter(torch.randn((1, out_size2)), requires_grad=False)
    final.bias = torch.nn.Parameter(torch.randn(1), requires_grad=False)

    input_domain = (input_domain).type(precision)
    lin1.weight = torch.nn.Parameter(lin1.weight.type(precision))
    lin1.bias = torch.nn.Parameter(lin1.bias.type(precision))
    lin2.weight = torch.nn.Parameter(lin2.weight.type(precision))
    lin2.bias = torch.nn.Parameter(lin2.bias.type(precision))
    final.weight = torch.nn.Parameter(final.weight.type(precision))
    final.bias = torch.nn.Parameter(final.bias.type(precision))

    layers = [lin1, relu1, lin2, relu2, final]

    return layers, input_domain


def parse_input(precision=torch.float):
    # Parse the input specifications: return network, domain, args

    torch.manual_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_filename', type=str, help='onnx file to load.')
    parser.add_argument('--random_net', type=str, choices=["cnn", "linear"], help='whether to use a random network')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--init_step', type=float, default=1e0, help="step size for optimization based algos")

    args = parser.parse_args()

    if args.random_net and args.network_filename:
        raise IOError("Test either on a random network or on a .rlv, not both.")

    if args.network_filename:
        # Test on loaded ONNX net.
        # For instance: ./models/onnx/cifar_base_kw.onnx
        assert args.network_filename.endswith('.onnx')

        model, in_shape, out_shape, dtype, model_correctness = vnnlib_utils.onnx_to_pytorch(args.network_filename)

        if not model_correctness:
            return None, False

        # Assert that the model specification is currently supported.
        supported = vnnlib_utils.is_supported_model(model)
        assert supported

        eps = 0.1
        # Define input bounds as being a l_inf ball of eps around a randomly sampled point in [0, 1]
        input_point = torch.rand(in_shape)
        input_bounds = torch.stack([(input_point - eps).clamp(0, 1), (input_point + eps).clamp(0, 1)], dim=-1)

        # ReLUify any maxpool...
        with torch.no_grad():
            layers = vnnlib_utils.remove_maxpools(copy.deepcopy(list(model.children())), input_bounds, dtype=dtype)

    else:
        if args.random_net == "cnn":
            # Test on small generated CNN.
            layers, input_bounds = generate_tiny_random_cnn()
        else:
            # Test on small generated fully connected network.
            layers, input_bounds = generate_tiny_random_linear(precision)

    return layers, input_bounds, args


def compute_bounds():

    precision = torch.float

    # A network is expressed as list of torch and custom layers (custom layers defined in tools/custom_torch_modules.py)
    layers, domain, args = parse_input(precision=precision)

    # make the input domain a batch of domains -- in this case of size 1
    batch_domain = domain.unsqueeze(0)

    gpu = args.gpu
    if gpu:
        # the copy is necessary as .cuda() acts in place for nn.Parameter
        exp_layers = [copy.deepcopy(lay).cuda() for lay in layers]
        exp_domain = batch_domain.cuda()
    else:
        exp_layers = layers
        exp_domain = batch_domain

    # Given the network, the l_inf input domain specified as a stack (over the last dimension) of lower and upper
    # bounds over the input, the various bounding classes defined in ./plnn/ create an internal representation of the
    # network with the classes defined in plnn.proxlp_solver.utils

    # The internal representation is internally built in plnn.dual_bounding, which defines the DualBounding that all
    # bounding methods inherit from. In particular, the conversion occurs within define_linear_approximation or
    # within build_model_using_bounds

    # Compute intermediate bounds using best bounds between CROWN and KW
    intermediate_net = Propagation(exp_layers, type="best_prop", params={"best_among": ["KW", "crown"]}, max_batch=2000)
    # intermediate_net = Propagation(exp_layers, type="naive", max_batch=2000)  # uses IBP bounds -- much looser
    with torch.no_grad():
        intermediate_net.define_linear_approximation(exp_domain)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds

    # Given the intermediate bounds, compute CROWN bounds
    prop_net = Propagation(exp_layers, type='crown')
    prop_start = time.time()
    with torch.no_grad():
        prop_net.build_model_using_bounds(exp_domain, (intermediate_lbs, intermediate_ubs))
        lb, ub = prop_net.compute_lower_bound()  # computes both lower and upper bounds over the net output
        # lb = prop_net.compute_lower_bound(node=(-1, 0))  # computes only the lower bound of the first output neuron
    prop_end = time.time()
    prop_lbs = lb.cpu()
    prop_ubs = ub.cpu()
    print(f"CROWN Time: {prop_end - prop_start}")
    print(f"CROWN LB: {prop_lbs}")
    print(f"CROWN UB: {prop_ubs}")

    # Alpha-CROWN bounds (beta-crown without branch-and-bound): at optimality they correspond to Planet
    prop_params = {
        'nb_steps': 5,
        'initial_step_size': args.init_step,
        'step_size_decay': 0.98,
        'betas': (0.9, 0.999),
    }
    prop_net = Propagation(exp_layers, type='alpha-crown', params=prop_params)
    prop_start = time.time()
    with torch.no_grad():
        prop_net.build_model_using_bounds(exp_domain, (intermediate_lbs, intermediate_ubs))
        lb, ub = prop_net.compute_lower_bound()
    prop_end = time.time()
    prop_lbs = lb.cpu()
    prop_ubs = ub.cpu()
    print(f"alpha-CROWN Time: {prop_end - prop_start}")
    print(f"alpha-CROWN LB: {prop_lbs}")
    print(f"alpha-CROWN UB: {prop_ubs}")

    # Get tighter bounds from https://openreview.net/forum?id=uQfOy7LrlTR
    explp_params = {
        "nb_iter": 100,
        'bigm': "init",
        'cut': "only",
        "bigm_algorithm": "adam",
        'cut_frequency': 450,
        'max_cuts': 8,
        'cut_add': 2,
        'betas': (0.9, 0.999),
        "initial_step_size": 1e-3,
        "final_step_size": 1e-6,
        "init_params": {
            "nb_outer_iter": 500,
            "initial_step_size": 1e-3,
            "final_step_size": 1e-4,
            "betas": [0.9, 0.999],
            "larger_irl_if_naive_init": True
        },
        'restrict_factor': 1.5
    }
    activeset_net = ExpLP(layers, params=explp_params, fixed_M=True, store_bounds_primal=True)
    prop_start = time.time()
    with torch.no_grad():
        activeset_net.build_model_using_bounds(exp_domain, (intermediate_lbs, intermediate_ubs))
        lb, ub = activeset_net.compute_lower_bound()
    prop_end = time.time()
    prop_lbs = lb.cpu()
    prop_ubs = ub.cpu()
    print(f"Active Set Time: {prop_end - prop_start}")
    print(f"Active Set LB: {prop_lbs}")
    print(f"Active Set UB: {prop_ubs}")

    # Run branch and bound with a timeout of 10s to get tighter bounds on the first output neuron
    if lb.shape[-1] != 1:
        # add an extra layer to make sure the output is scalar (will only select the first output)
        # NOTE: refer to local_robustness_from_onnx.py regarding how to create a local 1-vs-all robustness property
        # in canonical form
        final = torch.nn.Linear(lb.shape[-1], 1)
        weights = torch.zeros_like(lb)
        weights[:, 0] = 1
        final.weight = torch.nn.Parameter(weights, requires_grad=False)
        final.bias = torch.nn.Parameter(torch.zeros(1).to(lb.device), requires_grad=False)
        layers.append(final)

    timeout = 10  # time out BaB after 10s
    # Intermediate bounding specifications
    intermediate_dict = {
        'loose_ib': {"net": intermediate_net},
        'tight_ib': {"net": None},
        'fixed_ib': True,  # whether to keep IBs fixed throughout BaB (after root)
        'joint_ib': False,
    }
    # Output bounding specification
    out_bounds_dict = {
        # Net to use for the output bounding
        'nets': [{
            "net": Propagation(exp_layers, type='beta-crown', params=prop_params, store_bounds_primal=True),
            "batch_size": 10,
            "auto_iters": True
        }],
        'do_ubs': False,  # whether to compute UBs (can catch more infeasible domains)
        'parent_init': True,  # whether to initialize the dual variables from parent
    }
    # Branching strategy specification
    branching_dict = {
        'choice': "heuristic",
        'heuristic_type': "FSB",  # branching strategy from https://arxiv.org/abs/2104.06718
        'hscoremin': True,
        'bounding': {"net": Propagation(
            layers, type="best_prop", params={"best_among": ["KW", "crown"]}, max_batch=1000)},
        'gnn_name': None, "max_domains": 50  # larger than the batch size, but not too much
    }
    bab_start = time.time()
    brancher = BranchingChoice(branching_dict, layers)
    decision_threshold = 0.  # assumes the decision threshold on the property is 0
    min_lb, min_ub, _, nb_states = relu_bab(
        intermediate_dict, out_bounds_dict, brancher, domain, decision_threshold, timeout=timeout,
        return_bounds_if_timeout=True)
    bab_end = time.time()
    print(f"BaB Time: {bab_end - bab_start} -- n_branches {nb_states}")
    print(f"BaB LB: {min_lb.cpu()}")
    print(f"BaB UB: {min_ub.cpu()}")


if __name__ == '__main__':

    compute_bounds()
