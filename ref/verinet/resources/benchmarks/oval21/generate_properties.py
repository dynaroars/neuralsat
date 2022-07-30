import torch
from src.propagation import Propagation
from src.model_utils import load_cifar_oval_kw_1_vs_all
import time, copy, os, argparse, math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from src.mi_fgsm_attack import MI_FGSM_Attack


def run_root_planet_lb(domain, verif_layers):
    # Run KW/CROWN IB + 10 alpha-CROWN iterations
    # Returns SAT_status, verification time

    start_time = time.time()
    # Run lower bounding
    intermediate_net = Propagation(verif_layers, type="best_prop",
                                   params={"best_among": ["KW", "crown"]})
    with torch.no_grad():
        intermediate_net.define_linear_approximation(domain.unsqueeze(0), no_conv=False)
    prop_params = {
        'nb_steps': 10,
        'initial_step_size': 1e0,
        'step_size_decay': 0.98,
        'betas': (0.9, 0.999),
    }
    prop_net = Propagation(verif_layers, type="alpha-crown", params=prop_params, store_bounds_primal=True)
    with torch.no_grad():
        prop_net.build_model_using_bounds(
            domain.unsqueeze(0), (intermediate_net.lower_bounds, intermediate_net.upper_bounds))
        lb = prop_net.compute_lower_bound(node=(-1, 0))

    # Run upper bounding
    ub = prop_net.net(prop_net.get_lower_bound_network_input())

    if lb >= 0:
        bab_out = "False"
    elif ub < 0:
        # Verify that it is a valid solution
        bab_out = "True"
    else:
        # If the property is still undecided, it's equivalent to timing out in BaB
        bab_out = "timeout"
    end_time = time.time()

    return bab_out, end_time - start_time


def run_quick_attacks(data, model, gpu):
    start_time = time.time()

    # load mi-fgsm model
    adv_params = {
        'iters': 20,
        'optimizer': 'default',
        'lr': 0.1,
        'num_adv_ex': 100,
        'check_adv': 1,
        'mu': 0.5,
        'original_alpha': False,
    }
    adv_model = MI_FGSM_Attack(adv_params, store_loss_progress=True)

    x, y, lb, ub = data

    # perform targeted attacks
    for prop in range(10):
        if prop != y:
            adv_examples, is_adv = adv_model.create_adv_examples(data, model, return_criterion='one',
                                                                 gpu=gpu, target=prop)

            if is_adv.sum() > 0:
                break

    if is_adv.sum() > 0:
        bab_out = "True"
        adv_example = adv_examples[list(is_adv).index(True)]
        assert((adv_example - lb).min() >= 0)
        assert((ub - adv_example).min() >= 0)
    else:
        bab_out = "timeout"

    end_time = time.time()

    return bab_out, end_time - start_time


def binary_eps_search(eps_lower_bound, eps_upper_bound, bab_function, quantization=1e-3, mode="LB"):
    """
    Run binary search on the epsilon values in order to create a BaB dataset.
    :parameter eps_lower_bound: starting lower bound on epsilon
    :parameter eps_upper_bound: starting upper bound on epsilon
    :parameter bab_function: BaB function, takes only epsilon as input
    :parameter quantization: how spaced apart are the epsilons we are considering (default 1e-3).
        The effective quantization is the largest (eps_upper_bound - eps_lower_bound)/2^k > quantization.
        Min quantization/2, max quantization.

    ### Search criterion:
    LB mode: the property with min-eps (within quantization) that is either SAT or has timed out.
    UB mode: the property with max-eps (within quantization) that is either UNSAT or has timed out.

    Returns result, rounded upwards using the quantization.
    """
    assert mode in ["LB", "UB"]
    print(f"Starting epsilon bounds: LB: {eps_lower_bound}, UB: {eps_upper_bound}")

    while (eps_upper_bound - eps_lower_bound) > quantization:
        c_epsilon = (eps_upper_bound + eps_lower_bound) / 2

        # Run BaB with the current epsilon value.
        bab_status, bab_runtime = bab_function(c_epsilon)
        print(f"BaB status {bab_status}, BaB runtime {bab_runtime}")

        conditions = ["True"] if mode == "UB" else ["True", "timeout"]
        if bab_status in conditions:
            eps_upper_bound = c_epsilon
        else:
            eps_lower_bound = c_epsilon

        print(f"Current epsilon bounds: LB: {eps_lower_bound}, UB: {eps_upper_bound}")

    return_value = math.floor(eps_lower_bound / quantization) * quantization if mode == "UB" else \
        math.ceil(eps_upper_bound / quantization) * quantization
    return return_value


def create_oval21(seed, use_cpu):
    """
        Create OVAL21 dataset.
          Start from the entire CIFAR10 testset, sampling random correctly classified images.
          The allowed epsilon range is [0, 16/255], with a tolerance of 0.1/255.
          Find the an epsilon 2/3 through properties verified as UNSAT with Planet bounds (WK/CROWN IB), and
           epsilons that yield SAT properties by running a simple attack.
    """

    datasetcreation_start = time.time()

    # OVAL dataset specs.
    nn_names = ["cifar_base_kw", "cifar_wide_kw", "cifar_deep_kw"]

    # Bounding specs.
    max_solver_batch_list = [45000, 30000, 30000]

    # Employ random seed, fix operation nondeterminism.
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    elif not use_cpu:
        torch.backends.cudnn.deterministic = True
    if not use_cpu:
        torch.backends.cudnn.benchmark = False

    # Sample 10 CIFAR10 images, as per original COLT dataset specification. This way we can use 720 as VNN-COMP timeout
    max_index = int(1e4)
    n_images_per_net = 10

    vnnlib_path = "vnnlib/"
    if not os.path.exists(vnnlib_path):
        os.makedirs(vnnlib_path)

    # load Cifar test set to save time
    cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))

    # Create .onnx nets
    onnx_path = "nets/"
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    onnx_names = [f"{onnx_path}{cname}.onnx" for cname in nn_names]
    inp_example = torch.randn((1, 3, 32, 32))
    for onnx_name, nn_name, max_solv_batch in zip(onnx_names, nn_names, max_solver_batch_list):
        x, y, th_model, _ = load_cifar_oval_kw_1_vs_all(
            nn_name, 0, max_solver_batch=max_solv_batch, no_verif_layers=True, cifar_test=cifar_test, use_cpu=use_cpu)
        pytorch_to_onnx(onnx_name, th_model, inp_example)
    property_dict = {}
    for cname in onnx_names:
        property_dict[cname] = []

    # Sample correctly classified images
    images_indices = []
    for counter, (nn_name, max_solv_batch) in enumerate(zip(nn_names, max_solver_batch_list)):
        images_indices.append([])
        while len(images_indices[counter]) < n_images_per_net:
            tentative_indices = torch.randint(low=0, high=max_index, size=(n_images_per_net,)).tolist()
            for cidx in tentative_indices:
                x, _, _, _ = load_cifar_oval_kw_1_vs_all(
                    nn_name, int(cidx), max_solver_batch=max_solv_batch, no_verif_layers=True, cifar_test=cifar_test,
                    use_cpu=use_cpu)
                if (x is not None) and (cidx not in images_indices[counter]):
                    # Correctly classified image that hasn't already been included.
                    images_indices[counter].append(cidx)
                if len(images_indices[counter]) >= n_images_per_net:
                    break

    for counter, (nn_name, onnx_name, max_solv_batch, img_ids) in enumerate(zip(
            nn_names, onnx_names, max_solver_batch_list, images_indices)):

        for idx in range(n_images_per_net):

            # Load network, and property specifications.
            imag_idx = img_ids[idx]
            prop_idx = None  # 1 vs all
            eps_train = 2 / 255
            print(f'idx_{imag_idx}_prop_{prop_idx}_eps_train_{eps_train}')

            # Set binary search lower and upper bounds, and the BaB function with only epsilon as argument.
            # Range: [0, 16/255], with a tolerance of 0.1/255 -- original [0,1] space
            eps_lower_bound, eps_upper_bound = 0, 16 / 255
            tolerance = 0.1/255

            def lb_from_epsilon(epsilon_value):
                x, _, verif_layers, domain = load_cifar_oval_kw_1_vs_all(
                    nn_name, int(imag_idx), epsilon=epsilon_value, max_solver_batch=max_solv_batch,
                    cifar_test=cifar_test, use_cpu=use_cpu)
                if not use_cpu:
                    # Move to GPU.
                    verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
                    domain = domain.cuda()
                # --------- Run lower and upper bounding.
                bab_out, runtime = run_root_planet_lb(domain, verif_layers)
                return bab_out, runtime

            def ub_from_epsilon(epsilon_value):
                x, y, model, domain = load_cifar_oval_kw_1_vs_all(
                    nn_name, int(imag_idx), epsilon=epsilon_value, max_solver_batch=max_solv_batch,
                    cifar_test=cifar_test, no_verif_layers=True, use_cpu=use_cpu)
                # Move to GPU.
                if not use_cpu:
                    model = model.cuda()
                    x = x.cuda()
                    domain = domain.cuda()
                data = (x.squeeze(), y, domain.select(-1, 0), domain.select(-1, 1))
                # --------- Run attacks
                bab_out, runtime = run_quick_attacks(data, model, not use_cpu)
                return bab_out, runtime

            # Run the binary search for an epsilon LB.
            eps_lb = binary_eps_search(
                eps_lower_bound, eps_upper_bound, lb_from_epsilon, quantization=tolerance, mode="LB")

            # Run the binary search for an epsilon UB.
            eps_ub = binary_eps_search(
                eps_lower_bound, eps_upper_bound, ub_from_epsilon, quantization=tolerance, mode="UB")

            # Shift towards SAT to increase difficulty -- empirically.
            sat_shift = 2/3
            found_epsilon = min(eps_lb, eps_ub) * (1 - sat_shift) + max(eps_lb, eps_ub) * sat_shift

            # Record the results: vnn-lib specification.
            filename = vnnlib_path + f"{nn_name}-img{imag_idx}-eps{found_epsilon}.vnnlib"
            comment = f"Adversarial robustness property for network {nn_name}. l_inf radius: {found_epsilon}, " \
                      f"CIFAR10 test image n. {imag_idx}."
            _, ground_truth, _, domain = load_cifar_oval_kw_1_vs_all(
                nn_name, int(imag_idx), epsilon=found_epsilon, max_solver_batch=max_solv_batch,
                no_verif_layers=True, use_cpu=use_cpu)
            write_adversarial_robustness_vnnlib(filename, comment, domain, ground_truth)
            property_dict[onnx_name].append(filename)

    timeout = 720
    create_benchmark_csv("oval21_instances.csv", property_dict, timeout)

    datasetcreation_end = time.time()
    print(f"The dataset creation process took {datasetcreation_end - datasetcreation_start}")


def write_adversarial_robustness_vnnlib(filename, initial_comment, input_domain, ground_truth, n_classes=10):
    """
    Create a vnnlib [http://www.vnnlib.org/] specification for an adversarial robustness property.
    :param filename: filename for vnnlib property (string)
    :param initial_comment: comment for the first line of the vnnlib file (string)
    :param input_domain: tensor of shape (tensor_shape x 2) specifying the lower and upper bounds of the input domain
    :param ground_truth: correct classification class for the input point (int)
    :param n_classes: number of classes for classification (int)
    """

    with open(filename, "w") as f:
        f.write(f"; {initial_comment}\n")

        # Declare input variables.
        f.write("\n")
        linearized_domain = input_domain.view(-1, 2)
        for i in range(linearized_domain.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        for i in range(n_classes):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(linearized_domain.shape[0]):
            f.write(f"(assert (<= X_{i} {linearized_domain[i, 1]}))\n")  # UB
            f.write(f"(assert (>= X_{i} {linearized_domain[i, 0]}))\n")  # LB
            f.write("\n")
        f.write("\n")

        # Define output constraints, providing an unnecessary "and" to ease parsing in vnn-comp-21.
        f.write(f"; Output constraints (encoding the conditions for a property counter-example):\n")
        f.write(f"(assert (or\n")
        for i in range(n_classes):
            if i != ground_truth:
                f.write(f"\t(and (<= Y_{ground_truth} Y_{i}))\n")
        f.write(f"))\n")
        f.write("\n")


def create_benchmark_csv(csv_filename, property_dict, timeout):
    """
    Creates the benchmark specification's .csv file.
    :param property_dict: a dictionary indexed by onnx net names containing list of vnnlib files for that network
    """
    with open(csv_filename, "w") as f:
        for net in property_dict.keys():
            for prop in property_dict[net]:
                f.write(f"{net},{prop},{timeout}\n")


def pytorch_to_onnx(onnx_filename, model, input_example):
    """
    Exports a PyTorch model into the .onnx format. Requires an input example.
    """
    if not os.path.exists(onnx_filename):
        torch.onnx.export(model, input_example, onnx_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed for the image sampling', default=0)
    parser.add_argument('--cpu', action='store_true', help='Use cpu instead of a gpu (if not available)')
    args = parser.parse_args()

    create_oval21(args.seed, args.cpu)
