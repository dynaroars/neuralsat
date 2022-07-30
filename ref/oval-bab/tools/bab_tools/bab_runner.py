import argparse
import torch
from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
import plnn.branch_and_bound.utils as bab_utils
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_1toall_eth, \
    load_cifar_oval_kw_1_vs_all
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.proxlp_solver.propagation import Propagation
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from plnn.explp_solver.solver import ExpLP
from plnn.branch_and_bound.branching_scores import BranchingChoice
from adv_exp.mi_fgsm_attack_canonical_form import MI_FGSM_Attack_CAN
import time
import pandas as pd
import os, copy
import math
import torch.multiprocessing as mp
import csv
import gc
import json

# Pre-fixed parameters
gpu = True
decision_bound = 0


def bab(verif_layers, domain, return_dict, timeout, batch_size, method, tot_iter,  parent_init, args):
    epsilon = 1e-4
    gurobi_dict = {"gurobi": args.method in ["gurobi", "gurobi-anderson"], "p": args.gurobi_p}

    if gpu:
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # Handle intermediate bounds.
    if args.tighter_ib:
        # use a few alpha-crown iterations as IBs
        prop_params = {
            'nb_steps': 5,
            'initial_step_size': args.init_step,
            'step_size_decay': args.step_decay,
            'betas': (0.9, 0.999),
        }
        intermediate_net = Propagation(cuda_verif_layers, type="alpha-crown", params=prop_params,
                                       max_batch=args.max_solver_batch)
    else:
        # use best of CROWN and KW as intermediate bounds
        intermediate_net = Propagation(cuda_verif_layers, type="best_prop", params={"best_among": ["KW", "crown"]},
                                       max_batch=args.max_solver_batch)

    # Explicitly optimise over selected intermediate bounds  --  not worth its cost, usually.
    optib_net = None
    if args.opt_ib:
        optib_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        optib_net.set_decomposition('pairs', 'crown')
        optprox_params = {
            'nb_total_steps': 100,
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'initial_eta': args.eta,
            'final_eta': args.feta
        }
        optib_net.set_solution_optimizer('optimized_prox', optprox_params)
    intermediate_dict = {
        'loose_ib': {"net": intermediate_net},
        'tight_ib': {"net": optib_net},  # a tighter network to compute selected intermediate bounds with
        'k': args.opt_ib_k,  # how many neurons from the layer before the last to use the tight_net on
        'fixed_ib': args.fixed_ib,  # whether to keep IBs fixed throughout BaB (after root)
        'joint_ib': args.joint_ib,  # whether to optimize jointly over IBs and LBs
    }

    tighter_bounds_net = None
    # Split domains into easy and hard, define two separate bounding methods to handle their last layer.
    if method in ["sp-fw", "cut", "cut-sp-fw", "gurobi-anderson"]:

        # Set bounds net for easy domains.
        if method in ["sp-fw", "cut", "cut-sp-fw"]:
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": int(tot_iter),  # cifar_oval: 180
                'initial_step_size': args.dualinit_init_step,  # cifar_oval: 1e-2
                'final_step_size': args.dualinit_fin_step,  # cifar_oval: 1e-4
                'betas': (0.9, 0.999)
            }
            bounds_net = ExpLP(cuda_verif_layers, params=bigm_adam_params, store_bounds_primal=True)
        else:
            bounds_net = LinearizedNetwork(verif_layers)

        # Set bounds net for hard domains.
        if method in ["sp-fw", "cut-sp-fw"]:
            spfw_iter = args.hard_iter  # 1000 or 2000
            explp_params = {
                "anderson_algorithm": "saddle",
                "nb_iter": spfw_iter,
                'blockwise': False,
                "step_size_dict": {
                    'type': 'fw',
                    'fw_start': args.fw_start
                },
                "init_params": {
                    "nb_outer_iter": 500,
                    'initial_step_size': args.dualinit_init_step,
                    'final_step_size': args.dualinit_fin_step,
                    'betas': (0.9, 0.999),
                    'M_factor': 1.0
                },
                # With parent initialisation, this is called only at the root of the harder subtrees.
                "primal_init_params": {
                    'nb_bigm_iter': args.dualinit_iters,
                    'nb_anderson_iter': 100,
                    'initial_step_size': args.primalinit_init_step,
                    'final_step_size': args.primalinit_fin_step,
                    'betas': (0.9, 0.999)
                }
            }
            if method == "sp-fw":
                explp_params.update({"bigm": "init", "bigm_algorithm": "adam"})
                print(f"Running SP-FW with {spfw_iter} steps")
            else:
                cut_init_params = {
                    'cut_frequency': 450,
                    'max_cuts': 8,
                    'cut_add': args.cut_add,
                    'nb_iter': args.fw_cut_iters,
                    'initial_step_size': args.init_step,
                    'final_step_size': args.fin_step,
                }
                explp_params.update({"cut": "init", "cut_init_params": cut_init_params})
                print(f"Running AS + SP-FW with ({args.fw_cut_iters}, {spfw_iter}) steps")
            tighter_bounds_net = ExpLP(cuda_verif_layers, params=explp_params, fixed_M=True, store_bounds_primal=True)
        elif method == "cut":
            anderson_iter = args.hard_iter  # 100
            explp_params = {
                "nb_iter": anderson_iter,
                'bigm': "init",
                'cut': "only",
                "bigm_algorithm": "adam",
                'cut_frequency': 450,
                'max_cuts': 8,
                'cut_add': args.cut_add,  # 2
                'betas': (0.9, 0.999),
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                "init_params": {
                    "nb_outer_iter": args.dualinit_iters,
                    'initial_step_size': args.dualinit_init_step,
                    'final_step_size': args.dualinit_fin_step,
                    'betas': (0.9, 0.999),
                },
                'restrict_factor': args.restrict_factor,
            }
            tighter_bounds_net = ExpLP(cuda_verif_layers, params=explp_params, fixed_M=True, store_bounds_primal=True)
            print(f"Running cut for {anderson_iter} iterations")
        elif method == "gurobi-anderson":
            tighter_bounds_net = AndersonLinearizedNetwork(
                verif_layers, mode="lp-cut", n_cuts=args.n_cuts, cuts_per_neuron=True, decision_boundary=decision_bound)

        if args.no_easy:
            # Ignore the easy problems bounding, use the hard one for all.
            bounds_net = tighter_bounds_net
            tighter_bounds_net = None

    # Use only a single last layer bounding method for all problems.
    else:
        if method == "prox":
            bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
            bounds_net.set_decomposition('pairs', 'crown')
            optprox_params = {
                'nb_total_steps': int(tot_iter),
                'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
                'initial_eta': args.eta,
                'final_eta': args.feta,
                'log_values': False,
                'maintain_primal': True
            }
            bounds_net.set_solution_optimizer('optimized_prox', optprox_params)
            print(f"Running prox with {tot_iter} steps")
        elif method == "adam":
            bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
            bounds_net.set_decomposition('pairs', 'crown')
            adam_params = {
                'nb_steps': int(tot_iter),
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'step_decay_rate': args.step_decay,
                'betas': (0.9, 0.999),
                'log_values': False
            }
            bounds_net.set_solution_optimizer('adam', adam_params)
            print(f"Running adam with {tot_iter} steps")
        elif method == "dj-adam":
            adam_params = {
                'nb_steps': int(tot_iter),
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (0.9, 0.999),
                'log_values': False,
                'init': 'crown'
            }
            bounds_net = DJRelaxationLP(cuda_verif_layers, params=adam_params, store_bounds_primal=True,
                                        max_batch=args.max_solver_batch)
            print(f"Running DJ-adam with {tot_iter} steps")
        elif method == "bigm-adam":
            bigm_adam_params = {
                "bigm_algorithm": "adam",
                "bigm": "only",
                "nb_outer_iter": int(tot_iter),
                'initial_step_size': args.init_step,
                'final_step_size': args.fin_step,
                'betas': (0.9, 0.999)
            }
            bounds_net = ExpLP(cuda_verif_layers, params=bigm_adam_params, store_bounds_primal=True)
        elif method == "crown":
            bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
            bounds_net.set_solution_optimizer('best_inits', ["KW", "crown"])
            print(f"Running crown")
        elif method in ["alpha-crown", "gamma-crown", "beta-crown"]:
            # 1e0, 0.98 decay, 5 iters works very well for alpha-crown on cifar10 (and KW nets of 6k neurons)
            prop_params = {
                'nb_steps': int(tot_iter),
                'initial_step_size': args.init_step,
                'step_size_decay': args.step_decay,
                'betas': (0.9, 0.999),
                'joint_ib': args.joint_ib
            }
            bounds_net = Propagation(cuda_verif_layers, type=method, params=prop_params, store_bounds_primal=True,
                                     max_batch=args.max_solver_batch)
        elif method == "gurobi":
            bounds_net = LinearizedNetwork(verif_layers)

    # Store last layer bounding info into a dictionary.
    out_bounds_dict = {
        'nets': [{
            "net": bounds_net,
            "batch_size": batch_size,
            "auto_iters": tot_iter == -1 and not gurobi_dict["gurobi"]
        }],
        'do_ubs': args.do_ubs,  # whether to compute UBs (can catch more infeasible domains)
        'parent_init': parent_init,  # whether to initialize the dual variables from parent
    }
    if tighter_bounds_net is not None:
        out_bounds_dict["nets"].append({
            "net": tighter_bounds_net,
            "batch_size": batch_size if args.hard_batch_size == -1 else args.hard_batch_size,
            "auto_iters": args.hard_iter == -1 and not gurobi_dict["gurobi"],
            "hard_overhead": args.hard_overhead,  # assumed at full batch
        })

    # branching
    branching_dict = {
        'heuristic_type': args.bheuristic,  # "FSB"
        "bounding": None,
        'max_domains': args.max_domains,  # max number of domains at once for FSB heuristic
    }
    if args.bheuristic == "FSB":
        branching_net = Propagation(cuda_verif_layers, type="best_prop", params={"best_among": ["KW", "crown"]},
                                    max_batch=args.max_solver_batch)

        branching_dict["bounding"] = {"net": branching_net}
    brancher = BranchingChoice(branching_dict, cuda_verif_layers)

    with torch.no_grad():
        min_lb, min_ub, ub_point, nb_states = relu_bab(
            intermediate_dict, out_bounds_dict, brancher, domain, decision_bound, eps=epsilon,
            timeout=timeout, gurobi_dict=gurobi_dict, max_cpu_subdomains=args.max_cpu_subdomains)

    if not (min_lb or min_ub or ub_point):
        return_dict["min_lb"] = None;
        return_dict["min_ub"] = None;
        return_dict["ub_point"] = None;
        return_dict["nb_states"] = nb_states
        return_dict["bab_out"] = "timeout"
    else:
        return_dict["min_lb"] = min_lb.cpu()
        return_dict["min_ub"] = min_ub.cpu()
        return_dict["ub_point"] = ub_point.cpu()
        return_dict["nb_states"] = nb_states


def parse_bounding_algorithms(param_dict, cuda_verif_layers, nn_name):
    # Given a parameter dict, replace the entries corresponding to bounding algorithms by the respective class instance

    # Defines the expected structure of a dictionary defining the bounding algorithms to be employed.
    def parse_algo_in_dict(cdict):
        if "bounding_algorithm" in list(cdict.keys()):
            method = cdict.pop("bounding_algorithm")

            # Extract the correct values of network-dependent parameters.
            if "max_solver_batch" in cdict:
                max_solver_batch = cdict.pop("max_solver_batch")
                if isinstance(max_solver_batch, dict):
                    max_solver_batch = max_solver_batch[nn_name]
            else:
                max_solver_batch = int(2e4)
            if "batch_size" in cdict:
                cdict["batch_size"] = cdict["batch_size"]
                if isinstance(cdict["batch_size"], dict):
                    cdict["batch_size"] = cdict["batch_size"][nn_name]

            if method == "prox":
                net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=max_solver_batch)
                net.set_decomposition('pairs', 'crown')
                net.set_solution_optimizer('optimized_prox', cdict.pop("params"))
            elif method == "propagation":
                net = Propagation(cuda_verif_layers, type=cdict.pop("type"),
                                  params=cdict.pop("params"), store_bounds_primal=True,
                                  max_batch=max_solver_batch)
            elif method == "dual-anderson":
                net = ExpLP(cuda_verif_layers, params=cdict.pop("params"), fixed_M=True, store_bounds_primal=True)
            else:
                raise IOError(f"Bounding algorithm {method} not supported by bab_from_json")
            cdict["net"] = net

    # Look for bounding algorithms in the subdicts of the root dict (can also be stored as lists)
    for root_key in param_dict:
        for key in param_dict[root_key]:
            if isinstance(param_dict[root_key][key], dict):
                parse_algo_in_dict(param_dict[root_key][key])
            elif isinstance(param_dict[root_key][key], list):
                for list_entry in param_dict[root_key][key]:
                    if isinstance(list_entry, dict):
                        parse_algo_in_dict(list_entry)


def bab_from_json(json_params, verif_layers, domain, return_dict, nn_name, instance_timeout=None,
                  gpu=True, decision_bound=0, start_time=None):

    # Pass the parameters for the BaB code via a .json file, rather than through command line arguments.
    epsilon = 1e-4

    if gpu:
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # TODO: missing json support for gurobi, which also requires a gurobi_dict to be passed -- use CL arguments for now
    timeout = json_params["bab"]["timeout"] if instance_timeout is None else instance_timeout
    # Convert dictionaries specifying bounding algorithms into the corresponding class instances
    parse_bounding_algorithms(json_params, cuda_verif_layers, nn_name)
    max_cpu_domains = json_params["bab"]["max_cpu_subdomains"] \
        if "bab" in json_params and "max_cpu_subdomains" in json_params["bab"] else None
    early_terminate = json_params["bab"]["early_terminate"] \
        if "bab" in json_params and "early_terminate" in json_params["bab"] else False
    intermediate_dict = json_params["ibs"]
    out_bounds_dict = json_params["bounding"]
    branching_dict = json_params["branching"]
    if isinstance(branching_dict["max_domains"], dict):
        branching_dict["max_domains"] = branching_dict.pop("max_domains")[nn_name]  # max_domains varies by network
    brancher = BranchingChoice(branching_dict, cuda_verif_layers)

    # upper bounding
    if "upper_bounding" in json_params:
        data = (domain.select(-1, 0), domain.select(-1, 1))
        if json_params["upper_bounding"]["ub_method"] == 'mi_fgsm':
            adv_model = MI_FGSM_Attack_CAN(json_params["upper_bounding"]["adv_params"], model_can=cuda_verif_layers, data=data)
        else:
            raise NotImplementedError
    else:
        adv_model = None

    with torch.no_grad():
        min_lb, min_ub, ub_point, nb_states = relu_bab(
            intermediate_dict, out_bounds_dict, brancher, domain, decision_bound, eps=epsilon, ub_method=adv_model,
            timeout=timeout, max_cpu_subdomains=max_cpu_domains, start_time=start_time, early_terminate=early_terminate)

    if not (min_lb or min_ub or ub_point):
        return_dict["min_lb"] = None;
        return_dict["min_ub"] = None;
        return_dict["ub_point"] = None;
        return_dict["nb_states"] = nb_states
        return_dict["bab_out"] = "timeout" if nb_states != -2 else "babError"
    else:
        return_dict["min_lb"] = min_lb.cpu()
        return_dict["min_ub"] = min_ub.cpu()
        return_dict["ub_point"] = ub_point.cpu()
        return_dict["nb_states"] = nb_states


def bab_output_from_return_dict(return_dict):
    bab_min_lb = return_dict["min_lb"]
    bab_min_ub = return_dict["min_ub"]
    bab_nb_states = return_dict["nb_states"]
    if bab_min_lb is None:
        if "bab_out" in return_dict:
            bab_out = return_dict["bab_out"]
        else:
            bab_out = 'babError'
    else:
        if bab_min_lb >= 0:
            print("UNSAT")
            bab_out = "False"
        elif bab_min_ub < 0:
            # Verify that it is a valid solution
            print("SAT")
            bab_out = "True"
        else:
            print("Unknown")
            bab_out = 'ET'
    return bab_out, bab_nb_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--cpus_total', type=int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--result_path', type=str, help='where to store results')
    parser.add_argument('--dataset', type=str, default="cifar_oval")
    parser.add_argument('--json', type=str, help='json file storing the BaB settings (overrides other args)')
    parser.add_argument('--change_eps_const', type=float)
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--batch_size', type=int, help='batch size / 2 for how many domain computations in parallel')
    parser.add_argument('--gurobi_p', type=int, help='number of threads to use in parallelizing gurobi over domains, '
                                                     'or running the MIP solver', default=1)
    parser.add_argument('--method', type=str, choices=["prox", "adam", "gurobi", "gurobi-anderson", "dj-adam",
        "sp-fw", "cut", "cut-sp-fw", "anderson-mip", "bigm-adam", "crown", "alpha-crown", "gamma-crown",
        "beta-crown"], help='method to employ for bounds (or MIP)')
    parser.add_argument('--tot_iter', type=float, help='how many total iters to use for the method. '
                                                       '-1 means automatic from BaB', default=100)
    parser.add_argument('--max_solver_batch', type=float, default=10000, help='max batch size for bounding computations')
    parser.add_argument('--max_cpu_subdomains', type=float, help='max batch size for bounding computations')
    parser.add_argument('--parent_init', action='store_true', help='whether to initialize the code from the parent')
    parser.add_argument('--n_cuts', type=int, help='number of anderson cuts to employ (per neuron)')
    parser.add_argument('--eta', type=float, default=1e2)
    parser.add_argument('--feta', type=float, default=1e2)
    parser.add_argument('--init_step', type=float, default=1e-3)
    parser.add_argument('--fin_step', type=float, default=None)
    parser.add_argument('--step_decay', type=float, default=0.98)
    parser.add_argument('--topk', type=int, default=0)

    # Upper Bounding related parameters
    parser.add_argument('--do_ubs', action='store_true', help='whether to compute UBs as well with output bounding')
    parser.add_argument('--ub_method', type=str, choices=['mi-fgsm'],
                        help='type of attacking method used', default=None)

    # Branching-related parameters.
    parser.add_argument('--bheuristic', type=str, choices=['SR', 'FSB'],
                        help='type of branching choice used', default='FSB')
    parser.add_argument('--max_domains', type=int, help='max number of domains at once for FSB heuristic')

    # Anderson-based bounding methods parameters.
    parser.add_argument('--no_easy', action='store_true', help='whether to avoid the two-way bound system (easy+hard) '
                                                               'for anderson-based bounding algos')
    parser.add_argument('--mip_no_cuts', action='store_true', help='whether to do MIP without Anderson cuts')
    parser.add_argument('--hard_iter', type=float, default=100)
    parser.add_argument('--hard_overhead', type=float, default=100)   # hard bounding overhead at full batch
    parser.add_argument('--cut_add', type=float, default=2)
    parser.add_argument('--fw_start', type=float, default=10)
    parser.add_argument('--fw_cut_iters', type=int, default=100)
    parser.add_argument('--dualinit_iters', type=int, default=500)
    parser.add_argument('--dualinit_init_step', type=float, default=1e-3)
    parser.add_argument('--dualinit_fin_step', type=float, default=1e-4)
    parser.add_argument('--primalinit_init_step', type=float, default=1e-1)
    parser.add_argument('--primalinit_fin_step', type=float, default=1e-3)
    parser.add_argument('--hard_batch_size', type=int, default=-1)
    parser.add_argument('--restrict_factor', type=float, default=-1)

    # Intermediate bounds related parameters.
    parser.add_argument('--tighter_ib', action='store_true', help='whether to get tighter ibs for all')
    parser.add_argument('--opt_ib', action='store_true', help='whether to optimise on selected ibs')
    parser.add_argument('--opt_ib_k', type=int, help='how many selected ibs to optimise over per layer', default=5)
    parser.add_argument('--fixed_ib', action='store_true', help='whether to compute IBs only at root')
    parser.add_argument('--joint_ib', action='store_true', help='whether to compute IBs jointly with last layer bounds.'
                                                                ' Supported only by alpha-crown and gamma-crown.')

    args = parser.parse_args()

    if args.joint_ib and args.method not in ["alpha-crown", "gamma-crown", "beta-crown"]:
        raise IOError("Joint IB optimization supported only for alpha-crown and gamma/beta-crown.")
    
    if args.json:
        # load json file with parameters
        with open(args.json) as json_file:
            json_params = json.load(json_file)
        dataset = json_params["bab"]["dataset"]
        properties = json_params["bab"]["properties"][args.nn_name] if "properties" in json_params["bab"] else None
        max_solver_batch = json_params["ibs"]["loose_ib"]["max_solver_batch"][args.nn_name]
        del json_params
    else:
        dataset = args.dataset
        properties = args.pdprops
        max_solver_batch = args.max_solver_batch

    # initialize a file to record all results, record should be a pandas dataframe
    if dataset in ["cifar_oval", "cifar_oval_1vsall", "cifar_colt"]:
        path = './verification_properties/'
        result_path = './cifar_results/' if not args.result_path else args.result_path

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    elif dataset == "mnist_colt":
        path = './mnist_batch_verification_results/'
        result_path = './mnist_results/' if not args.result_path else args.result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise NotImplementedError

    # load all properties
    if dataset=="mnist_colt" or dataset=="cifar_colt":
        dataset_prefix = "mnist" if dataset == "mnist_colt" else "cifar10"
        csvfile = open('./verification_properties/%s_test.csv'%(dataset_prefix), 'r')
        tests = list(csv.reader(csvfile, delimiter=','))
        if properties:
            gt_results = pd.read_pickle(path + properties)
            bnb_ids = gt_results.index
            batch_ids = bnb_ids
            enum_batch_ids = enumerate(batch_ids)
        else:
            # Use a constant epsilon.
            batch_ids = range(100)
            batch_ids_run = batch_ids
            enum_batch_ids = [(bid, None) for bid in batch_ids_run]
    elif dataset in ["cifar_oval", "cifar_oval_1vsall"]:
        gt_results = pd.read_pickle(path + properties)
        bnb_ids = gt_results.index
        batch_ids = bnb_ids
        enum_batch_ids = enumerate(batch_ids)

    # Set up the correct file naming. (should be simplified).
    if args.json:
        # Use json name for the record naming.
        json_name = os.path.basename(args.json.replace(".json", ""))
        if properties:
            record_name = result_path + properties.replace(".pkl", "") + "_" + json_name + ".pkl"
        else:
            record_name = result_path + args.nn_name + "_" + json_name + ".pkl"
    elif args.record_name is not None:
        record_name = args.record_name
    else:
        method_name = ''

        parent_init = "-pinit" if args.parent_init else ""
        algo_string = ""
        if args.method == "prox":
            algo_string += f"-eta{args.eta}-feta{args.feta}"
        elif args.method in ["adam", "dj-adam", "cut", "cut-sp-fw", "bigm-adam"]:
            algo_string += f"-ilr{args.init_step},flr{args.fin_step}"
        elif args.method in ["alpha-crown", "gamma-crown", "beta-crown"]:
            algo_string += f"-ilr{args.init_step}-lrd{args.step_decay}"
        if "cut" in args.method:
            algo_string += f"-cut_add{args.cut_add}"
            if args.restrict_factor != -1:
                algo_string += f"-restrict_factor{args.restrict_factor}"
        if "sp-fw" in args.method:
            algo_string += f"-fw_start{args.fw_start}"
        if args.method in ["sp-fw", "cut", "cut-sp-fw"]:
            algo_string += f"-diilr{args.dualinit_init_step},diflr{args.dualinit_fin_step}"

        if args.method not in ["gurobi", "gurobi-anderson", "sp-fw", "cut", "cut-sp-fw", "anderson-mip", "crown"]:
            algorithm_name = f"{args.method}_{int(args.tot_iter)}"
        elif args.method in ["sp-fw", "cut"]:
            algorithm_name = f"{args.method}_{int(args.hard_iter)}"
        elif args.method == "cut-sp-fw":
            algorithm_name = f"{args.method}_{int(args.fw_cut_iters)}-{int(args.hard_iter)}"
        elif args.method == "gurobi-anderson":
            algorithm_name = f'{args.method}_{args.n_cuts}'
        else:
            algorithm_name = f'{args.method}'

        add_flags = ""
        flag_keys = ["do_ubs", "no_easy", "tighter_ib", "fixed_ib", "joint_ib"]
        for ckey in flag_keys:
            add_flags += f"_{ckey}" if args.__dict__[ckey] else ""
        add_flags += f"_opt_ib{args.opt_ib_k}" if args.opt_ib else ""
        add_flags += f"_large_batch" if args.batch_size > 500 else ""
        algorithm_name += add_flags

        branching_name = f'{args.bheuristic}'

        # branching choices
        if args.method == "anderson-mip":
            method_name += f'{algorithm_name}'
            method_name += f"_no_cuts" if args.mip_no_cuts else ""
        else:
            method_name += f'{branching_name}_{algorithm_name}{parent_init}{algo_string}'

        if dataset=="mnist_colt" or dataset=="cifar_colt":
            base_name = f'{args.nn_name}' if not properties else f'{properties[:-4]}'
        else:
            base_name = f'{properties[:-4]}'
        record_name = result_path + f'{base_name}_{method_name}.pkl'

    columns = ["Idx", "Eps", "prop"]
    if args.json:
        column_name = json_name
    elif args.record_name:
        column_name = record_name.replace(".pkl", "")
    elif args.method == "anderson-mip":
        column_name = algorithm_name
    else:
        column_name = f"{branching_name}_{algorithm_name}"
    columns += [f'BSAT_{column_name}', f'BBran_{column_name}', f'BTime_{column_name}']

    if os.path.isfile(record_name):
        graph_df = pd.read_pickle(record_name)
    else:
        indices = list(range(len(batch_ids)))

        graph_df = pd.DataFrame(index=indices, columns=columns)
        graph_df.to_pickle(record_name)

    if args.method in ["gurobi", "gurobi-anderson"]:
        if args.gurobi_p > 1:
            mp.set_start_method('spawn')  # for some reason, everything hangs w/o this

    problem_id = []
    for new_idx, idx in enum_batch_ids:

        torch.cuda.empty_cache()
        gc.collect()  # Garbage-collect cpu memory.

        # record_info
        # print(record_name)
        graph_df = pd.read_pickle(record_name)
        if pd.isna(graph_df.loc[new_idx]['Eps']) == False:
            print(f'the {new_idx}th element is done')
            # skip = True
            continue

        if dataset in ["cifar_oval", "cifar_oval_1vsall"]:

            imag_idx = gt_results.loc[idx]["Idx"]
            prop_idx = gt_results.loc[idx]['prop']
            eps_temp = gt_results.loc[idx]["Eps"]

            # skip the nan prop_idx or eps_temp
            if (math.isnan(imag_idx) or math.isnan(eps_temp)):
                continue

            if dataset == "cifar_oval":
                x, verif_layers, test = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
                # since we normalise cifar data set, it is unbounded now
                assert test == prop_idx
                domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
            elif dataset == "cifar_oval_1vsall":
                x, _, verif_layers, domain = load_cifar_oval_kw_1_vs_all(args.nn_name, int(imag_idx), epsilon=eps_temp,
                                                                      max_solver_batch=max_solver_batch)
            adv_spec = None

        elif dataset == "mnist_colt" or dataset=="cifar_colt":
            # colt nets as used in VNN-COMP.

            if properties:
                # Pickle file with epsilons for each image provided.
                imag_idx = gt_results.loc[idx]["Idx"]
                prop_idx = gt_results.loc[idx]['prop']
                eps_temp = gt_results.loc[idx]["Eps"]
                if math.isnan(imag_idx):
                    # handle misclassified images
                    continue
            else:
                # Use a constant epsilon (in the net file).
                imag_idx = new_idx
                try:
                    eps_temp = float(args.nn_name[6:]) if dataset == "mnist_colt" else float(args.nn_name.split('_')[1])/float(args.nn_name.split('_')[2])
                except:
                    eps_temp = None

            x, y, verif_layers, domain = load_1toall_eth(dataset, args.nn_name, idx=imag_idx, test=tests, eps_temp=eps_temp,
                                                         max_solver_batch=max_solver_batch)

            if x is None:
                # handle misclassified images
                continue

            _, _, model, _ = load_1toall_eth(dataset, args.nn_name, idx=imag_idx, test=tests, eps_temp=eps_temp,
                                                         max_solver_batch=max_solver_batch, no_verif_layers=True)

            adv_spec = (model, x, y)
            prop_idx = None

        else:
            raise NotImplementedError(f"Dataset {dataset} currently not supported.")

        ### BaB
        bab_start = time.time()
        if args.json or args.method != "anderson-mip":
            if eps_temp is not None:
                gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
            else:
                gt_prop = f'{dataset}_property_{new_idx}'
            print(gt_prop)
            return_dict = dict()
            if args.json:
                with open(args.json) as json_file:
                    json_params = json.load(json_file)
                bab_from_json(json_params, verif_layers, domain, return_dict, args.nn_name, gpu=gpu)
                del json_params
            else:
                bab(verif_layers, domain, return_dict, args.timeout, args.batch_size, args.method, args.tot_iter,
                    args.parent_init, args)

            bab_out, bab_nb_states = bab_output_from_return_dict(return_dict)
        else:
            # Run MIP with Anderson cuts.
            if gpu:
                cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
                domain = domain.cuda()
            else:
                cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

            # use best of naive interval propagation and KW as intermediate bounds
            if args.tighter_ib:
                # use best of CROWN and KW as intermediate bounds
                intermediate_net = Propagation(cuda_verif_layers, type="best_prop",
                                               params={"best_among": ["KW", "crown"]}, max_batch=args.max_solver_batch)
            else:
                # use best of naive interval propagation and KW as intermediate bounds
                intermediate_net = Propagation(cuda_verif_layers, type="best_prop",
                                               params={"best_among": ["KW", "crown"]}, max_batch=args.max_solver_batch)
            intermediate_net.define_linear_approximation(domain.unsqueeze(0))

            anderson_mip_net = AndersonLinearizedNetwork(
                verif_layers, mode="mip-exact", n_cuts=args.n_cuts, decision_boundary=decision_bound)

            cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab_utils.subproblems_to_cpu(
                domain.unsqueeze(0), intermediate_net.lower_bounds, intermediate_net.upper_bounds, squeeze=True)
            anderson_mip_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs),
                                                      n_threads=args.gurobi_p)

            sat_status, global_lb, bab_nb_states = anderson_mip_net.solve_mip(timeout=args.timeout,
                                                                              insert_cuts=(not args.mip_no_cuts))

            bab_out = str(sat_status) if sat_status is not None else "timeout"
            print(f"MIP SAT status: {bab_out}")

        print(f"Nb states visited: {bab_nb_states}")
        print('\n')

        bab_end = time.time()
        bab_time = bab_end - bab_start
        print('total time required: ', bab_time)

        print('\n')
        graph_df.loc[new_idx]["Idx"] = imag_idx
        graph_df.loc[new_idx]["Eps"] = eps_temp
        graph_df.loc[new_idx]["prop"] = prop_idx

        graph_df.loc[new_idx][f"BSAT_{column_name}"] = bab_out
        graph_df.loc[new_idx][f"BBran_{column_name}"] = bab_nb_states
        graph_df.loc[new_idx][f"BTime_{column_name}"] = bab_time
        graph_df.to_pickle(record_name)

    print("count True", list(graph_df['BSAT_cifar_colt_eth_gammacrown_as']).count('True'))
    print("count False", list(graph_df['BSAT_cifar_colt_eth_gammacrown_as']).count('False'))
    print("count timeout", list(graph_df['BSAT_cifar_colt_eth_gammacrown_as']).count('timeout'))
    print('problematic idx:', problem_id)


if __name__ == '__main__':
    main()
