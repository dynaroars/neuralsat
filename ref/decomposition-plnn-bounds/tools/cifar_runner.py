import argparse
import torch
from tools.experiment_utils import load_network, make_elided_models, cifar_loaders
from plnn_bounds.proxlp_solver.solver import SaddleLP
from plnn_bounds.proxlp_solver.dj_relaxation import DJRelaxationLP
import copy
import time


def runner():
    parser = argparse.ArgumentParser(description="Compute a bound and plot the results")
    parser.add_argument('network_filename', type=str,
                        help='Path ot the network')
    parser.add_argument('eps', type=float, help='Epsilon')
    parser.add_argument('--img_idx', type=int, default=0)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--init_step', type=float)
    parser.add_argument('--fin_step', type=float)
    parser.add_argument('--out_iters', type=int)
    parser.add_argument('--prox_momentum', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--define_linear_approximation', action='store_true',
                        help="if this flag is true, compute all intermediate bounds w/ the selected algorithm")
    parser.add_argument('--algorithm', type=str, choices=["planet-adam", "proxlp",
                                                          "gurobi-time", "planet-auto-adagrad", "planet-auto-adam",
                                                          "jacobi-proxlp", "dj-adam"],
                        help="which algorithm to use, in case one does init or uses it alone")
    
    args = parser.parse_args()

    # Load all the required data, setup the model
    model = load_network(args.network_filename)
    elided_models = make_elided_models(model)
    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if idx != args.img_idx:
            continue
        elided_model = elided_models[y.item()]
    domain = torch.stack([X.squeeze(0) - args.eps,
                          X.squeeze(0) + args.eps], dim=-1).cuda()

    lin_approx_string = "" if not args.define_linear_approximation else "-allbounds"

    # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
    # and optimize only the last layer
    cuda_elided_model = copy.deepcopy(elided_model).cuda()
    cuda_domain = domain.cuda()
    intermediate_net = SaddleLP([lay for lay in cuda_elided_model])
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(cuda_domain, force_optim=False, no_conv=False)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds

    folder = "./timings_cifar/"

    if args.algorithm == "proxlp":
        # ProxLP
        acceleration_dict = {
            'momentum': args.prox_momentum,  # decent momentum: 0.6 w/ increasing eta
        }

        optprox_params = {
            'nb_total_steps': args.out_iters,
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'eta': args.eta,  # eta is kept the same as in simpleprox
            'initial_eta': args.eta if args.feta else None,
            'final_eta': args.feta if args.feta else None,
            'log_values': False,
            'inner_cutoff': 0,
            'maintain_primal': True,
            'acceleration_dict': acceleration_dict
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        optprox_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        optprox_start = time.time()
        with torch.no_grad():
            optprox_net.set_initialisation('naive')
            optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
            if not args.define_linear_approximation:
                optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = optprox_net.compute_lower_bound(all_optim=True)
            else:
                optprox_net.define_linear_approximation(cuda_domain)
                ub = optprox_net.upper_bounds[-1]
        optprox_end = time.time()
        optprox_time = optprox_end - optprox_start
        optprox_ubs = ub.cpu().mean()
        print(f"ProxLP Time: {optprox_time}")
        print(f"ProxLP UB: {optprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{optprox_ubs},Time:{optprox_time},Eta{args.eta},Out-iters:{args.out_iters}\n")

        acceleration_string = f"-mom:{args.prox_momentum}"
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},eta:{args.eta}-feta:{args.feta}{acceleration_string}{lin_approx_string}.pickle"
        torch.save(optprox_net.logger, pickle_name)

    elif args.algorithm == "jacobi-proxlp":
        # ProxLP, jacobi

        optprox_params = {
            'nb_total_steps': args.out_iters,
            'max_nb_inner_steps': 5,
            'eta': args.eta,  # eta is kept the same as in simpleprox
            'initial_eta': args.eta if args.feta else None,
            'final_eta': args.feta if args.feta else None,
            'log_values': False,
            'inner_cutoff': 0,
            'maintain_primal': True,
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        optprox_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        optprox_start = time.time()
        with torch.no_grad():
            optprox_net.set_initialisation('naive')
            optprox_net.set_solution_optimizer('prox', optprox_params)
            if not args.define_linear_approximation:
                optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = optprox_net.compute_lower_bound(all_optim=True)
            else:
                optprox_net.define_linear_approximation(cuda_domain)
                ub = optprox_net.upper_bounds[-1]
        optprox_end = time.time()
        optprox_time = optprox_end - optprox_start
        optprox_ubs = ub.cpu().mean()
        print(f"ProxLP Time: {optprox_time}")
        print(f"ProxLP UB: {optprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(
                folder + f"{args.algorithm},UB:{optprox_ubs},Time:{optprox_time},Eta{args.eta},Out-iters:{args.out_iters}\n")

        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},eta:{args.eta}-feta:{args.feta}{lin_approx_string}.pickle"
        torch.save(optprox_net.logger, pickle_name)

    elif args.algorithm == "planet-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        adam_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        adam_start = time.time()
        with torch.no_grad():
            adam_net.set_initialisation('naive')
            adam_net.set_solution_optimizer('adam', adam_params)
            if not args.define_linear_approximation:
                adam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = adam_net.compute_lower_bound(all_optim=True)
            else:
                adam_net.define_linear_approximation(cuda_domain)
                ub = adam_net.upper_bounds[-1]
        adam_end = time.time()
        adam_time = adam_end - adam_start
        adam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {adam_time}")
        print(f"Planet adam UB: {adam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{adam_ubs},Time:{adam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(adam_net.logger, pickle_name)

    elif args.algorithm == "planet-auto-adam":
        adam_params = {
            'algorithm': "adam",
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        adam_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        adam_start = time.time()
        with torch.no_grad():
            adam_net.set_initialisation('naive')
            adam_net.set_solution_optimizer('autograd', adam_params)
            if not args.define_linear_approximation:
                adam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = adam_net.compute_lower_bound(all_optim=True)
            else:
                adam_net.define_linear_approximation(cuda_domain)
                ub = adam_net.upper_bounds[-1]
        adam_end = time.time()
        adam_time = adam_end - adam_start
        adam_ubs = ub.cpu().mean()
        print(f"Planet adam autograd Time: {adam_time}")
        print(f"Planet adam autograd UB: {adam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{adam_ubs},Time:{adam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(adam_net.logger, pickle_name)

    elif args.algorithm == "dj-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        djadam_net = DJRelaxationLP([lay for lay in cuda_elided_model], params=adam_params,
                                    store_bounds_progress=len(intermediate_net.weights))
        djadam_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                djadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = djadam_net.compute_lower_bound(all_optim=True)
            else:
                djadam_net.define_linear_approximation(cuda_domain)
                ub = djadam_net.upper_bounds[-1]
        djadam_end = time.time()
        djadam_time = djadam_end - djadam_start
        djadam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {djadam_time}")
        print(f"Planet adam UB: {djadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{djadam_ubs},Time:{djadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(djadam_net.logger, pickle_name)

    elif args.algorithm == "planet-auto-adagrad":
        adam_params = {
            'algorithm': "adagrad",
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'log_values': False
        }
        cuda_elided_model = copy.deepcopy(elided_model).cuda()
        cuda_domain = domain.cuda()
        adagrad_net = SaddleLP([lay for lay in cuda_elided_model], store_bounds_progress=len(intermediate_net.weights))
        adam_start = time.time()
        with torch.no_grad():
            adagrad_net.set_initialisation('KW')
            adagrad_net.set_solution_optimizer('autograd', adam_params)
            if not args.define_linear_approximation:
                adagrad_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                _, ub = adagrad_net.compute_lower_bound(all_optim=True)
            else:
                adagrad_net.define_linear_approximation(cuda_domain)
                ub = adagrad_net.upper_bounds[-1]
        adam_end = time.time()
        adam_time = adam_end - adam_start
        adam_ubs = ub.cpu().mean()
        print(f"Planet adam autograd Time: {adam_time}")
        print(f"Planet adam autograd UB: {adam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},UB:{adam_ubs},Time:{adam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{args.img_idx}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(adagrad_net.logger, pickle_name)


if __name__ == '__main__':
    runner()
