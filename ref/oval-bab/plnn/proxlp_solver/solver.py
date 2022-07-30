import math
import time
import torch

from plnn.dual_bounding import DualBounding
from plnn.proxlp_solver.utils import prod, OptimizationTrace, ProxOptimizationTrace, bdot, apply_transforms
from plnn.proxlp_solver.by_pairs import ByPairsDecomposition, DualVarSet
from plnn.branch_and_bound.utils import ParentInit


class SaddleLP(DualBounding):

    def __init__(self, layers, store_bounds_progress=-1, store_bounds_primal=False, max_batch=20000):
        """
        :param store_bounds_progress: whether to store bounds progress over time (-1=False 0=True)
        :param store_bounds_primal: whether to store the primal solution used to compute the final bounds
        :param max_batch: maximal batch size for parallel bounding çomputations over both output neurons and domains
        """
        super().__init__(layers)
        self.optimizers = {
            'init': self.init_optimizer,
            'adam': self.adam_subg_optimizer,
            'autograd': self.autograd_optimizer,
            'subgradient': self.subgradient_optimizer,
            'prox': self.prox_optimizer,
            'optimized_prox': self.optimized_prox_optimizer,
            'comparison': self.comparison_optimizer,
            'best_inits': self.best_init_optimizers
        }

        self.decomposition = ByPairsDecomposition('KW')
        self.optimize, _ = self.init_optimizer(None)

        self.store_bounds_progress = store_bounds_progress
        self.store_bounds_primal = store_bounds_primal
        self.max_batch = max_batch

    def set_decomposition(self, decomp_style, decomp_args, ext_init=None):
        decompositions = {
            'pairs': ByPairsDecomposition,
        }
        assert decomp_style in decompositions
        self.decomposition = decompositions[decomp_style](decomp_args)

        if ext_init is not None:
            self.decomposition.set_external_initial_solution(ext_init)

    def set_solution_optimizer(self, method, method_args=None):
        assert method in self.optimizers
        self.method = method
        self.optimize, self.logger = self.optimizers[method](method_args)

    def init_optimizer(self, method_args):
        return self.init_optimize, None

    def best_init_optimizers(self, method_args):
        # method_args must contain a list of init types. We return the best amongst these. (the logger is dummy)
        # children_init is going to store the duals of the last algorithm in the methods list
        default_methods = ["naive", "KW"]
        methods = default_methods if method_args is None else method_args
        c_fun, c_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            self.set_decomposition('pairs', methods[0])
            best_bounds = c_fun(*args, **kwargs)
            for method in methods[1:]:
                self.set_decomposition('pairs', method)
                c_bounds = c_fun(*args, **kwargs)
                best_bounds = torch.max(c_bounds, best_bounds)
            return best_bounds

        return optimize, [c_logger for _ in methods]

    def init_optimize(self, weights, final_coeffs,
                      lower_bounds, upper_bounds):
        '''
        Simply use the values that it has been initialized to.
        '''
        dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                             lower_bounds, upper_bounds)
        matching_primal_vars = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                   lower_bounds, upper_bounds,
                                                                   dual_vars)
        if self.store_bounds_primal:
            self.bounds_primal = matching_primal_vars
            # store last dual solution for future usage
            self.children_init = DecompositionPInit(dual_vars.rhos)
        bound = self.decomposition.compute_objective(dual_vars, matching_primal_vars, final_coeffs)
        return bound

    def subgradient_optimizer(self, method_args):
        # Define default values
        args = {
            'nb_steps': 100,
            'initial_step_size': 1e-2,
            'final_step_size': 1e-4
        }
        args.update(method_args)

        self.steps = args['nb_steps']
        initial_step_size = args['initial_step_size']
        final_step_size = args['final_step_size']
        logger = None
        if self.store_bounds_progress >= 0:
            logger = OptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()
            dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                 lower_bounds, upper_bounds)
            for step in range(self.steps):
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)
                dual_subg = matching_primal.as_dual_subgradient()
                step_size = initial_step_size + (step / self.steps) * (final_step_size - initial_step_size)
                dual_vars.add_(step_size, dual_subg)

                if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                    if (step - 1) % 10 == 0:
                        start_logging_time = time.time()
                        matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                              lower_bounds, upper_bounds,
                                                                              dual_vars)
                        bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                        logging_time = time.time() - start_logging_time
                        logger.add_point(len(weights), bound, logging_time=logging_time)

            # End of the optimization
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
            return bound

        return optimize, logger

    def adam_subg_optimizer(self, method_args):
        # step size can decay linearly or exponentially. For linear decay, specify final_step_size, and leave step_size_decay as None.
        # for exponential decay, do the opposite.
        # Define default values
        args = {
            'nb_steps': 100,
            'outer_cutoff': None,
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            'step_decay_rate': None,
            'betas': (0.9, 0.999)
        }
        args.update(method_args)

        self.steps = args['nb_steps']
        outer_cutoff = args['outer_cutoff']
        use_cutoff = (outer_cutoff is not None) and outer_cutoff > 0
        initial_step_size = args['initial_step_size']
        if args['final_step_size'] is not None:
            final_step_size = args['final_step_size']
            exp_decay_steps = False
        elif args['step_decay_rate'] is not None:
            step_decay_rate = args['step_decay_rate']
            exp_decay_steps = True
        else:
            raise NotImplementedError
        beta_1 = args['betas'][0]
        beta_2 = args['betas'][1]
        logger = None
        if self.store_bounds_progress >= 0:
            logger = OptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()

            dual_vars = self.decomposition.initial_dual_solution(
                weights, final_coeffs, lower_bounds, upper_bounds)
            matching_primal = self.decomposition.get_optim_primal(
                weights, final_coeffs, lower_bounds, upper_bounds, dual_vars)
            init_bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
            exp_avg = dual_vars.zero_like()
            exp_avg_sq = dual_vars.zero_like()

            if use_cutoff:
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)
                old_bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                diff_avg = torch.zeros_like(old_bound)

            if exp_decay_steps:
                step_schedule = initial_step_size

            for step in range(1, self.steps+1):
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)

                dual_subg = matching_primal.as_dual_subgradient()

                if exp_decay_steps:
                    step_schedule = step_schedule * step_decay_rate
                    step_size = step_schedule
                else:
                    step_size = initial_step_size + (step / self.steps) * (final_step_size - initial_step_size)

                bias_correc1 = 1 - beta_1 ** step
                bias_correc2 = 1 - beta_2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta_1).add_(1-beta_1, dual_subg)
                exp_avg_sq.mul_(beta_2).addcmul_(1-beta_2, dual_subg, dual_subg)
                denom = (exp_avg_sq.sqrt().div_cte_(math.sqrt(bias_correc2))).add_cte_(1e-8)

                step_size = step_size / bias_correc1

                dual_vars.addcdiv_(step_size, exp_avg, denom)

                if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                    if (step - 1) % 10 == 0:
                        start_logging_time = time.time()
                        matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                              lower_bounds, upper_bounds,
                                                                              dual_vars)
                        bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                        logging_time = time.time() - start_logging_time
                        logger.add_point(len(weights), bound, logging_time=logging_time)

                # Stop outer iterations if improvement in bounds (running average of bounds diff) is small.
                if use_cutoff:
                    matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                          lower_bounds, upper_bounds,
                                                                          dual_vars)
                    bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                    diff_avg = 0.5 * diff_avg + 0.5 * (bound - old_bound)
                    old_bound = bound.clone()
                    if diff_avg.mean() < outer_cutoff and step > 10:
                        print(
                            f"Breaking inner optimization after {step} iterations, decrease {diff_avg.mean()}")
                        break

            # store last dual solution for future usage
            # TODO: make this compatible with L1 code too
            if isinstance(self.decomposition, ByPairsDecomposition):
                self.children_init = DecompositionPInit(dual_vars.rhos)
            else:
                self.children_init = ParentInit()

            # End of the optimization
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            if self.store_bounds_primal:
                self.bounds_primal = matching_primal
            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)

            return torch.max(init_bound, bound)

        return optimize, logger

    def prox_optimizer(self, method_args):
        # Define default values
        args = {
            'nb_total_steps': 100,
            'max_nb_inner_steps': 10,
            'eta': 1e-3,
            'initial_eta': None,
            'final_eta': None,
            'inner_cutoff': 1e-3,
        }
        args.update(method_args)
        self.steps = args['nb_total_steps']
        max_nb_inner_steps = args['max_nb_inner_steps']
        default_eta = args['eta']
        initial_eta = args['initial_eta']
        final_eta = args['final_eta']
        inner_cutoff = args['inner_cutoff']
        logger = None
        if self.store_bounds_progress >= 0:
            logger = ProxOptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()
            dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                 lower_bounds, upper_bounds)
            primal_vars = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                              lower_bounds, upper_bounds,
                                                              dual_vars)
            init_bound = self.decomposition.compute_objective(dual_vars, primal_vars, final_coeffs)
            prox_dual_vars = dual_vars
            steps = 0
            # We operate in the primal, and are going to keep updating our
            # primal_vars. For each primal vars, we have a formula giving the
            # associated dual variables, and the hope is that optimizing
            # correctly the primal variables to shrink the dual gap will lead
            # to a good solution on the dual.
            while steps < self.nb_total_steps:
                prox_dual_vars = dual_vars

                if (initial_eta is not None) and (final_eta is not None):
                    eta = initial_eta + (steps / self.steps) * (final_eta - initial_eta)
                    # eta = initial_eta / (1 + (steps / nb_total_steps) * (initial_eta/final_eta - 1))
                else:
                    eta = default_eta
                # Get lambda, rho:
                # For the proximal problem, they are the gradient on the z_a - z_b differences.
                dual_vars = prox_dual_vars.add(primal_vars.as_dual_subgradient(),
                                               1/eta)
                nb_inner_step = min(max_nb_inner_steps, self.steps - steps)
                for inner_step in range(nb_inner_step):
                    # Get the conditional gradient over z, zhat by maximizing
                    # the linear function (given by gradient), over the
                    # feasible domain
                    cond_grad = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                    lower_bounds, upper_bounds,
                                                                    dual_vars)


                    # Compute the optimal step size
                    # decrease gives the improvement we make in the primal proximal problem
                    opt_step_size, decrease = SaddleLP.proximal_optimal_step_size(final_coeffs, dual_vars,
                                                                                  primal_vars, cond_grad,
                                                                                  eta)
                    # Update the primal variables
                    primal_vars = primal_vars.weighted_combination(cond_grad, opt_step_size)

                    # Update the dual variables
                    dual_vars = prox_dual_vars.add(primal_vars.as_dual_subgradient(),
                                                   1/eta)
                    steps += 1

                    # Depending on how much we made as improvement on the
                    # primal proximal problem, maybe move to the next proximal
                    # iteration
                    if decrease.max() < inner_cutoff:
                        # print(f"Breaking inner optimization after {inner_step} iterations")
                        break

                    if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                        if steps % 10 == 0:
                            start_logging_time = time.time()
                            objs = self.decomposition.compute_proximal_objective(primal_vars, dual_vars, prox_dual_vars,
                                                                                 final_coeffs, eta)
                            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                                  lower_bounds, upper_bounds,
                                                                                  dual_vars)
                            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                            logging_time = time.time() - start_logging_time
                            logger.add_proximal_point(len(weights), bound, objs, logging_time=logging_time)

            # End of optimization
            # Compute an actual bound
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
            return torch.max(init_bound, bound)

        return optimize, logger

    # IMPORTANT: this is slower than adam_subg_optimizer (recomputes the grad for no reason)
    def autograd_optimizer(self, method_args):
        # employ a pytorch autograd optimizer on this derivation (variable splitting)

        # Define default values
        args = {
            'nb_steps': 100,
            'algorithm': 'adam',
            'initial_step_size': 1e-3,
            'betas': (0.9, 0.999)
        }
        args.update(method_args)

        self.steps = args['nb_steps']
        initial_step_size = args['initial_step_size']
        algorithm = args['algorithm']
        assert algorithm in ["adam", "adagrad"]
        beta_1 = args['betas'][0]
        beta_2 = args['betas'][1]
        logger = None
        if self.store_bounds_progress >= 0:
            logger = OptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()
            assert type(self.decomposition) is ByPairsDecomposition

            with torch.enable_grad():
                c_rhos = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds).rhos

                # define objective function
                def obj(rhos):
                    c_dual_vars = DualVarSet(rhos)
                    matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                          lower_bounds, upper_bounds,
                                                                          c_dual_vars)
                    bound = self.decomposition.compute_objective(c_dual_vars, matching_primal, final_coeffs)
                    return bound

                for rho in c_rhos:
                    rho.requires_grad = True

                if algorithm == "adam":
                    optimizer = torch.optim.Adam(c_rhos, lr=initial_step_size, betas=(beta_1, beta_2))
                else:
                    # "adagrad"
                    optimizer = torch.optim.Adagrad(c_rhos, lr=initial_step_size)  # lr=1e-2 works best

                # do autograd-adam
                for step in range(self.steps):
                    optimizer.zero_grad()
                    obj_value = -obj(c_rhos)
                    obj_value.mean().backward()
                    # print(obj_value.mean())
                    optimizer.step()

                    if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                        if (step - 1) % 10 == 0:
                            start_logging_time = time.time()
                            dual_detached = [rho.detach() for rho in c_rhos]
                            bound = obj(dual_detached)
                            logging_time = time.time() - start_logging_time
                            logger.add_point(len(weights), bound, logging_time=logging_time)

                dual_detached = [rho.detach() for rho in c_rhos]
                # store last dual solution for future usage
                self.children_init = DecompositionPInit(dual_detached)

                # End of the optimization
                dual_vars = DualVarSet(dual_detached)
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)
                bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)

            return bound

        return optimize, logger

    def optimized_prox_optimizer(self, method_args):
        # Define default values
        args = {
            'nb_total_steps': 100,
            'max_nb_inner_steps': 10,
            'eta': 1e-3,
            'initial_eta': None,
            'final_eta': None,
            'outer_cutoff': None,
            'acceleration_dict': {'momentum': 0}
        }
        args.update(method_args)
        self.steps = args['nb_total_steps']
        max_nb_inner_steps = args['max_nb_inner_steps']
        default_eta = args['eta']
        initial_eta = args['initial_eta']
        final_eta = args['final_eta']
        outer_cutoff = args['outer_cutoff']
        use_cutoff = (outer_cutoff is not None) and outer_cutoff > 0
        acceleration_dict = args['acceleration_dict']

        if acceleration_dict and acceleration_dict['momentum'] != 0:
            assert type(self.decomposition) is ByPairsDecomposition

        logger = None
        if self.store_bounds_progress >= 0:
            logger = ProxOptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()

            dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                 lower_bounds, upper_bounds)
            primal_vars = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                              lower_bounds, upper_bounds,
                                                              dual_vars)
            init_bound = self.decomposition.compute_objective(dual_vars, primal_vars, final_coeffs)

            if use_cutoff:
                old_bound = self.decomposition.compute_objective(dual_vars, primal_vars, final_coeffs)
                diff_avg = torch.zeros_like(old_bound)

            prox_dual_vars = dual_vars.copy()
            steps = 0
            # We operate in the primal, and are going to keep updating our
            # primal_vars. For each primal vars, we have a formula giving the
            # associated dual variables, and the hope is that optimizing
            # correctly the primal variables to shrink the dual gap will lead
            # to a good solution on the dual.
            while steps < self.steps:
                dual_vars.update_acceleration(acceleration_dict=acceleration_dict)
                prox_dual_vars = dual_vars.copy()

                if (initial_eta is not None) and (final_eta is not None):
                    eta = initial_eta + (steps / self.steps) * (final_eta - initial_eta)
                else:
                    eta = default_eta
                # Get lambda, rho:
                # For the proximal problem, they are the gradient on the z_a - z_b differences.
                dual_vars.update_from_anchor_points(prox_dual_vars, primal_vars, eta, acceleration_dict=acceleration_dict)
                nb_inner_step = min(max_nb_inner_steps, self.steps - steps)
                for inner_step in range(nb_inner_step):
                    # Get the conditional gradient over z, zhat by maximizing
                    # the linear function (given by gradient), over the
                    # feasible domain

                    n_layers = len(weights)
                    for lay_idx, (layer, lb_k, ub_k) in enumerate(zip(weights,
                                                                      lower_bounds,
                                                                      upper_bounds)):
                        # Perform conditional gradient steps after each subgradient update.
                        subproblem_condgrad = self.decomposition.get_optim_primal_layer(
                            lay_idx, n_layers, layer, final_coeffs, lb_k, ub_k, dual_vars)

                        # Compute the optimal step size
                        # c_decrease gives the improvement we make in the primal proximal problem
                        opt_step_size, _ = subproblem_condgrad.proximal_optimal_step_size_subproblem(
                            final_coeffs, dual_vars, primal_vars, n_layers, eta)

                        # Update the primal variables
                        primal_vars.weighted_combination_subproblem(subproblem_condgrad, opt_step_size)

                        # Store primal variables locally, for use in initializing ExpLP
                        if type(self.decomposition) is ByPairsDecomposition:
                            self.last_primals = primal_vars

                        # Update the dual variables
                        duals_to_update = []
                        if lay_idx < n_layers - 1:
                            duals_to_update.append(lay_idx)
                        if lay_idx > 0:
                            duals_to_update.append(lay_idx-1)
                        dual_vars.update_from_anchor_points(prox_dual_vars, primal_vars, eta, lay_idx=duals_to_update,
                                                            acceleration_dict=acceleration_dict)

                    steps += 1
                    if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                        if steps % 10 == 0:
                            start_logging_time = time.time()
                            objs = self.decomposition.compute_proximal_objective(primal_vars, dual_vars, prox_dual_vars,
                                                                                 final_coeffs, eta)
                            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                                  lower_bounds, upper_bounds,
                                                                                  dual_vars)
                            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                            logging_time = time.time() - start_logging_time
                            logger.add_proximal_point(len(weights), bound, objs, logging_time=logging_time)

                # Stop outer iterations if improvement in bounds (running average of bounds diff) is small.
                if use_cutoff:
                    matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                          lower_bounds, upper_bounds,
                                                                          dual_vars)
                    bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
                    diff_avg = 0.5 * diff_avg + 0.5 * (bound - old_bound)
                    old_bound = bound.clone()
                    if diff_avg.mean() < outer_cutoff and steps > 10:
                        print(
                            f"Breaking inner optimization after {steps} iterations, decrease {diff_avg.mean()}")
                        break

            # store last dual solution for future usage
            self.children_init = DecompositionPInit(dual_vars.rhos)

            # End of optimization
            # Compute an actual bound
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            if self.store_bounds_primal:
                # This yields better UBs in BaB than matching_primal
                self.bounds_primal = primal_vars
            bound = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
            return torch.max(init_bound, bound)

        return optimize, logger

    def comparison_optimizer(self, method_args):
        opt_to_run = []
        loggers = []
        for param_set in method_args:
            optimize_fun, logger = self.optimizers[param_set['optimizer']](param_set['params'])
            opt_to_run.append(optimize_fun)
            loggers.append(logger)

        def optimize(*args, **kwargs):
            bounds = []
            for opt_fun in opt_to_run:
                bounds.append(opt_fun(*args, **kwargs))
            all_bounds = torch.stack(bounds, 0)
            bounds, _ = torch.max(all_bounds, 0)
            return bounds

        return optimize, loggers

    @staticmethod
    def proximal_optimal_step_size(additional_coeffs, diff_grad,
                                   primal_vars, cond_grad,
                                   eta):

        # TODO: not sure this works w/ batchification

        # If we write the objective function as a function of the step size, this gives:
        # \frac{a}/{2} \gamma^2 + b \gamma + c
        # The optimal step size is given by \gamma_opt = -\frac{b}{2*a}
        # The change in value is given by \frac{a}{2} \gamma_opt^2 + b * \gamma
        # a = \sum_k \frac{1}{eta_k} ||xahat - zahat - (xbhat - zbhat||^2
        # b = \sum_k rho_k (xbhat - zbhat - (xahat - zahat)) + (xahat,n - zahat,n)
        # c is unnecessary

        var_to_cond = primal_vars.as_dual_subgradient().add(cond_grad.as_dual_subgradient(), -1)
        upper = var_to_cond.bdot(diff_grad)
        for layer, add_coeff in additional_coeffs.items():
            # TODO: Check if this is the correct computation ON PAPER
            upper += bdot(add_coeff, primal_vars.zahats[layer-1] - cond_grad.zahats[layer-1])

        lower = var_to_cond.weighted_squared_norm(1/eta)
        torch.clamp(lower, 1e-8, None, out=lower)

        opt_step_size = upper / lower

        opt_step_size = upper / lower
        # Set to 0 the 0/0 entries.
        up_mask = upper == 0
        low_mask = lower == 0
        sum_mask = up_mask + low_mask
        opt_step_size[sum_mask > 1] = 0

        decrease = -0.5 * lower * opt_step_size.pow(2) + upper * opt_step_size

        return opt_step_size, decrease

    def compute_saddle_dual_gap(self, primal_vars, dual_vars, prox_dual_vars,
                                weights, final_coeffs,
                                lower_bounds, upper_bounds,
                                eta, include_prox_terms=False):

        # Compute the objective if we plug in the solution for the dual vars,
        # and are trying to minimize over the primals
        p_as_dual = primal_vars.as_dual_subgradient()
        for_prim_opt_dual_vars = prox_dual_vars.add(p_as_dual, 1/eta)
        primal_val = self.decomposition.compute_objective(for_prim_opt_dual_vars, primal_vars, final_coeffs)
        if include_prox_terms:
            primal_val += p_as_dual.weighted_squared_norm(1/(2*eta))

        # Compute the objective if we plug in the solution for the primal vars, and
        # are trying to maximize over the dual
        matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                              lower_bounds, upper_bounds,
                                                              dual_vars)
        dual_minus_proxdual = dual_vars.add(prox_dual_vars, -1)
        dual_val = self.decomposition.compute_objective(dual_vars, matching_primal, final_coeffs)
        if include_prox_terms:
            dual_val -= dual_minus_proxdual.weighted_squared_norm(eta/2)

        dual_gap = (primal_val - dual_val)

        return primal_val, dual_val, dual_gap

    def dump_instance(self, path_to_file):
        to_save = {
            'layers': self.layers,
            'lbs': self.lower_bounds,
            'ubs': self.upper_bounds,
            'input_domain': self.input_domain
        }
        torch.save(to_save, path_to_file)

    @classmethod
    def load_instance(cls, path_to_file):
        saved = torch.load(path_to_file)

        intermediate_bounds = (saved['lbs'], saved['ubs'])

        inst = cls(saved['layers'])
        inst.build_model_using_bounds(saved['input_domain'],
                                      intermediate_bounds)

        return inst

    def get_lower_bound_network_input(self):
        """
        Return the input of the network that was used in the last bounds computation.
        Converts back from the conditioned input domain to the original one.
        Assumes that the last layer is a single neuron.
        """
        assert self.store_bounds_primal
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        if isinstance(self.decomposition, ByPairsDecomposition):
            assert self.bounds_primal.z0.shape[1] in [1, 2], "the last layer must have a single neuron"
            net_input = (1/2) * (u_0 - l_0) * self.bounds_primal.z0.select(1, self.bounds_primal.z0.shape[1]-1) +\
                        (1/2) * (u_0 + l_0)
        else:
            assert self.bounds_primal.var[0]['x'].shape[1] in [1, 2], "the last layer must have a single neuron"
            net_input = (1 / 2) * (u_0 - l_0) * self.bounds_primal.var[0]['x'].select(1, self.bounds_primal.var[0]['x'].shape[1] - 1) + \
                        (1 / 2) * (u_0 + l_0)

        return apply_transforms(self.input_transforms, net_input, inverse=True)

    def initialize_from(self, external_init):
        # setter to have the optimizer initialized from an external list of dual variables (as list of tensors)
        self.set_decomposition('pairs', 'external', ext_init=external_init)

    def internal_init(self):
        self.set_decomposition('pairs', 'crown')

    # BaB-related method to implement automatic no. of iters.
    def default_iters(self, set_min=False):
        # Set no. of iters to default for the algorithm this class represents.
        if self.method in ["prox", "optimized_prox"]:
            self.min_steps = 50
            self.max_steps = 600
            self.step_increase = 50
        elif self.method in ["adam", "autograd", "subgradient"]:
            self.min_steps = 80
            self.max_steps = 1000
            self.step_increase = 75
        else:
            self.min_steps = 0
            self.max_steps = 0
            self.step_increase = 0
        if set_min and self.get_iters() != -1:
            self.min_steps = self.get_iters()
        self.steps = self.min_steps

    # BaB-related method to implement automatic no. of iters.
    def set_iters(self, iters):
        self.steps = iters

    # BaB-related method to implement automatic no. of iters.
    def get_iters(self):
        return self.steps

    def get_l0_intercept_scores(self):
        """
        Return the L0 intercept score (lagrangian multiplier of the upper Planet constraint times its bias
        in the (xhat_k, x_k) plane).
        Assumes that last layer bounds have recently been computed.
        """
        batch_shape = self.children_init.rhos[0].shape[:2]
        add_coeffs = next(iter(self.additional_coeffs.values()))

        # Do a backward pass over rhos to obtain lambdas, whose positive part corresponds to the lagrangian multiplier
        # of the upper Planet constraint.
        lambdas = []
        for idx, crho in enumerate(self.children_init.rhos[1:]):
            lambdas.append(self.weights[idx+1].backward(crho))
        lambdas.append(self.weights[-1].backward(-add_coeffs))

        # Compute intercept scores, setting them to 0 for non-ambiguous neurons.
        intercept_scores = []
        for lay, (clbs, cubs) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            bias = - (clbs * cubs) / (cubs - clbs)
            bias.masked_fill_(clbs >= 0, 0)
            bias.masked_fill_(cubs <= 0, 0)
            bias.masked_fill_(clbs == cubs, 0)
            if lay > 0 and lay < len(self.lower_bounds) - 1:
                cscore = (bias.unsqueeze(1) * lambdas[lay - 1].clamp_(0, None)).view(batch_shape + (-1,))
            else:
                cscore = torch.zeros(batch_shape + (prod(cubs.shape[1:]),), device=cubs.device, dtype=cubs.dtype)
            intercept_scores.append(cscore)
        return intercept_scores


class DecompositionPInit(ParentInit):
    """
    Parent Init class for Lagrangian Decomposition on PLANET (the prox and supergradient solvers of this file).
    """
    def __init__(self, parent_rhos):
        # parent_rhos are the rhos values (list of tensors, dual values for ByPairsDecomposition) at parent termination
        self.rhos = parent_rhos

    def to_cpu(self):
        # Move content to cpu.
        self.rhos = [crho.cpu() for crho in self.rhos]

    def to_device(self, device):
        # Move content to device "device"
        self.rhos = [crho.to(device) for crho in self.rhos]

    def as_stack(self, stack_size):
        # Repeat (copies) the content of this parent init to form a stack of size "stack_size"
        stacked_rhos = self.do_stack_list(self.rhos, stack_size)
        return DecompositionPInit(stacked_rhos)

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        for x_idx in range(len(self.rhos)):
            self.set_parent_entries(self.rhos[x_idx], parent_solution.rhos[x_idx], batch_idx)

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        return DecompositionPInit(self.get_entry_list(self.rhos, batch_idx))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        # this operation makes sense only in the BaB context (single output neuron), when both lb and ub where computed.
        assert self.rhos[0].shape[1] == 2
        return DecompositionPInit(self.lb_only_list(self.rhos))

    def expand_layer_batch(self, expand_size):
        self.rhos = [crho.expand(crho.shape[0], expand_size, *crho.shape[2:]) for crho in self.rhos]

    def contract_layer_batch(self):
        self.rhos = [crho.select(1, -1).unsqueeze(1) for crho in self.rhos]

    def clone(self):
        return DecompositionPInit([crho.clone() for crho in self.rhos])

    def get_bounding_score(self, x_idx):
        # Get a score for which intermediate bound to tighten on layer x_idx (larger is better)
        scores = self.rhos[x_idx-1] if self.rhos[x_idx-1].dim() <= 3 else \
            self.rhos[x_idx-1].view(*self.rhos[x_idx-1].shape[:2], -1)
        lb_index = 0 if scores.shape[1] == 1 else 1
        return scores[:, lb_index]
