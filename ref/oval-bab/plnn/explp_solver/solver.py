import itertools
import torch
from torch import nn
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver import utils
from plnn.explp_solver import anderson_optimization, bigm_optimization, cut_anderson_optimization, \
    saddle_anderson_optimization
from time import time
from math import ceil


default_params = {
    "initial_eta": 100,
    "nb_inner_iter": 100,
    "nb_outer_iter": 10,
    "nb_iter": 500,
    "anderson_algorithm": "saddle",  # either saddle or prox
    "bigm": "init", # whether to use the specialized bigm relaxation solver. Alone: "only". As initializer: "init"
    "bigm_algorithm": "adam",  # either prox or adam
    "init_params": {
        'nb_outer_iter': 100,
        'initial_step_size': 1e-3,
        'final_step_size': 1e-6,
        'betas': (0.9, 0.999)
    }
}
eps_tol=0.01


class ExpLP(SaddleLP):

    def __init__(
            self,
            layers,
            debug=False,
            params=None,
            use_preactivation=True,
            store_bounds_progress=-1,
            fixed_M=False,
            view_tensorboard=False,
            store_bounds_primal=False,
            max_batch=20000
    ):
        """
        :param store_bounds_progress: which layer to store bounds progress for. -1=don't.
        :param fixed_M: whether to keep M (artificial upper bound) fixed for alpha and beta, rather than update it dynamically.
        :param use_preactivation: whether to strengthen constraints (2 out of exponentially many) by using pre-activation
        :param bigm: whether to use the specialized bigm relaxation solver. Alone, instead of Anderson: "only". As initializer: "init"
        """
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.debug = debug
        self.params = dict(default_params, **params) if params is not None else default_params
        self.use_preactivation = use_preactivation
        if self.params["bigm"] and self.params["bigm"] not in ["only", "init"]:
            raise IOError('bigm argument supports only False, "only", "init"')
        if self.params["bigm"] and self.params["bigm_algorithm"] not in ["prox", "adam"]:
            raise IOError('bigm_algorithm supports only False, "prox", "adam"')
        self.bigm_init = self.params["bigm"] and (self.params["bigm"] == "init")
        self.bigm_only = self.params["bigm"] and (self.params["bigm"] == "only")
        self.cut_init = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "init")
        self.cut_only = ("cut" in self.params) and self.params["cut"] and (self.params["cut"] == "only")
        self.max_batch = max_batch
        self.store_bounds_primal = store_bounds_primal
        self.external_init = None
        self.steps = self.get_iters()

        # whether to use only the planet constraints in the exponential problem (performs poorly)
        self.prox_dual_planet_only = False

        for param in self.net.parameters():
            param.requires_grad = False
        # Dummy placeholder for the trackers
        self.obj_tracker = None
        self.bound_tracker = None
        precision = torch.float
        for clayer in layers:
            if isinstance(clayer, (nn.Linear, nn.Conv2d)):
                precision = clayer.weight.dtype
        self.precision = precision
        self.fixed_M = fixed_M
        self.store_bounds_progress = store_bounds_progress
        self.view_tensorboard = view_tensorboard
        if self.view_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment="bigm_%s_cut_%s_i_%f_f_%f_cf_%d_c_%d" % (
            self.params["bigm"], self.params["cut"] , self.params['init_params']['initial_step_size'],
            self.params['init_params']['final_step_size'],
            self.params['cut_frequency'], self.params['max_cuts']))
        if self.bigm_only:
            if self.params["bigm_algorithm"] == "prox":
                self.optimize = self.bigm_prox_optimizer
                self.logger = utils.ProxOptimizationTrace()
            else:
                self.optimize = self.bigm_subgradient_optimizer
                self.logger = utils.OptimizationTrace()
        elif self.cut_only:
            self.optimize = self.cut_anderson_optimizer
            self.logger = utils.OptimizationTrace()
        else:
            # self.params["anderson_algorithm"] == "saddle"
            self.optimize = self.saddle_anderson_optimizer
            self.logger = utils.OptimizationTrace()

    def saddle_anderson_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        """
        Implementation of the "Saddle-Point Frank Wolfe" derivation for the full Anderson relaxation.
        """

        opt_args = {
            "nb_iter": 100,
            "alpha_M": 1e0,
            "beta_M": 1e0,
            "blockwise": False,
            "step_size_dict": {
              'type': 'grad',
              'fw_start': 100,
              'initial_step_size': 1e-2,
              'final_step_size': 1e-3
            },
            "bigm_algorithm": "adam",
            "init_params": {
                'nb_outer_iter': 800,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.1  # constant factor of the max big-m variables to use for M
            },
            "primal_init_params": {
                'nb_bigm_iter': 900,
                'nb_anderson_iter': 100,  # if the primal gets too close to the optimum, the blocks might get stuck
                'initial_step_size': 1e-2,  # 1e-2,
                'final_step_size': 1e-5,  # 1e-3,
                'betas': (0.9, 0.999)
            }
        }
        opt_args.update(self.params)

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs
        for cweight in weights:
            # CompositeLinearOp not supported by masked operators
            assert type(cweight) in [utils.LinearOp, utils.BatchLinearOp, utils.ConvOp, utils.BatchConvOp]

        if self.store_bounds_progress >= 0:
            self.logger.start_timing()

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        device = lower_bounds[-1].device
        input_size = tuple(lower_bounds[0].shape[1:])

        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1
        # Build the naive bounds
        nubs = [lin_k.interval_forward(cl_k, cu_k)[1] for (lin_k, cl_k, cu_k) in zip(weights, clbs, cubs)]  # 1 to n

        alpha_M = [opt_args["alpha_M"]] * len(clbs)
        beta_M = [opt_args["beta_M"]] * len(clbs)

        # Initialize dual variables to all 0s, primals to mid-boxes (replaced by other inits).
        dual_vars = saddle_anderson_optimization.SaddleDualVars.naive_initialization(
            weights, additional_coeffs, device, input_size, alpha_M=alpha_M, beta_M=beta_M)
        primal_vars = saddle_anderson_optimization.SaddlePrimalVars.mid_box_initialization(dual_vars, clbs, cubs)
        bound_init = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        # Initialize masked operators, and L_check/U_check.
        masked_ops, u_checks, l_checks = anderson_optimization.precompute_l_u_check(
            weights, dual_vars, primal_vars, clbs, cubs)

        if self.external_init is not None and type(self.external_init) in \
                [saddle_anderson_optimization.SaddleAndersonInit, bigm_optimization.BigMPInit]:

            # Initialise bigm's duals from parent (then run supergradient ascent on them).
            bound_init = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            dual_vars, _ = saddle_anderson_optimization.SaddleDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, clbs, cubs, lower_bounds, upper_bounds,
                bigm_M_factor=opt_args["init_params"]["M_factor"])
            # Initialise primals from parent, without any subgradient descent.
            primal_vars = self.external_init.primals

        elif self.bigm_init and not self.cut_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            if opt_args["bigm_algorithm"] == "prox":
                bounds = self.bigm_prox_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            else:
                # opt_args["bigm_algorithm"] == "adam"
                bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            print(f"Average bounds after bigm init: {bounds.mean().item()}")

            # this function initializes the M values from big-M as well
            dual_vars, primal_vars = saddle_anderson_optimization.SaddleDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, clbs, cubs, lower_bounds, upper_bounds,
                bigm_M_factor=opt_args["init_params"]["M_factor"])
            primal_vars = saddle_anderson_optimization.primals_subgrad_initializer(
                weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, lower_bounds, upper_bounds, dual_vars,
                saddle_anderson_optimization.SaddlePrimalVars, opt_args["primal_init_params"])
            bound_init = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)

            if opt_args["bigm_algorithm"] == "prox":
                xt, zt = self.bigm_primal_vars
                primal_vars = saddle_anderson_optimization.SaddlePrimalVars(xt, zt)
        elif self.cut_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            bounds = self.cut_anderson_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            print(f"Average bounds after cut init: {bounds.mean().item()}")

            # this function initializes the M values from big-M as well
            dual_vars, primal_vars = saddle_anderson_optimization.SaddleDualVars.cut_initialization(
                self.cut_dual_vars, self.cut_primal_vars, cut_M_factor=opt_args["init_params"]["M_factor"])
            self.cut_dual_vars = []; self.cut_primal_vars = []
            primal_vars = saddle_anderson_optimization.primals_subgrad_initializer(
                weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, lower_bounds, upper_bounds, dual_vars,
                saddle_anderson_optimization.SaddlePrimalVars, opt_args["primal_init_params"])
            bound_init = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)

        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        print(f"Average bound at initialisation: {bound_init.mean().item()}")
        torch.max(best_bound, bound_init, out=best_bound)

        obj_gap = saddle_anderson_optimization.compute_primal_dual_gap(
            additional_coeffs, weights, masked_ops, nubs, l_checks, u_checks, lower_bounds, upper_bounds, primal_vars,
            dual_vars, bound_init, precision=torch.float)
        print(f"Average obj at initialization: {obj_gap.mean().item()}")

        for steps in itertools.count():
            if steps > opt_args["nb_iter"]:
                break

            # Each inner_iteration will consist of a forward-backward
            # iteration through the network
            nb_relu_layers = len(dual_vars.fs)
            lay_to_solve = itertools.chain(range(1, nb_relu_layers), range(nb_relu_layers - 1, 0, -1))

            if opt_args["blockwise"]:
                # Perform (in Gauss-Seidel style) the step over primal and dual variables. The steps are performed in
                # primal-dual blocks of the form (duals_k, primals_k, primals_{k-1}).
                for lay_idx in lay_to_solve:
                    # Compute the SP-FW update for block duals and primals at once.
                    saddle_anderson_optimization.spfw_block_k_step(
                        lay_idx, weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, lower_bounds, upper_bounds,
                        dual_vars,  primal_vars, steps, opt_args["step_size_dict"], precision=torch.float,
                        fixed_M=self.fixed_M, max_iters=opt_args["nb_iter"])
            else:
                # Perform the step over all primal and dual variables at once.
                saddle_anderson_optimization.spfw_full_step(
                    weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, lower_bounds, upper_bounds, dual_vars,
                    primal_vars, steps, opt_args["step_size_dict"], precision=torch.float, fixed_M=self.fixed_M,
                    max_iters=opt_args["nb_iter"])

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                if steps % 250 == 0:
                    start_logging_time = time()
                    c_bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                    torch.max(best_bound, c_bound, out=best_bound)
                    obj_gap = saddle_anderson_optimization.compute_primal_dual_gap(
                        additional_coeffs, weights, masked_ops, nubs, l_checks, u_checks, lower_bounds, upper_bounds,
                        primal_vars, dual_vars, c_bound, precision=torch.float)
                    logging_time = time() - start_logging_time
                    self.logger.add_exp_proximal_point(len(weights), best_bound.clone(), c_bound, obj_gap,
                                                       logging_time=logging_time)

            if not self.store_bounds_primal:
                bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"Average best bound: {best_bound.mean().item()}")
                print(f"Average bound: {bound.mean().item()}")

        self.children_init = saddle_anderson_optimization.SaddleAndersonInit(self.bigm_dual_vars, primal_vars)

        if self.store_bounds_primal:
            self.bounds_primal = primal_vars
            bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
            torch.max(best_bound, bound, out=best_bound)
            nb_neurons = int(best_bound.shape[1]/2)
            print(f"Average LB improvement: {(best_bound - bound_init)[:, nb_neurons:].mean()}")

        return best_bound

    def cut_anderson_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space
        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            "nb_iter": 100,
            "alpha_M": 1e-3,
            "beta_M": 1e-3,
            "bigm_algorithm": "adam",
            ###
            'random_cuts': False,
            'cut_frequency': 400,
            'max_cuts': 10,
            'cut_add': 2,
            ###
            'betas': (0.9, 0.999),
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            "init_params": {
                'nb_outer_iter': 100,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'M_factor': 1.0
            },
            'nb_outer_iter': 10,
            'bigm': 'init',
            'restrict_factor': -1,  # If != -1, the fraction of the convolutional channels to use
        }
        if self.cut_only:
            opt_args.update(self.params)
        else:
            opt_args.update(self.params["cut_init_params"])

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs  # if the last layer's coefficients are present
        for cweight in weights:
            # CompositeLinearOp not supported by masked operators
            assert type(cweight) in [utils.LinearOp, utils.BatchLinearOp, utils.ConvOp, utils.BatchConvOp]
        self.logger.start_timing()

        device = lower_bounds[-1].device
        input_size = tuple(lower_bounds[0].shape[1:])
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        # Build the naive bounds
        nubs = [lin_k.interval_forward(cl_k, cu_k)[1] for (lin_k, cl_k, cu_k) in zip(weights, clbs, cubs)]  # 1 to n

        if self.external_init is not None and type(self.external_init) is bigm_optimization.BigMPInit:

            # Initialise bigm's duals from parent (then run supergradient ascent on them).
            bound_init = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
            dual_vars, _ = cut_anderson_optimization.CutDualVars.bigm_initialization(
                self.bigm_dual_vars, weights, additional_coeffs, device, tuple(lower_bounds[0].shape[1:]), clbs, cubs,
                lower_bounds, upper_bounds, opt_args)
            # Initialise primals from parent, without any subgradient descent.
            primal_vars = cut_anderson_optimization.CutPrimalVars.mid_box_initialization(dual_vars, clbs, cubs)

            ## Adam-related quantities.
            adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
            adam_stats.bigm_adam_initialization(dual_vars.sum_beta, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                                beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

        elif self.bigm_init:
            # Initialize alpha/beta with the output of the chosen big-m solver.
            if opt_args["bigm_algorithm"] == "prox":
                bounds = self.bigm_prox_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)

                print(f"Average bounds after init with Bigm prox: {bounds.mean().item()}")

                dual_vars, primal_vars = cut_anderson_optimization.CutDualVars.bigm_initialization(
                    self.bigm_dual_vars, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                    upper_bounds, opt_args)

                ## Adam-related quantities.
                adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                     beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

            else:
                # opt_args["bigm_algorithm"] == "adam"
                bounds = self.bigm_subgradient_optimizer(weights, additional_coeffs, lower_bounds, upper_bounds)
                print(f"Average bounds after init with Bigm adam: {bounds.mean().item()}")

                dual_vars, primal_vars = cut_anderson_optimization.CutDualVars.bigm_initialization(
                    self.bigm_dual_vars, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                    upper_bounds, opt_args)

                ## Adam-related quantities.
                adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                     beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))
                adam_stats.bigm_adam_initialization(dual_vars.sum_beta, self.bigm_adam_stats, beta1=opt_args['betas'][0],
                                                    beta2=opt_args['betas'][1])
                # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

            if opt_args["bigm_algorithm"] == "prox":
                xt, zt = self.bigm_primal_vars
                primal_vars = cut_anderson_optimization.CutPrimalVars(xt, zt)
        else:
            # Initialize dual variables to all 0s, primals to mid-boxes.
            print('it is doing naive initialisation')
            dual_vars = cut_anderson_optimization.CutDualVars.naive_initialization(weights, additional_coeffs, device,
                                                                                   input_size)
            primal_vars = cut_anderson_optimization.CutPrimalVars.mid_box_initialization(dual_vars, clbs, cubs)

            ## Adam-related quantities.
            adam_stats = cut_anderson_optimization.DualADAMStats(dual_vars.sum_beta, beta1=opt_args['betas'][0],
                                                                 beta2=opt_args['betas'][1])
            # initializes adam stats(momentum 1 and momentum 2 for dual variables(alpha,beta_0,beta_1))

        # TODO: make this a function
        if opt_args["restrict_factor"] != -1:
            # Compute intercept scores, setting them to 0 for non-ambiguous neurons.
            intercept_scores = []
            for lay, (clb, cub) in enumerate(zip(lower_bounds, upper_bounds)):
                bias = - (clb * cub) / (cub - clb)
                bias.masked_fill_(clb >= 0, 0)
                bias.masked_fill_(cub <= 0, 0)
                bias.masked_fill_(clb == cub, 0)
                if lay > 0 and lay < len(clbs) - 1:
                    cscore = (bias.unsqueeze(1) * dual_vars.sum_beta[lay])  # A bit better than CROWN
                else:
                    cscore = torch.zeros((batch_size + cub.shape[1:]), device=cub.device, dtype=cub.dtype)
                intercept_scores.append(cscore)

            # Set Anderson neurons.
            # For CNNs, choosing topk channels and spatial dimensions
            and_set = [[] for lay in range(1, len(clbs)-1)]
            k = 1e30  # all ambiguous of a layer
            for lay in range(1, len(clbs) - 1):
                if len(upper_bounds[lay].shape) > 3:
                    # Handle convolution explictly.
                    cscores = intercept_scores[lay]
                    n_channels = upper_bounds[lay].shape[1]
                    # Number of neurons that are ambiguous at least once across channels.
                    ambiguous_relus = ((upper_bounds[lay] > 0) & (lower_bounds[lay] < 0)).view(
                        batch_size[0], n_channels, -1).max(0)[0]
                    # Top channels.
                    n_chan_ambiguous = ambiguous_relus.max(-1)[0].sum()
                    ck = min(int(ceil(n_chan_ambiguous.item()/opt_args["restrict_factor"])), k)
                    print(f"channel current k {ck}/{ambiguous_relus.shape[-2]}")
                    and_set_chan = torch.topk(cscores.view(batch_size + (n_channels, -1)).mean(-1), ck)[1].sort(-1)[0]
                    # Top spatial dimensions.
                    n_spat_ambiguous = ambiguous_relus.max(-2)[0].sum()
                    ck = min(int(ceil(n_spat_ambiguous.item() / opt_args["restrict_factor"])), k)
                    print(f"spatial current k {ck}/{ambiguous_relus.shape[-1]}")
                    gathered_scores = cscores.view(batch_size + (n_channels, -1)).gather(
                        -2, and_set_chan.unsqueeze(-1).expand((and_set_chan.shape + (ambiguous_relus.shape[-1],))))
                    and_set_spat = torch.topk(gathered_scores.mean(-2), ck)[1].sort(-1)[0]
                    # Linearize the indexing for faster gather/scatter.
                    and_set_both = (and_set_chan.unsqueeze(-1) * ambiguous_relus.shape[-1] +
                                    and_set_spat.unsqueeze(-2)).view(*batch_size, -1).sort(-1)[0]
                    and_set[lay - 1] = (and_set_chan, and_set_spat, and_set_both)

                else:
                    cscores = intercept_scores[lay]
                    n_ambiguous = \
                        ((upper_bounds[lay] > 0) & (lower_bounds[lay] < 0)).view(batch_size[0], -1).sum(-1).max(0)[0]
                    ck = min(n_ambiguous.item(), k)
                    and_set[lay-1] = torch.topk(cscores, ck)[1]

            dual_vars.and_set = and_set

        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        bound_init = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        print(f"Average bound at initialisation: {bound_init.mean().item()}")
        torch.max(best_bound, bound_init, out=best_bound)

        # Initialize masked operators, and L_check/U_check.
        masked_ops, u_checks, l_checks = anderson_optimization.precompute_l_u_check(
            weights, dual_vars, primal_vars, clbs, cubs)

        if self.debug:
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                       self.params["init_params"]["nb_outer_iter"])
                self.writer.add_scalar('Average bound', -bound_init.mean().item(), self.params["init_params"]["nb_outer_iter"])

        init_step_size = opt_args['initial_step_size']
        final_step_size = opt_args['final_step_size']

        for steps in itertools.count():
            if steps >= opt_args["nb_iter"]:
                break

            step_size = init_step_size + ((steps + 1) / opt_args["nb_iter"]) * (final_step_size - init_step_size)
            # do adam for subgradient ascent
            dual_vars_subg_updated = adam_stats.active_set_adam_step(
                weights, masked_ops, step_size, steps, dual_vars, primal_vars, clbs, cubs, nubs, lower_bounds,
                upper_bounds, l_checks, u_checks, opt_args, precision=self.precision)
            dual_vars.update_from_step(weights, dual_vars_subg_updated)

            if self.debug:
                bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"{steps} Average best bound: {best_bound.mean().item()}")
                print(f"{steps} Average bound: {bound.mean().item()}")
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)
                    self.writer.add_scalar('Average bound', -bound.mean().item(),
                                           self.params["init_params"]["nb_outer_iter"] + steps + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                if steps % 10 == 0:
                    start_logging_time = time()
                    bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                    torch.max(best_bound, bound, out=best_bound)
                    logging_time = time() - start_logging_time
                    self.logger.add_point(len(weights), best_bound.clone(), logging_time=logging_time)

            if not self.store_bounds_primal:
                bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
                torch.max(best_bound, bound, out=best_bound)
                print(f"Average best bound: {best_bound.mean().item()}")
                print(f"Average bound: {bound.mean().item()}")

        if self.cut_init:
            # store the dual vars and primal vars for possible future usage
            self.cut_dual_vars = dual_vars
            self.cut_primal_vars = primal_vars

        self.children_init = bigm_optimization.BigMPInit(self.bigm_dual_vars)
        
        if self.store_bounds_primal:
            self.bounds_primal = primal_vars  # store the last matching primal
            bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
            torch.max(best_bound, bound, out=best_bound)
            nb_neurons = int(best_bound.shape[1]/2)
            print(f"Average LB improvement: {(best_bound - bound_init)[:, nb_neurons:].mean()}")

        bound = anderson_optimization.compute_bounds(dual_vars, weights, clbs, cubs)
        return bound

    def bigm_subgradient_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # ADAM subgradient ascent for a specific big-M solver which operates directly in the dual space

        # hard-code default parameters, which are overridden by self.params
        opt_args = {
            'nb_outer_iter': 100,
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            'betas': (0.9, 0.999),
            # Useful within BaB, where entries with naive init might need a larger initial step size
            'larger_irl_if_naive_init': False
        }
        if self.bigm_init or self.cut_init:
            opt_args.update(self.params['init_params'])
        else:
            opt_args.update(self.params)

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        if self.store_bounds_progress >= 0 and self.bigm_only:
            self.logger.start_timing()

        device = lower_bounds[-1].device
        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        if self.external_init is not None and type(self.external_init) in [bigm_optimization.BigMPInit,
            saddle_anderson_optimization.SaddleAndersonInit]:

            pinit = True
            dual_vars = self.external_init.duals
            bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Bounds pinit: {bound.mean()}")
            init_step_size = opt_args['initial_step_size']
        else:
            pinit = False
            dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                        lower_bounds[0].shape[1:])
            if opt_args["larger_irl_if_naive_init"]:
                init_step_size = opt_args['initial_step_size'] * 10

        # Adam-related quantities.
        adam_stats = bigm_optimization.DualADAMStats(dual_vars.beta_0, beta1=opt_args['betas'][0], beta2=opt_args['betas'][1])
        final_step_size = opt_args['final_step_size']

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        if self.debug:
            obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
            print(f"Average bound (and objective, they concide) at naive initialisation: {obj_val.mean().item()}")
            torch.max(best_bound, obj_val, out=best_bound)
            if self.view_tensorboard:
                self.writer.add_scalar('Average best bound', -best_bound.mean().item(), 0)
                self.writer.add_scalar('Average bound', -obj_val.mean().item(), 0)

        n_outer_iters = opt_args["nb_outer_iter"]
        for outer_it in itertools.count():
            if outer_it > n_outer_iters:
                break

            if self.debug:
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            dual_vars_subg = bigm_optimization.compute_dual_subgradient(
                weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)

            step_size = init_step_size + ((outer_it + 1) / n_outer_iters) * (final_step_size - init_step_size)

            # normal subgradient ascent
            # dual_vars.projected_linear_combination(
            #     step_size, dual_vars_subg, weights)

            # do adam for subgradient ascent
            adam_stats.update_moments_take_projected_step(
                weights, step_size, outer_it, dual_vars, dual_vars_subg)

            dual_vars.update_f_g(lower_bounds, upper_bounds)

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                print(f"Average obj at the end of adam iteration {outer_it}: {obj_val.mean().item()}")
                torch.max(best_bound, obj_val, out=best_bound)
                if self.view_tensorboard:
                    self.writer.add_scalar('Average best bound', -best_bound.mean().item(), outer_it + 1)
                    self.writer.add_scalar('Average bound', -obj_val.mean().item(), outer_it + 1)

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress and self.bigm_only:
                if outer_it % 10 == 0:
                    start_logging_time = time()
                    bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
                    logging_time = time() - start_logging_time
                    self.logger.add_point(len(weights), bound, logging_time=logging_time)

        # store the dual vars and primal vars for possible future usage
        self.bigm_dual_vars = dual_vars
        self.bigm_primal_vars = None
        self.bigm_adam_stats = adam_stats

        if self.bigm_only:
            self.children_init = bigm_optimization.BigMPInit(dual_vars)
            if self.store_bounds_primal:
                # Compute last matching primal (could make it a function...)
                nb_relu_layers = len(dual_vars.beta_0)
                xkm1, _ = bigm_optimization.layer_primal_linear_minimization(0, dual_vars.fs[0], None, clbs[0], cubs[0])
                zt = []
                xt = [xkm1]
                for lay_idx in range(1, nb_relu_layers):
                    # solve the inner problems.
                    xk, zk = bigm_optimization.layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    xt.append(xk)
                    zt.append(zk)
                self.bounds_primal = anderson_optimization.PrimalVars(xt, zt)

        bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds)
        return bound

    def bigm_prox_optimizer(self, weights, additional_coeffs, lower_bounds, upper_bounds):
        # Proximal maximization for a specific big-M solver which operates directly in the dual space

        opt_args = {
            'initial_eta': 2e2,
            'final_eta': None,
            "nb_inner_iter": 5,
            "nb_outer_iter": 100,
        }
        if not self.bigm_init:
            opt_args.update(self.params)
        else:
            opt_args.update(self.params['init_params'])

        # TODO: at the moment we don't support optimization with objective
        # function on something else than the last layer (although it could be
        # adapted) similarly to the way it was done in the proxlp_solver. For
        # now, just assert that we're in the expected case.
        assert len(additional_coeffs) == 1
        assert len(weights) in additional_coeffs

        if self.store_bounds_progress >= 0 and self.bigm_only:
            self.logger.start_timing()

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]
        device = lower_bounds[-1].device

        self.eta = opt_args["initial_eta"]

        dual_vars = bigm_optimization.DualVars.naive_initialization(weights, additional_coeffs, device,
                                                                    lower_bounds[0].shape[1:])

        # Build the clamped bounds
        clbs = [lower_bounds[0]] + [torch.clamp(bound, 0, None) for bound in lower_bounds[1:]]  # 0 to n-1
        cubs = [upper_bounds[0]] + [torch.clamp(bound, 0, None) for bound in upper_bounds[1:]]  # 0 to n-1

        xhatt = []  # Indexed from 1 to n-1 (pre-activation version of xt)
        # Build the variable holders
        xt = []  # Indexed from 0 to n-1
        zt = []  # Indexed from 1 to n-1

        y = bigm_optimization.YVars.initialization_init(dual_vars)

        for lay_idx, layer in enumerate(weights):
            xk, zk = bigm_optimization.layer_primal_linear_minimization(
                lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx-1], clbs[lay_idx], cubs[lay_idx])
            xt.append(xk)
            if lay_idx > 0:
                zt.append(zk)
            if lay_idx < len(weights)-1:
                xhatt.append(layer.forward(xk))

        best_bound = -float("inf") * torch.ones(batch_size, device=device, dtype=self.precision)

        anchor_dual_vars = dual_vars.copy()
        dual_vars.update_from_anchor_points(
            anchor_dual_vars, xt, zt, xhatt, y, self.eta, weights, lower_bounds, upper_bounds)
        dual_vars.update_f_g(lower_bounds, upper_bounds)

        if self.debug:
            obj_val = bigm_optimization.compute_prox_obj(xt, zt, xhatt, y, self.eta, anchor_dual_vars, weights,
                                                         lower_bounds, upper_bounds)
            print(f"Initial objective: {obj_val.mean().item()}")
            bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds, prox=True)
            print(f"Average bound at naive initialisation: {bound.mean().item()}")

        outer_iters = opt_args["nb_outer_iter"]
        inner_iters = opt_args["nb_inner_iter"]
        nb_total_steps = outer_iters * inner_iters
        steps = 0
        for outer_it in itertools.count():
            if outer_it > outer_iters:
                break

            if opt_args['final_eta'] is not None:
                self.eta = opt_args['initial_eta'] + (steps / nb_total_steps) * (opt_args['final_eta'] - opt_args['initial_eta'])

            for inner_iter in range(inner_iters):
                # each inner iter is a forward/backward pass over the net
                nb_relu_layers = len(dual_vars.beta_0)
                lay_to_solve = itertools.chain(range(nb_relu_layers), range(nb_relu_layers - 1, -1, -1))
                for lay_idx in lay_to_solve:
                    # do block-coordinate descent on (x,t) and y. A step of FW and a step of NNMP, respectively.

                    # ------- x, z optimization
                    # do a FW step on x,z (with optimal step size)
                    bigm_optimization.primal_vars_do_fw_step(
                        lay_idx, dual_vars, anchor_dual_vars, self.eta, xt, zt, xhatt, y, weights, additional_coeffs,
                        clbs, cubs, lower_bounds, upper_bounds)

                    # update the closed-form dual variables
                    duals_to_update = []
                    fg_to_update = [lay_idx]
                    if lay_idx < nb_relu_layers-1:
                        duals_to_update.append(lay_idx + 1)
                        fg_to_update.append(lay_idx + 1)
                    if lay_idx > 0:
                        duals_to_update.append(lay_idx)
                        fg_to_update.append(lay_idx-1)
                    dual_vars.update_from_anchor_points(
                        anchor_dual_vars, xt, zt, xhatt, y, self.eta, weights, lower_bounds, upper_bounds,
                        lay_idx=duals_to_update)

                    # ------- y optimization
                    if lay_idx > 0:
                        # do a NNMP step on y (with optimal step size)
                        y.do_nnmp_step_layer(lay_idx, dual_vars, anchor_dual_vars, self.eta, xt, zt, xhatt,
                                             lower_bounds, upper_bounds, precision=self.precision)

                        # update the closed-form dual variables
                        dual_vars.update_from_anchor_points(
                            anchor_dual_vars, xt, zt, xhatt, y, self.eta, weights, lower_bounds, upper_bounds, lay_idx=lay_idx)
                    dual_vars.update_f_g(lower_bounds, upper_bounds, lay_idx=fg_to_update)
                steps += 1

            if self.debug:
                # This is the value "at convergence" of the dual problem
                obj_val = bigm_optimization.compute_prox_obj(xt, zt, xhatt, y, self.eta, anchor_dual_vars, weights,
                                                             lower_bounds, upper_bounds)
                print(f"Average obj at the end of proximal iteration {outer_it}: {obj_val.mean().item()}")

            # Update the dual variables anchors, and their functions.
            anchor_dual_vars = dual_vars.copy()

            bound = bigm_optimization.compute_bounds(weights, dual_vars, clbs, cubs, lower_bounds, upper_bounds, prox=True)
            torch.max(best_bound, bound, out=best_bound)
            # print(f"Average bound: {best_bound.mean().item()}")

            if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress and self.bigm_only:
                if outer_it % 5 == 0:
                    start_logging_time = time()
                    obj_val = bigm_optimization.compute_prox_obj(xt, zt, xhatt, y, self.eta, anchor_dual_vars,
                                                                 weights, lower_bounds, upper_bounds)
                    logging_time = time() - start_logging_time
                    self.logger.add_proximal_point(len(weights), bound, obj_val, logging_time=logging_time)

        # store the dual vars and primal vars for possible future usage
        self.bigm_dual_vars = dual_vars
        self.bigm_primal_vars = (xt, zt)

        return best_bound

    def get_lower_bound_network_input(self):
        """
        Return the input of the network that was used in the last bounds computation.
        Converts back from the conditioned input domain to the original one.
        Assumes that the last layer is a single neuron.
        """
        assert self.store_bounds_primal
        assert self.bounds_primal.xt[0].shape[1] in [1, 2], "the last layer must have a single neuron"
        l_0 = self.input_domain.select(-1, 0)
        u_0 = self.input_domain.select(-1, 1)
        net_input = (1/2) * (u_0 - l_0) * self.bounds_primal.xt[0].select(1, self.bounds_primal.xt[0].shape[1]-1) +\
                    (1/2) * (u_0 + l_0)
        return utils.apply_transforms(self.input_transforms, net_input, inverse=True)

    def initialize_from(self, external_init):
        # setter to initialise from an external list of dual/primal variables (instance of AndersonPInit)
        self.external_init = external_init

    def internal_init(self):
        self.external_init = None

    # BaB-related method to implement automatic no. of iters.
    def set_iters(self, iters):
        if self.cut_only:
            self.params["nb_iter"] = iters
        elif self.bigm_only:
            self.params["nb_outer_iter"] = iters
        else:
            # SP-FW
            self.params["nb_iter"] = iters
        self.steps = iters

    # BaB-related method to implement automatic no. of iters.
    def get_iters(self):
        if self.cut_only:
            return self.params["nb_iter"]
        elif self.bigm_only:
            return self.params["nb_outer_iter"]
        else:
            # SP-FW
            return self.params["nb_iter"]

    # BaB-related method to implement automatic no. of iters.
    def default_iters(self, set_min=False):
        # Set no. of iters to default for the algorithm this class represents.
        if self.cut_only:
            self.min_steps = 100
            self.max_steps = 2500
            self.step_increase = 400
        elif self.bigm_only:
            self.min_steps = 100
            self.max_steps = 1100
            self.step_increase = 100
        else:
            # SP-FW
            self.min_steps = 1000
            self.max_steps = 4000
            self.step_increase = 1000
        if set_min and self.get_iters() != -1:
            self.min_steps = self.get_iters()
        self.set_iters(self.min_steps)
