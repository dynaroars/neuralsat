import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from plnn_bounds.network_linear_approximation import LinearizedNetwork
from plnn_bounds.proxlp_solver.utils import LinearOp, ConvOp, prod, OptimizationTrace, ProxOptimizationTrace, bdot
from plnn_bounds.proxlp_solver.by_pairs import ByPairsDecomposition, DualVarSet


class SaddleLP(LinearizedNetwork):
    """
        Class implementing the dual iterative methods for neural network bounds presented in "Lagrangian Decomposition
        for Neural Network Verification", https://arxiv.org/abs/2002.10410.
    """
    def __init__(self, layers, store_bounds_progress=-1):
        self.optimizers = {
            'init': self.init_optimizer,
            'adam': self.adam_subg_optimizer,
            'autograd': self.autograd_optimizer,
            'subgradient': self.subgradient_optimizer,
            'prox': self.prox_optimizer,
            'optimized_prox': self.optimized_prox_optimizer,
            'comparison': self.comparison_optimizer,
            'best_naive_kw': self.best_naive_kw_optimizer
        }

        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False

        self.decomposition = ByPairsDecomposition('KW')
        self.optimize, _ = self.init_optimizer(None)

        self.store_bounds_progress = store_bounds_progress

    def set_initialisation(self, init_type):
        assert init_type in ["naive", "KW"]
        self.decomposition = ByPairsDecomposition(init_type)

    def set_solution_optimizer(self, method, method_args=None):
        assert method in self.optimizers
        self.optimize, self.logger = self.optimizers[method](method_args)

    @staticmethod
    def build_first_conditioned_layer(l_0, u_0, layer, no_conv=False):
        w_1 = layer.weight
        b_1 = layer.bias

        pos_w1 = torch.clamp(w_1, 0, None)
        neg_w1 = torch.clamp(w_1, None, 0)

        if isinstance(layer, nn.Linear):
            l_1 = pos_w1 @ l_0 + neg_w1 @ u_0 + b_1
            u_1 = pos_w1 @ u_0 + neg_w1 @ l_0 + b_1

            # Build the "conditioned" first layer
            range_0 = (u_0 - l_0)

            # range_1 = (u_1 - l_1)
            # cond_w_1 = (1/range_1).unsqueeze(1) * w_1 * range_0
            # cond_b_1 = (1/range_1) * (2 * b_1 - (u_1 + l_1) + w_1 @ (u_0 + l_0))
            cond_w_1 = w_1 * 0.5 * range_0
            cond_b_1 = b_1 + 0.5 * w_1 @ (u_0 + l_0)

            cond_layer = LinearOp(cond_w_1, cond_b_1)
        elif isinstance(layer, nn.Conv2d):
            l_1 = (F.conv2d(l_0.unsqueeze(0), pos_w1, b_1,
                            layer.stride, layer.padding,
                            layer.dilation, layer.groups)
                   + F.conv2d(u_0.unsqueeze(0), neg_w1, None,
                              layer.stride, layer.padding,
                              layer.dilation, layer.groups)).squeeze(0)
            u_1 = (F.conv2d(u_0.unsqueeze(0), pos_w1, b_1,
                            layer.stride, layer.padding,
                            layer.dilation, layer.groups)
                   + F.conv2d(l_0.unsqueeze(0), neg_w1, None,
                              layer.stride, layer.padding,
                              layer.dilation, layer.groups)).squeeze(0)

            range_0 = (u_0 - l_0)/2
            out_bias = F.conv2d((u_0 + l_0).unsqueeze(0) / 2, w_1, b_1,
                                layer.stride, layer.padding,
                                layer.dilation, layer.groups).squeeze(0)

            cond_layer = ConvOp(w_1, out_bias,
                                layer.stride, layer.padding,
                                layer.dilation, layer.groups)
            cond_layer.add_prerescaling(range_0)

            if no_conv:
                cond_layer = cond_layer.equivalent_linear(l_0)
        return l_1, u_1, cond_layer

    @staticmethod
    def build_obj_layer(prev_ub, layer, no_conv=False, orig_shape_prev_ub=None):
        w_kp1 = layer.weight
        b_kp1 = layer.bias

        obj_layer_orig = None
        if isinstance(layer, nn.Conv2d):
            obj_layer = ConvOp(w_kp1, b_kp1,
                               layer.stride, layer.padding,
                               layer.dilation, layer.groups)
            if no_conv:
                obj_layer_orig = obj_layer
                obj_layer = obj_layer.equivalent_linear(orig_shape_prev_ub)
        else:
            obj_layer = LinearOp(w_kp1, b_kp1)

        if isinstance(obj_layer, LinearOp) and (prev_ub.dim() > 1):
            # This is the first LinearOp,
            # We need to include the flattening
            obj_layer.flatten_from(prev_ub.shape)

        return obj_layer, obj_layer_orig

    def compute_lower_bound(self, node=(-1, None), upper_bound=False,
                            all_optim=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        all_optim: Should the bounds be computed only in the case where they are not already leading to
              non relaxed version. This option is only useful if the batch mode based on None in node is
              used.
        '''
        additional_coeffs = {}
        current_lbs = self.lower_bounds[node[0]]
        current_ubs = self.upper_bounds[node[0]]
        if current_lbs.dim() == 0:
            current_lbs = current_lbs.unsqueeze(0)
            current_ubs = current_ubs.unsqueeze(0)
        node_layer_shape = current_lbs.shape
        self.opt_time_per_layer = []

        lay_to_opt = len(self.lower_bounds)+node[0] if node[0] < 0 else node[0]
        is_batch = (node[1] is None)
        if is_batch:
            # Optimize all the bounds
            nb_out = prod(node_layer_shape)
            opt_to_bound = torch.eye(nb_out, device=current_lbs.device).type(current_lbs.type())

            if not all_optim:
                ubs_to_opt = (current_ubs > 0)
                lbs_to_opt = (current_lbs < 0)

                # Only one needing optim are the ambiguous ones
                to_opt = ubs_to_opt & lbs_to_opt
                opt_to_bound = opt_to_bound[to_opt.view(-1)]

            nb_nodes = opt_to_bound.size(0)
            coeffs = opt_to_bound.view((nb_nodes,) + current_lbs.shape)
            coeffs = torch.cat([-coeffs, coeffs], 0)
        else:
            coeffs = torch.zeros_like(current_lbs).unsqueeze(0)
            if coeffs.dim() == 2:
                coeffs[0, node[1]] = -1 if upper_bound else 1
            elif coeffs.dim() == 4:
                coeffs[0, node[1][0], node[1][1], node[1][2]] = -1 if upper_bound else 1
            else:
                raise NotImplementedError

        nb_targets = coeffs.size(0)
        if nb_targets == 0:
            return current_lbs, current_ubs
        else:
            additional_coeffs[lay_to_opt] = coeffs

            start_opt_time = time.time()
            bound = self.optimize(self.weights, additional_coeffs,
                                  self.lower_bounds, self.upper_bounds)
            end_opt_time = time.time()
            self.opt_time_per_layer.append(end_opt_time - start_opt_time)
            if not is_batch:
                if upper_bound:
                    bound = -bound
                return bound
            else:
                opted_ubs = -bound[:nb_nodes]
                opted_lbs = bound[nb_nodes:]
                if all_optim:
                    ubs = opted_ubs.view(node_layer_shape)
                    lbs = opted_lbs.view(node_layer_shape)
                else:
                    ubs = torch.where(to_opt, (opted_ubs @ opt_to_bound).view(node_layer_shape), current_ubs)
                    lbs = torch.where(to_opt, (opted_lbs @ opt_to_bound).view(node_layer_shape), current_lbs)
                assert (ubs - lbs).min() >= 0, "Incompatible bounds"

                return lbs, ubs

    def define_linear_approximation(self, input_domain, force_optim=False, no_conv=False, override_numerical_errors=False):
        '''
        no_conv is an option to operate only on linear layers, by transforming all
        the convolutional layers into equivalent linear layers.
        '''
        self.no_conv = no_conv
        # Setup the bounds on the inputs
        self.input_domain = input_domain
        self.opt_time_per_layer = []
        l_0 = input_domain.select(-1, 0)
        u_0 = input_domain.select(-1, 1)

        next_is_linear = True
        for lay_idx, layer in enumerate(self.layers):
            if lay_idx == 0:
                assert next_is_linear
                next_is_linear = False
                l_1, u_1, cond_first_linear = self.build_first_conditioned_layer(
                    l_0, u_0, layer, no_conv)

                if no_conv:
                    # when linearizing conv layers, we need to keep track of the original shape of the bounds
                    self.original_shape_lbs = [-torch.ones_like(l_0), l_1]
                    self.original_shape_ubs = [torch.ones_like(u_0), u_1]
                    l_0 = l_0.view(-1)
                    u_0 = u_0.view(-1)
                    l_1 = l_1.view(-1)
                    u_1 = u_1.view(-1)
                self.lower_bounds = [-torch.ones_like(l_0), l_1]
                self.upper_bounds = [torch.ones_like(u_0), u_1]
                weights = [cond_first_linear]
            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                all_optim = force_optim or lay_idx==(len(self.layers)-1)

                orig_shape_prev_ub = self.original_shape_ubs[-1] if no_conv else None
                obj_layer, obj_layer_orig = self.build_obj_layer(self.upper_bounds[-1], layer, no_conv,
                                                 orig_shape_prev_ub=orig_shape_prev_ub)
                weights.append(obj_layer)

                layer_opt_start_time = time.time()
                l_kp1, u_kp1 = self.solve_problem(weights,
                                                  self.lower_bounds, self.upper_bounds,
                                                  all_optim, override_numerical_errors=override_numerical_errors)
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"[PROX] Time used for layer {lay_idx}: {time_used}")
                self.opt_time_per_layer.append(layer_opt_end_time - layer_opt_start_time)

                if no_conv:
                    if isinstance(layer, nn.Conv2d):
                        self.original_shape_lbs.append(
                            l_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_lbs[-1].unsqueeze(0).shape)).
                            squeeze(0)
                        )
                        self.original_shape_ubs.append(
                            u_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_ubs[-1].unsqueeze(0).shape)).
                            squeeze(0)
                        )
                    else:
                        self.original_shape_lbs.append(l_kp1)
                        self.original_shape_ubs.append(u_kp1)
                self.lower_bounds.append(l_kp1)
                self.upper_bounds.append(u_kp1)
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass
        self.weights = weights

    def build_model_using_bounds(self, domain, intermediate_bounds, no_conv=False):
        """
        Build the model from the provided intermediate bounds.
        If no_conv is true, convolutional layers are treated as their equivalent linear layers. In that case,
        provided intermediate bounds should retain the convolutional structure.
        """
        self.no_conv = no_conv
        self.input_domain = domain
        ref_lbs, ref_ubs = intermediate_bounds

        # Bounds on the inputs
        l_0 = domain.select(-1, 0)
        u_0 = domain.select(-1, 1)

        _, _, cond_first_linear = self.build_first_conditioned_layer(
            l_0, u_0, self.layers[0], no_conv=no_conv)
        # Add the first layer, appropriately rescaled.
        self.weights = [cond_first_linear]
        # Change the lower bounds and upper bounds corresponding to the inputs
        if not no_conv:
            self.lower_bounds = ref_lbs.copy()
            self.upper_bounds = ref_ubs.copy()
            self.lower_bounds[0] = -torch.ones_like(l_0)
            self.upper_bounds[0] = torch.ones_like(u_0)
        else:
            self.original_shape_lbs = ref_lbs.copy()
            self.original_shape_ubs = ref_ubs.copy()
            self.original_shape_lbs[0] = -torch.ones_like(l_0)
            self.original_shape_ubs[0] = torch.ones_like(u_0)
            self.lower_bounds = [-torch.ones_like(l_0.view(-1))]
            self.upper_bounds = [torch.ones_like(u_0.view(-1))]
            for lay_idx in range(1, len(ref_lbs)):
                self.lower_bounds.append(ref_lbs[lay_idx].view(-1).clone())
                self.upper_bounds.append(ref_ubs[lay_idx].view(-1).clone())

        next_is_linear = False
        lay_idx = 1
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False
                orig_shape_prev_ub = self.original_shape_ubs[lay_idx] if no_conv else None
                new_layer, _ = self.build_obj_layer(
                    self.upper_bounds[lay_idx], layer, no_conv=no_conv, orig_shape_prev_ub=orig_shape_prev_ub)
                self.weights.append(new_layer)
                lay_idx += 1
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass

    def solve_problem(self, weights, lower_bounds, upper_bounds, all_optim=False, override_numerical_errors=False):
        '''
        Compute bounds on the last layer of the problem.
        '''
        ini_lbs, ini_ubs = weights[-1].interval_forward(torch.clamp(lower_bounds[-1], 0, None),
                                                        torch.clamp(upper_bounds[-1], 0, None))
        out_shape = ini_lbs.shape
        nb_out = prod(out_shape)

        opt_to_bound = torch.eye(nb_out, device=ini_lbs.device).type(ini_lbs.type())
        if not all_optim:
            ubs_to_opt = (ini_ubs > 0)
            lbs_to_opt = (ini_lbs < 0)

            # Only one needing optim are the ambiguous ones
            to_opt = ubs_to_opt & lbs_to_opt
            opt_to_bound = opt_to_bound[to_opt.view(-1)]

        nb_targets = opt_to_bound.size(0)

        if nb_targets == 0:
            # No ReLU is ambiguous, no need for any opt here.
            lbs, ubs = ini_lbs, ini_ubs
        else:
            final_coeffs = opt_to_bound.view((nb_targets,) + out_shape)
            bothopt_finalcoeffs = torch.cat([-final_coeffs, final_coeffs], 0)

            additional_coeffs = {
                len(lower_bounds): bothopt_finalcoeffs
            }

            bound = self.optimize(weights, additional_coeffs,
                                  lower_bounds, upper_bounds)

            if not all_optim:
                opted_ubs = -bound[:nb_targets]
                opted_lbs = bound[nb_targets:]
                ubs = torch.where(to_opt, (opted_ubs @ opt_to_bound).view(ini_ubs.shape), ini_ubs)
                lbs = torch.where(to_opt, (opted_lbs @ opt_to_bound).view(ini_lbs.shape), ini_lbs)
            else:
                ubs = -bound[:nb_targets]
                lbs = bound[nb_targets:]
                lbs = lbs.view(out_shape)
                ubs = ubs.view(out_shape)

        if not override_numerical_errors:
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"
        else:
            ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"


        return lbs, ubs

    def best_naive_kw_optimizer(self, method_args):
        # best bounds out of kw and naive interval propagation
        kw_fun, kw_logger = self.optimizers['init'](None)
        naive_fun, naive_logger = self.optimizers['init'](None)

        def optimize(*args, **kwargs):
            self.set_initialisation('KW')
            bounds_kw = kw_fun(*args, **kwargs)
            self.set_initialisation('naive')
            bounds_naive = naive_fun(*args, **kwargs)
            bounds = torch.max(bounds_kw, bounds_naive)
            return bounds

        return optimize, [kw_logger, naive_logger]

    def init_optimizer(self, method_args):
        return self.init_optimize, None

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
        bound = compute_objective(dual_vars, matching_primal_vars, final_coeffs)
        return bound

    def subgradient_optimizer(self, method_args):
        # Define default values
        args = {
            'nb_steps': 100,
            'step_size': 1e-3
        }
        args.update(method_args)

        nb_steps = args['nb_steps']
        step_size = args['step_size']
        logger = None
        if self.store_bounds_progress >= 0:
            logger = OptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()
            dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                 lower_bounds, upper_bounds)
            for step in range(nb_steps):
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)
                dual_subg = matching_primal.as_dual_subgradient()
                dual_vars.add_(step_size, dual_subg)

                if self.store_bounds_progress >= 0 and len(weights) == self.store_bounds_progress:
                    if (step - 1) % 10 == 0:
                        start_logging_time = time.time()
                        matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                              lower_bounds, upper_bounds,
                                                                              dual_vars)
                        bound = compute_objective(dual_vars, matching_primal, final_coeffs)
                        logging_time = time.time() - start_logging_time
                        logger.add_point(len(weights), bound, logging_time=logging_time)

            # End of the optimization
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = compute_objective(dual_vars, matching_primal, final_coeffs)
            return bound

        return optimize, logger

    def adam_subg_optimizer(self, method_args):
        # Define default values
        args = {
            'nb_steps': 100,
            'initial_step_size': 1e-3,
            'final_step_size': 1e-6,
            'betas': (0.9, 0.999)
        }
        args.update(method_args)

        nb_steps = args['nb_steps']
        initial_step_size = args['initial_step_size']
        final_step_size = args['final_step_size']
        beta_1 = args['betas'][0]
        beta_2 = args['betas'][1]
        logger = None
        if self.store_bounds_progress >= 0:
            logger = OptimizationTrace()

        def optimize(weights, final_coeffs,
                     lower_bounds, upper_bounds):
            if self.store_bounds_progress >= 0:
                logger.start_timing()

            dual_vars = self.decomposition.initial_dual_solution(weights, final_coeffs,
                                                                 lower_bounds, upper_bounds)
            exp_avg = dual_vars.zero_like()
            exp_avg_sq = dual_vars.zero_like()

            for step in range(1, nb_steps+1):
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)

                dual_subg = matching_primal.as_dual_subgradient()

                step_size = initial_step_size + (step / nb_steps) * (final_step_size - initial_step_size)
                # step_size = initial_step_size / (1 + (step / nb_steps)*(initial_step_size / final_step_size - 1))

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
                        bound = compute_objective(dual_vars, matching_primal, final_coeffs)
                        logging_time = time.time() - start_logging_time
                        logger.add_point(len(weights), bound, logging_time=logging_time)

            # store last dual solution for future usage
            self.last_duals = dual_vars.rhos

            # End of the optimization
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = compute_objective(dual_vars, matching_primal, final_coeffs)

            return bound

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
            'maintain_primal': True
        }
        args.update(method_args)
        nb_total_steps = args['nb_total_steps']
        max_nb_inner_steps = args['max_nb_inner_steps']
        default_eta = args['eta']
        initial_eta = args['initial_eta']
        final_eta = args['final_eta']
        inner_cutoff = args['inner_cutoff']
        maintain_primal = args['maintain_primal']
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
            prox_dual_vars = dual_vars
            steps = 0
            # We operate in the primal, and are going to keep updating our
            # primal_vars. For each primal vars, we have a formula giving the
            # associated dual variables, and the hope is that optimizing
            # correctly the primal variables to shrink the dual gap will lead
            # to a good solution on the dual.
            while steps < nb_total_steps:
                prox_dual_vars = dual_vars
                if not maintain_primal:
                    # That means that what we want to maintain the dual vars.
                    # In any case, we want prox_dual_vars to be the current
                    # dual_vars For those two to match, this means that we need
                    # to update the primal variables such that there is no gap
                    # between the A variables and the B variables. That just
                    # means making a forward (possibly relaxed pass)
                    primal_vars = self.decomposition.make_primal_full_feasible(dual_vars,
                                                                               weights, final_coeffs,
                                                                               lower_bounds, upper_bounds)
                    # primal_vars.assert_subproblems_feasible(weights, final_coeffs,
                    #                                         lower_bounds, upper_bounds)
                    # primal_vars.as_dual_subgradient().assert_zero()

                if (initial_eta is not None) and (final_eta is not None):
                    eta = initial_eta + (steps / nb_total_steps) * (final_eta - initial_eta)
                    # eta = initial_eta / (1 + (steps / nb_total_steps) * (initial_eta/final_eta - 1))
                else:
                    eta = default_eta
                # Get lambda, rho:
                # For the proximal problem, they are the gradient on the z_a - z_b differences.
                dual_vars = prox_dual_vars.add(primal_vars.as_dual_subgradient(),
                                               1/eta)
                nb_inner_step = min(max_nb_inner_steps, nb_total_steps - steps)
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
                            objs = compute_proximal_objective(primal_vars, dual_vars, prox_dual_vars, final_coeffs, eta)
                            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                                  lower_bounds, upper_bounds,
                                                                                  dual_vars)
                            bound = compute_objective(dual_vars, matching_primal, final_coeffs)
                            logging_time = time.time() - start_logging_time
                            logger.add_proximal_point(len(weights), bound, objs, logging_time=logging_time)

            # End of optimization
            # Compute an actual bound
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = compute_objective(dual_vars, matching_primal, final_coeffs)
            return bound

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

        nb_steps = args['nb_steps']
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
                    bound = compute_objective(c_dual_vars, matching_primal, final_coeffs)
                    return bound

                for rho in c_rhos:
                    rho.requires_grad = True

                if algorithm == "adam":
                    optimizer = torch.optim.Adam(c_rhos, lr=initial_step_size, betas=(beta_1, beta_2))
                else:
                    # "adagrad"
                    optimizer = torch.optim.Adagrad(c_rhos, lr=initial_step_size)  # lr=1e-2 works best

                # do autograd-adam
                for step in range(nb_steps):
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
                self.last_duals = dual_detached

                # End of the optimization
                dual_vars = DualVarSet(dual_detached)
                matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                      lower_bounds, upper_bounds,
                                                                      dual_vars)
                bound = compute_objective(dual_vars, matching_primal, final_coeffs)

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
            'inner_cutoff': 1e-3,
            'maintain_primal': True,
            'acceleration_dict': {'momentum': 0}
        }
        args.update(method_args)
        nb_total_steps = args['nb_total_steps']
        max_nb_inner_steps = args['max_nb_inner_steps']
        default_eta = args['eta']
        initial_eta = args['initial_eta']
        final_eta = args['final_eta']
        inner_cutoff = args['inner_cutoff']
        maintain_primal = args['maintain_primal']
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
            prox_dual_vars = dual_vars.copy()
            steps = 0
            # We operate in the primal, and are going to keep updating our
            # primal_vars. For each primal vars, we have a formula giving the
            # associated dual variables, and the hope is that optimizing
            # correctly the primal variables to shrink the dual gap will lead
            # to a good solution on the dual.
            while steps < nb_total_steps:
                dual_vars.update_acceleration(acceleration_dict=acceleration_dict)
                prox_dual_vars = dual_vars.copy()
                if not maintain_primal:
                    # That means that what we want to maintain the dual vars.
                    # In any case, we want prox_dual_vars to be the current
                    # dual_vars For those two to match, this means that we need
                    # to update the primal variables such that there is no gap
                    # between the A variables and the B variables. That just
                    # means making a forward (possibly relaxed pass)
                    primal_vars = self.decomposition.make_primal_full_feasible(dual_vars,
                                                                               weights, final_coeffs,
                                                                               lower_bounds, upper_bounds)
                    # primal_vars.assert_subproblems_feasible(weights, final_coeffs,
                    #                                         lower_bounds, upper_bounds)
                    # primal_vars.as_dual_subgradient().assert_zero()

                if (initial_eta is not None) and (final_eta is not None):
                    eta = initial_eta + (steps / nb_total_steps) * (final_eta - initial_eta)
                else:
                    eta = default_eta
                # Get lambda, rho:
                # For the proximal problem, they are the gradient on the z_a - z_b differences.
                dual_vars.update_from_anchor_points(prox_dual_vars, primal_vars, eta, acceleration_dict=acceleration_dict)
                nb_inner_step = min(max_nb_inner_steps, nb_total_steps - steps)
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
                        # decrease gives the improvement we make in the primal proximal problem
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
                            objs = compute_proximal_objective(primal_vars, dual_vars, prox_dual_vars, final_coeffs, eta)
                            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                                  lower_bounds, upper_bounds,
                                                                                  dual_vars)
                            bound = compute_objective(dual_vars, matching_primal, final_coeffs)
                            logging_time = time.time() - start_logging_time
                            logger.add_proximal_point(len(weights), bound, objs, logging_time=logging_time)

            # End of optimization
            # Compute an actual bound
            matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                                  lower_bounds, upper_bounds,
                                                                  dual_vars)
            bound = compute_objective(dual_vars, matching_primal, final_coeffs)
            return bound

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
        primal_val = compute_objective(for_prim_opt_dual_vars, primal_vars, final_coeffs)
        if include_prox_terms:
            primal_val += p_as_dual.weighted_squared_norm(1/(2*eta))

        # Compute the objective if we plug in the solution for the primal vars, and
        # are trying to maximize over the dual
        matching_primal = self.decomposition.get_optim_primal(weights, final_coeffs,
                                                              lower_bounds, upper_bounds,
                                                              dual_vars)
        dual_minus_proxdual = dual_vars.add(prox_dual_vars, -1)
        dual_val = compute_objective(dual_vars, matching_primal, final_coeffs)
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


def compute_objective(dual_vars, primal_vars, additional_coeffs):
    '''
    We assume that all the constraints are satisfied.
    '''
    val = dual_vars.bdot(primal_vars.as_dual_subgradient())
    for layer, add_coeff in additional_coeffs.items():
        # zahats are going from 1 so we need to remove 1 to the index
        val += bdot(add_coeff, primal_vars.zahats[layer-1])
    return val


def compute_proximal_objective(primal_vars, current_dual_vars, anchor_dual_vars, additional_coeffs, eta):
    """
    Given primal variables as lists of tensors, and dual anchor variables
    (and functions thereof) as DualVars, compute the value of the objective of the proximal problem (Wolfe dual of
    proximal on dual variables).
    :return: a tensor of objectives, of size 2 x n_neurons of the layer to optimize.
    """

    val = current_dual_vars.bdot(primal_vars.as_dual_subgradient())
    for layer, add_coeff in additional_coeffs.items():
        # zahats are going from 1 so we need to remove 1 to the index
        val += bdot(add_coeff, primal_vars.zahats[layer - 1])

    val -= current_dual_vars.subtract(1, anchor_dual_vars).weighted_squared_norm(eta/2)

    return val
