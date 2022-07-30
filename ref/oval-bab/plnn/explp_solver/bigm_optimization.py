import torch
from plnn.proxlp_solver import utils
import math
import copy
from plnn.explp_solver import anderson_optimization


def layer_primal_linear_minimization(lay_idx, f_k, g_k, cl_k, cu_k):
    """
    Given the post-activation bounds and the (functions of) dual variables of the current layer tensors
    (shape 2 * n_neurons_to_opt x c_layer_size), compute the values of the primal variables (x and z) minimizing the
    inner objective.
    :return: optimal x, optimal z (tensors, shape: 2 * n_neurons_to_opt x c_layer_size)
    """
    opt_x_k = (torch.where(f_k >= 0, cu_k.unsqueeze(1), cl_k.unsqueeze(1)))
    if lay_idx > 0:
        opt_z_k = (torch.where(g_k >= 0, torch.ones_like(g_k), torch.zeros_like(g_k)))
    else:
        # g_k is defined from 1 to n - 1.
        opt_z_k = None

    return opt_x_k, opt_z_k


def compute_bounds(weights, dual_vars, clbs, cubs, l_preacts, u_preacts, prox=False):
    """
    Given the network layers, post- and pre-activation bounds  as lists of tensors, and dual variables
    (and functions thereof) as DualVars. compute the value of the (batch of) network bounds.
    If we are solving the prox problem (by default, no), check for non-negativity of dual vars. If they are negative,
    the bounds need to be -inf. This is because these non-negativity constraints have been relaxed, in that case.
    :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
    upper bound of each neuron, the second the lower bound.
    """

    if prox:
        # in the prox case the dual variables might be negative (the constraint has been dualized). We therefore need
        # to clamp them to obtain a valid bound.
        c_dual_vars = dual_vars.get_nonnegative_copy(weights, l_preacts, u_preacts)
    else:
        c_dual_vars = dual_vars

    bounds = 0
    for lin_k, alpha_k_1 in zip(weights, c_dual_vars.alpha[1:]):
        b_k = lin_k.get_bias()
        bounds += utils.bdot(alpha_k_1, b_k)

    for f_k, cl_k, cu_k in zip(c_dual_vars.fs, clbs, cubs):
        bounds -= utils.bdot(torch.clamp(f_k, 0, None), cu_k.unsqueeze(1))
        bounds -= utils.bdot(torch.clamp(f_k, None, 0), cl_k.unsqueeze(1))

    for g_k in c_dual_vars.gs:
        bounds -= torch.clamp(g_k, 0, None).view(*g_k.shape[:2], -1).sum(dim=-1)  # z to 1

    for beta_k_1, l_preact, lin_k in zip(c_dual_vars.beta_1[1:], l_preacts[1:], weights):
        bounds += utils.bdot(beta_k_1, (l_preact.unsqueeze(1) - lin_k.get_bias()))

    return bounds


def compute_dual_subgradient(weights, dual_vars, lbs, ubs, l_preacts, u_preacts):
    """
    Given the network layers, post- and pre-activation bounds as lists of
    tensors, and dual variables (and functions thereof) as DualVars, compute the subgradient of the dual objective.
    :return: DualVars instance representing the subgradient for the dual variables (does not contain fs and gs)
    """

    # The step needs to be taken for all layers at once, as coordinate ascent seems to be problematic,
    # see https://en.wikipedia.org/wiki/Coordinate_descent

    nb_relu_layers = len(dual_vars.beta_0)

    alpha_subg = [torch.zeros_like(dual_vars.alpha[0])]
    beta_0_subg = [torch.zeros_like(dual_vars.beta_0[0])]
    beta_1_subg = [torch.zeros_like(dual_vars.beta_1[0])]
    xkm1, _ = layer_primal_linear_minimization(0, dual_vars.fs[0], None, lbs[0], ubs[0])
    for lay_idx in range(1, nb_relu_layers):
        # For each layer, we will do one step of subgradient descent on all dual variables at once.
        lin_k = weights[lay_idx - 1]
        # solve the inner problems.
        xk, zk = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1],
                                                  lbs[lay_idx], ubs[lay_idx])

        # compute and store the subgradients.
        xk_hat = lin_k.forward(xkm1)
        alpha_subg.append(xk_hat - xk)
        beta_0_subg.append(xk - zk * u_preacts[lay_idx].unsqueeze(1))
        beta_1_subg.append(xk + (1 - zk) * l_preacts[lay_idx].unsqueeze(1) - xk_hat)

        xkm1 = xk

    return DualVars(alpha_subg, beta_0_subg, beta_1_subg, None, None, None, None)


def compute_prox_obj(xt, zt, xhatt, y, eta, anchor_vars, weights, l_preacts, u_preacts):
    """
    Given the network layers, post-activation bounds and primal variables as lists of tensors, and dual anchor variables
    (and functions thereof) as DualVars, compute the value of the objective of the proximal problem (Wolfe dual of
    proximal on dual variables).
    :return: a tensor of objectives, of size 2 x n_neurons of the layer to optimize.
    """
    objs = 0

    # Compute the linear part of the objective.
    for lin_k, alpha_k_1 in zip(weights[:-1], anchor_vars.alpha[1:-1]):
        b_k = lin_k.get_bias()
        objs += utils.bdot(alpha_k_1, b_k)
    # The last layer bias is positive nevertheless
    objs += utils.bdot(torch.abs(anchor_vars.alpha[-1]), weights[-1].get_bias())

    for f_k, xt_k in zip(anchor_vars.fs, xt):
        objs -= utils.bdot(xt_k, f_k)
    for g_k, zt_k in zip(anchor_vars.gs, zt):
        objs -= (utils.bdot(zt_k, g_k))

    for beta_k_1, l_preact, lin_k in zip(anchor_vars.beta_1[1:], l_preacts[1:], weights):
        objs += utils.bdot(beta_k_1, (l_preact.unsqueeze(1) - lin_k.get_bias()))

    for lay_idx in range(1, len(anchor_vars.beta_0)):
        objs += (utils.bdot(y.ya[lay_idx], anchor_vars.alpha[lay_idx]) +
                 utils.bdot(y.yb0[lay_idx], anchor_vars.beta_0[lay_idx]) +
                 utils.bdot(y.yb1[lay_idx], anchor_vars.beta_1[lay_idx]))

    # Compute the quadratic part of the objective.
    for lay_idx in range(1, len(anchor_vars.beta_0)):
        quadratic_term = 0
        # compute the quadratic terms (-the subgradients, in l2 norm).
        quadratic_term += utils.bl2_norm(xt[lay_idx] - xhatt[lay_idx-1] - y.ya[lay_idx])
        quadratic_term += utils.bl2_norm(zt[lay_idx-1] * u_preacts[lay_idx].unsqueeze(1) - xt[lay_idx] - y.yb0[lay_idx])
        quadratic_term += utils.bl2_norm(xhatt[lay_idx-1] - xt[lay_idx] - (1 - zt[lay_idx-1]) *
                                         l_preacts[lay_idx].unsqueeze(1) - y.yb1[lay_idx])
        objs += (1 / (4 * eta)) * quadratic_term

    return objs


def primal_vars_do_fw_step(lay_idx, dual_vars, anchor_vars, eta, xt, zt, xhatt, y, weights, additional_coeffs, clbs,
                           cubs, l_preacts, u_preacts):
    """
    Given the up-to-date (through the closed form update) dualvars (DualVars instance),
    anchor_points (DualVarts instance), and primal variables (lists of tensors) compute the conditional gradient for
    the x, z optimization, and take a step with optimal step size in that direction. Works in place.
    lay_idx is the layer for which to perform the update.
    """

    # compute the conditional gradient and the directions to it
    xk_cd, zk_cd = layer_primal_linear_minimization(
        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])

    # optimal step size computation
    dx = xk_cd - xt[lay_idx]

    if lay_idx < len(weights)-1:
        # compute "a" terms relative to the following layer
        Wdx = weights[lay_idx].forward(dx) - weights[lay_idx].get_bias()
        a_kp1terms = 2 * utils.bl2_norm(Wdx)
    else:
        a_kp1terms = torch.zeros(*xk_cd.shape[:2], device=xk_cd.device)

    if lay_idx > 0:
        dz = zk_cd - zt[lay_idx - 1]
        u_preact_d = u_preacts[lay_idx].unsqueeze(1) * dz - dx
        l_preact_d = l_preacts[lay_idx].unsqueeze(1) * dz - dx
        a_kterms = utils.bl2_norm(dx) + utils.bl2_norm(u_preact_d) + utils.bl2_norm(l_preact_d)
    else:
        a_kterms = torch.zeros(*xk_cd.shape[:2], device=xk_cd.device)

    # Let's compute the optimal step size: we end up with a polynomial looking like a*s^2 + b*s + cte
    a = (1 / (4 * eta)) * (a_kp1terms + a_kterms)

    if lay_idx < len(weights)-1:
        alphakp1_bcoeff = (1 / (2 * eta)) * (xt[lay_idx + 1] - xhatt[lay_idx] - y.ya[lay_idx + 1]) - \
                           anchor_vars.alpha[lay_idx + 1]
        beta1kp1_bcoeff = (1 / (2 * eta)) * (
                xhatt[lay_idx] - xt[lay_idx + 1] - (1 - zt[lay_idx]) * l_preacts[lay_idx + 1].unsqueeze(1) -
                y.yb1[lay_idx + 1]) - anchor_vars.beta_1[lay_idx + 1]
        b_kp1terms = -utils.bdot(alphakp1_bcoeff, Wdx) + utils.bdot(beta1kp1_bcoeff, Wdx)
    else:
        b_kp1terms = utils.bdot(weights[-1].backward(additional_coeffs[len(weights)]), dx)

    if lay_idx > 0:
        alphak_bcoeff = (1 / (2 * eta)) * (xt[lay_idx] - xhatt[lay_idx - 1] - y.ya[lay_idx]) - \
            anchor_vars.alpha[lay_idx]
        beta0k_bcoeff = (1 / (2 * eta)) * (
            zt[lay_idx - 1] * u_preacts[lay_idx].unsqueeze(1) - xt[lay_idx] - y.yb0[lay_idx]) - \
            anchor_vars.beta_0[lay_idx]
        beta1k_bcoeff = (1 / (2 * eta)) * (xhatt[lay_idx - 1] - xt[lay_idx] - (1 - zt[lay_idx - 1]) *
            l_preacts[lay_idx].unsqueeze(1) - y.yb1[lay_idx]) - anchor_vars.beta_1[lay_idx]
        b_kterms = utils.bdot(alphak_bcoeff, dx) + utils.bdot(beta0k_bcoeff, u_preact_d) + \
                   utils.bdot(beta1k_bcoeff, l_preact_d)
    else:
        b_kterms = torch.zeros(*xk_cd.shape[:2], device=xk_cd.device)

    b = (b_kterms + b_kp1terms)

    # By definition, b should be negative but there might be some floating point trouble
    torch.clamp(b, None, 0, out=b)

    optimal_step_size = (- b / (2 * a)).view((*a.shape[:2], *((1,) * (len(xk_cd.shape) - 2))))
    # If a==0, that means that the conditional gradient is equal to the current position, so no need to move.
    optimal_step_size[a == 0] = 0
    optimal_step_size.clamp_(0, 1)

    # take a convex combination of the current primals and the conditional gradient
    xt[lay_idx].addcmul_(optimal_step_size, dx)
    if lay_idx < len(weights)-1:
        xhatt[lay_idx] = weights[lay_idx].forward(xt[lay_idx])
    if lay_idx > 0:
        zt[lay_idx-1].addcmul_(optimal_step_size, dz)


class DualVars:
    """
    Class representing the dual variables alpha, beta_0, and beta_1, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    """
    def __init__(self, alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.fs = fs
        self.gs = gs
        self.alpha_back = alpha_back
        self.beta_1_back = beta_1_back

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        beta_0 = []  # Indexed from 0 to n-1, the first is always zero
        beta_1 = []  # Indexed from 0 to n, the first and last are always zero
        alpha_back = []  # Indexed from 1 to n,
        beta_1_back = []  # Indexed from 1 to n, last always 0

        # Build also the shortcut terms f and g
        fs = []  # Indexed from 0 to n-1
        gs = []  # Indexed from 1 to n-1

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device, dtype=add_coeff.dtype)
        # Insert the dual variables for the box bound
        fs.append(zero_tensor(input_size))
        fixed_0_inpsize = zero_tensor(input_size)
        beta_0.append(fixed_0_inpsize)
        beta_1.append(fixed_0_inpsize)
        alpha.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_0[-1])[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_0.append(zero_tensor(nb_outputs))
            beta_1.append(zero_tensor(nb_outputs))

            # Initialize the shortcut terms
            fs.append(zero_tensor(nb_outputs))
            gs.append(zero_tensor(nb_outputs))

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_1.append(torch.zeros_like(alpha[-1]))

        for lay_idx in range(1, len(alpha)):
            alpha_back.append(weights[lay_idx-1].backward(alpha[lay_idx]))
            beta_1_back.append(weights[lay_idx-1].backward(beta_1[lay_idx]))

        # Adjust the fact that the last term for the f shorcut is not zero,
        # because it depends on alpha.
        fs[-1] = -weights[-1].backward(additional_coeffs[len(weights)])

        return DualVars(alpha, beta_0, beta_1, fs, gs, alpha_back, beta_1_back)

    def update_f_g(self, l_preacts, u_preacts, lay_idx="all"):
        """
        Given the network pre-activation bounds as lists of tensors, update f_k and g_k in place.
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.beta_0))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            self.fs[lay_idx] = (
                    self.alpha[lay_idx] - self.alpha_back[lay_idx] -
                    (self.beta_0[lay_idx] + self.beta_1[lay_idx]) + self.beta_1_back[lay_idx])
            if lay_idx > 0:
                self.gs[lay_idx - 1] = (self.beta_0[lay_idx] * u_preacts[lay_idx].unsqueeze(1) +
                                        self.beta_1[lay_idx] * l_preacts[lay_idx].unsqueeze(1))

    def projected_linear_combination(self, coeff, o_vars, weights):
        """
        Given a batch of coefficients (a tensor) and another set of dual variables (instance of this calss), perform a
        linear combination according to the coefficient.
        Then project on the feasible domain (non-negativity constraints).
        This is done in place the set of variables of this class.
        """
        for lay_idx in range(1, len(self.beta_0)):
            self.alpha[lay_idx] = torch.clamp(self.alpha[lay_idx] + coeff * o_vars.alpha[lay_idx], 0, None)
            self.beta_0[lay_idx] = torch.clamp(self.beta_0[lay_idx] + coeff * o_vars.beta_0[lay_idx], 0, None)
            self.beta_1[lay_idx] = torch.clamp(self.beta_1[lay_idx] + coeff * o_vars.beta_1[lay_idx], 0, None)
            self.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(self.alpha[lay_idx])
            self.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(self.beta_1[lay_idx])
        return

    def get_nonnegative_copy(self, weights, l_preacts, u_preacts):
        """
        Given the network layers and pre-activation bounds as lists of tensors, clamp all dual variables to be
        non-negative. A heuristic to compute some bounds.
        Returns a copy of this instance where all the entries are non-negative and the f and g functions are up-to-date.
        """
        nonneg = self.copy()
        for lay_idx in range(1, len(nonneg.beta_0)):
            nonneg.alpha[lay_idx].clamp_(0, None)
            nonneg.beta_0[lay_idx].clamp_(0, None)
            nonneg.beta_1[lay_idx].clamp_(0, None)
            nonneg.alpha_back[lay_idx-1] = weights[lay_idx-1].backward(nonneg.alpha[lay_idx])
            nonneg.beta_1_back[lay_idx-1] = weights[lay_idx-1].backward(nonneg.beta_1[lay_idx])
        nonneg.update_f_g(l_preacts, u_preacts)
        return nonneg

    def update_from_anchor_points(self, anchor_point, xt, zt, xhatt, y, eta, weights, l_preacts, u_preacts, lay_idx="all"):
        """
        Given the anchor point (DualVars instance), post-activation bounds, primal vars as lists of
        tensors (y is a YVars instance), compute and return the updated the dual variables (anchor points) with their
        closed-form from KKT conditions. The update is performed in place.
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(1, len(self.beta_0))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            # For each layer, do the dual anchor points' update.
            # compute the quadratic terms (-the subgradients, in l2 norm).
            self.alpha[lay_idx] = anchor_point.alpha[lay_idx] - (1 / (2 * eta)) * (xt[lay_idx] - xhatt[lay_idx-1]
                                                                                       - y.ya[lay_idx])
            self.beta_0[lay_idx] = anchor_point.beta_0[lay_idx] - (1 / (2 * eta)) * (
                    zt[lay_idx-1] * u_preacts[lay_idx].unsqueeze(1) - xt[lay_idx] - y.yb0[lay_idx])
            self.beta_1[lay_idx] = anchor_point.beta_1[lay_idx] - (1 / (2 * eta)) * (
                    xhatt[lay_idx - 1] - xt[lay_idx] - (1 - zt[lay_idx-1]) * l_preacts[lay_idx].unsqueeze(1) -
                    y.yb1[lay_idx])
            self.alpha_back[lay_idx-1] = weights[lay_idx-1].backward(self.alpha[lay_idx])
            self.beta_1_back[lay_idx-1] = weights[lay_idx-1].backward(self.beta_1[lay_idx])

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return DualVars(
            copy.deepcopy(self.alpha),
            copy.deepcopy(self.beta_0),
            copy.deepcopy(self.beta_1),
            copy.deepcopy(self.fs),
            copy.deepcopy(self.gs),
            copy.deepcopy(self.alpha_back),
            copy.deepcopy(self.beta_1_back)
        )

    def as_explp_initialization(self, weights, clbs, cubs, l_preacts, u_preacts, prox=False):
        """
        Given the network layers and pre-activation bounds as lists of tensors,
        compute and return the corresponding initialization of the explp (Anderson) variables from the instance of this
        class.
        """
        # as the duals of the proximal method might be non-negative, clipping is necessary to use them for init
        if prox:
            dual_vars = self.get_nonnegative_copy(weights, l_preacts, u_preacts)
        else:
            dual_vars = self.copy()
        sum_beta = [None] * len(dual_vars.beta_0)
        sum_Wp1Ibetap1 = [None] * (len(dual_vars.beta_0))
        sum_W1mIubeta = [None] * len(dual_vars.beta_0)
        sum_WIlbeta = [None] * len(dual_vars.beta_0)
        xs = [None] * len(dual_vars.beta_0)
        zs = [None] * (len(dual_vars.beta_0) - 1)
        for lay_idx in range(len(dual_vars.beta_0)+1):
            if lay_idx == 0:
                sum_beta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_W1mIubeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_WIlbeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                xs[lay_idx], _ = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], None, clbs[lay_idx],
                                                          cubs[lay_idx])
            elif lay_idx > 0:
                if lay_idx < len(dual_vars.beta_0):
                    sum_beta[lay_idx] = dual_vars.beta_0[lay_idx] + dual_vars.beta_1[lay_idx]
                    sum_W1mIubeta[lay_idx] = dual_vars.beta_0[lay_idx] * (u_preacts[lay_idx].unsqueeze(1)
                                                                          - weights[lay_idx-1].get_bias())
                    sum_WIlbeta[lay_idx] = dual_vars.beta_1[lay_idx] * (l_preacts[lay_idx].unsqueeze(1)
                                                                        - weights[lay_idx-1].get_bias())
                    xs[lay_idx], zs[lay_idx-1] = layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                sum_Wp1Ibetap1[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])

        return dual_vars.alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, dual_vars.fs, dual_vars.gs, xs, zs

    def as_cut_initialization(self, weights, clbs, cubs, l_preacts, u_preacts, prox=False):
        """
        Given the network layers and pre-activation bounds as lists of tensors,
        compute and return the corresponding initialization of the explp (Anderson) variables from the instance of this
        class.
        """
        # as the duals of the proximal method might be non-negative, clipping is necessary to use them for init
        if prox:
            dual_vars = self.get_nonnegative_copy(weights, l_preacts, u_preacts)
        else:
            dual_vars = self.copy()
        sum_beta = [None] * len(dual_vars.beta_0)
        sum_Wp1Ibetap1 = [None] * (len(dual_vars.beta_0))
        sum_W1mIubeta = [None] * len(dual_vars.beta_0)
        sum_WIlbeta = [None] * len(dual_vars.beta_0)
        xs = [None] * len(dual_vars.beta_0)
        zs = [None] * (len(dual_vars.beta_0) - 1)
        beta_list = []
        I_list = []
        for lay_idx in range(len(dual_vars.beta_0)+1):
            xkm1 = xs[lay_idx-1]
            if lay_idx == 0:
                sum_beta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_W1mIubeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                sum_WIlbeta[lay_idx] = torch.zeros_like(dual_vars.beta_0[lay_idx])
                xs[lay_idx], _ = layer_primal_linear_minimization(lay_idx, dual_vars.fs[lay_idx], None, clbs[lay_idx],
                                                                  cubs[lay_idx])
            elif lay_idx > 0:
                I_list.append([])
                beta_list.append([])
                if lay_idx < len(dual_vars.beta_0):
                    sum_beta[lay_idx] = dual_vars.beta_0[lay_idx] + dual_vars.beta_1[lay_idx]
                    beta_list[lay_idx-1].append(dual_vars.beta_0[lay_idx])
                    beta_list[lay_idx-1].append(dual_vars.beta_1[lay_idx])
                    xs[lay_idx], zs[lay_idx - 1] = layer_primal_linear_minimization(
                        lay_idx, dual_vars.fs[lay_idx], dual_vars.gs[lay_idx - 1], clbs[lay_idx], cubs[lay_idx])
                    # Dummy (smaller) I's (not used)
                    I_list[lay_idx - 1].extend([torch.empty_like(xs[lay_idx]), torch.empty_like(xs[lay_idx])])
                    sum_W1mIubeta[lay_idx] = dual_vars.beta_0[lay_idx] * (u_preacts[lay_idx].unsqueeze(1) -
                                                                          weights[lay_idx-1].get_bias())
                    sum_WIlbeta[lay_idx] = dual_vars.beta_1[lay_idx] * (l_preacts[lay_idx].unsqueeze(1) -
                                                                        weights[lay_idx-1].get_bias())
                sum_Wp1Ibetap1[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])

        return dual_vars.alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, dual_vars.fs, dual_vars.gs, xs, zs, beta_list, I_list


class YVars:
    """
    Class defining the primal vars associated to the lagrangian multipliers of the non-negativity constraints of the
    dual variables. Stored as lists of tensors (indexed from 0 to n-1 ReLU layers).
    """
    def __init__(self, ya, yb0, yb1):
        self.ya = ya
        self.yb0 = yb0
        self.yb1 = yb1

    @staticmethod
    def initialization_init(dual_vars):
        """
        Given dual vars to get the right shape, initialize the y variables as all 0s..
        """
        ya = []
        yb0 = []
        yb1 = []
        for lay_idx in range(len(dual_vars.beta_0)):
            ya.append(torch.zeros_like(dual_vars.beta_0[lay_idx]))
            yb0.append(torch.zeros_like(dual_vars.beta_0[lay_idx]))
            yb1.append(torch.zeros_like(dual_vars.beta_0[lay_idx]))
        return YVars(ya, yb0, yb1)

    def do_nnmp_step_layer(self, lay_idx, dual_vars, anchor_vars, eta, xt, zt, xhatt, l_preacts, u_preacts,
                           precision=torch.float):
        """
        Given the up-to-date (through the closed form update) dualvars (DualVars instance),
        anchor_points (DualVarts instance), and primal variables (lists of tensors) compute the NNMP atom for the y
        optimization, and take a step with optimal step size in that direction. Works in place.
        lay_idx is the layer for which to perform the update.
        """
        # Compute the gradients
        ya_grad = dual_vars.alpha[lay_idx]
        yb0_grad = dual_vars.beta_0[lay_idx]
        yb1_grad = dual_vars.beta_1[lay_idx]

        # Compute the LMO output
        lmo_ya = ((ya_grad < yb1_grad) & (ya_grad < yb0_grad) & (ya_grad < 0)).type(precision)
        lmo_yb0 = ((yb0_grad < ya_grad) & (yb0_grad < yb1_grad) & (yb0_grad < 0)).type(precision)
        lmo_yb1 = ((yb1_grad < ya_grad) & (yb1_grad < yb0_grad) & (yb1_grad < 0)).type(precision)

        # Compute the inner product with the current iterate
        inner_ya = utils.bdot(self.ya[lay_idx], ya_grad)
        inner_yb0 = utils.bdot(self.yb0[lay_idx], yb0_grad)
        inner_yb1 = utils.bdot(self.yb1[lay_idx], yb1_grad)
        clamped_norm = torch.clamp((self.ya[lay_idx] + self.yb0[lay_idx] + self.yb1[lay_idx]
                                    ).view(*self.ya[lay_idx].shape[:2], -1).sum(dim=-1), 1e-6, None)
        inner_grad_iterate = -1 / clamped_norm * (inner_ya + inner_yb0 + inner_yb1)

        # take the max with the best atom gradient by storing the difference and conditioning further operations on it.
        inner_grad_atom = (utils.bdot(lmo_ya, ya_grad) + utils.bdot(lmo_yb0, yb0_grad) + utils.bdot(lmo_yb1, yb1_grad))
        inner_grad_diff = (inner_grad_atom - inner_grad_iterate).view(
            (*inner_grad_atom.shape[:2], *((1,) * (len(lmo_yb0.shape) - 2))))
        # TODO: it is not true that the normalisation of the current iterate does not matter, as it influences what is the argmin here (seems to be OK anyways)

        # select the atom with which to take the linear combination
        clamped_norm = clamped_norm.view_as(inner_grad_diff)  # this is necessary as broadcasting always works pre-pending dimensions (not appending)
        atom_ya = torch.where(inner_grad_diff <= 0, lmo_ya, -self.ya[lay_idx] / clamped_norm)
        atom_yb0 = torch.where(inner_grad_diff <= 0, lmo_yb0, -self.yb0[lay_idx] / clamped_norm)
        atom_yb1 = torch.where(inner_grad_diff <= 0, lmo_yb1, -self.yb1[lay_idx] / clamped_norm)

        # Let's compute the optimal step size: we end up with a polynomial looking like a*s^2 + b*s + cte
        a = (1 / (4 * eta)) * (utils.bl2_norm(atom_ya) + utils.bl2_norm(atom_yb0) + utils.bl2_norm(atom_yb1))
        y_a_bcoeff = -(1/(2*eta)) * (xt[lay_idx] - xhatt[lay_idx-1] - self.ya[lay_idx]) + anchor_vars.alpha[lay_idx]
        y_b0_bcoeff = -(1/(2*eta)) * (zt[lay_idx-1] * u_preacts[lay_idx].unsqueeze(1) - xt[lay_idx] -
                                     self.yb0[lay_idx]) + anchor_vars.beta_0[lay_idx]
        y_b1_bcoeff = -(1/(2*eta)) * (xhatt[lay_idx - 1] - xt[lay_idx] - (1 - zt[lay_idx-1]) *
                                     l_preacts[lay_idx].unsqueeze(1) - self.yb1[lay_idx]) + anchor_vars.beta_1[lay_idx]
        b = (utils.bdot(y_a_bcoeff, atom_ya) + utils.bdot(y_b0_bcoeff, atom_yb0) + utils.bdot(y_b1_bcoeff, atom_yb1))
        # By definition, b should be negative but there might be some floating point trouble
        torch.clamp(b, None, 0, out=b)
        optimal_step_size = (- b / (2 * a)).view((*inner_grad_atom.shape[:2], *((1,) * (len(lmo_yb0.shape) - 2))))
        # If a==0, that means that selected atom is equal to the current position, so no need to move.
        optimal_step_size[a == 0] = 0

        # if (inner_grad_diff < 0).any():
        #    print("choosing -x_t")

        # Selectively clamp step size to avoid that backwards iters (with -x_t) don't cross the origin
        optimal_step_size = torch.where(
            inner_grad_diff <= 0, optimal_step_size,
            torch.max(torch.min(optimal_step_size, clamped_norm * 0.999999), torch.zeros_like(optimal_step_size)))

        # Perform the update in-place.
        self.ya[lay_idx].addcmul_(optimal_step_size, atom_ya)
        self.yb0[lay_idx].addcmul_(optimal_step_size, atom_yb0)
        self.yb1[lay_idx].addcmul_(optimal_step_size, atom_yb1)


class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, beta_0, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_alpha = []
        self.m1_beta_0 = []
        self.m1_beta_1 = []
        # second moments
        self.m2_alpha = []
        self.m2_beta_0 = []
        self.m2_beta_1 = []
        for lay_idx in range(1, len(beta_0)):
            self.m1_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m1_beta_1.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_alpha.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_0.append(torch.zeros_like(beta_0[lay_idx]))
            self.m2_beta_1.append(torch.zeros_like(beta_0[lay_idx]))

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def update_moments_take_projected_step(self, weights, step_size, outer_it, dual_vars, dual_vars_subg):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the projected step from
        dual_vars.
        Update performed in place on dual_vars.
        """
        for lay_idx in range(1, len(dual_vars.beta_0)):
            # Update the ADAM moments.
            self.m1_alpha[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.alpha[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_0[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_0[lay_idx], alpha=1-self.coeff1)
            self.m1_beta_1[lay_idx-1].mul_(self.coeff1).add_(dual_vars_subg.beta_1[lay_idx], alpha=1-self.coeff1)
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.alpha[lay_idx], dual_vars_subg.alpha[lay_idx], value=1 - self.coeff2)
            self.m2_beta_0[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_0[lay_idx], dual_vars_subg.beta_0[lay_idx], value=1 - self.coeff2)
            self.m2_beta_1[lay_idx-1].mul_(self.coeff2).addcmul_(dual_vars_subg.beta_1[lay_idx], dual_vars_subg.beta_1[lay_idx], value=1 - self.coeff2)

            bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
            bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
            corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

            # Take the projected (non-negativity constraints) step.
            alpha_step_size = self.m1_alpha[lay_idx-1] / (self.m2_alpha[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.alpha[lay_idx] = torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)

            beta_0_step_size = self.m1_beta_0[lay_idx-1] / (self.m2_beta_0[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_0[lay_idx] = torch.clamp(dual_vars.beta_0[lay_idx] + corrected_step_size * beta_0_step_size, 0, None)

            beta_1_step_size = self.m1_beta_1[lay_idx-1] / (self.m2_beta_1[lay_idx-1].sqrt() + self.epsilon)
            dual_vars.beta_1[lay_idx] = torch.clamp(dual_vars.beta_1[lay_idx] + corrected_step_size * beta_1_step_size, 0, None)

            # update pre-computed backward passes.
            dual_vars.alpha_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.alpha[lay_idx])
            dual_vars.beta_1_back[lay_idx - 1] = weights[lay_idx - 1].backward(dual_vars.beta_1[lay_idx])


class BigMPInit(anderson_optimization.AndersonPInit):
    """
    Parent Init class for Anderson-relaxation-based solvers.
    """

    def as_stack(self, stack_size):
        # Repeat the content of this parent init to form a stack of size "stack_size"
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(
                [pinits[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (pinits.dim() - 1))) for pinits in varset])
        return BigMPInit(DualVars(*stacked_dual_list))

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([csol[batch_idx].unsqueeze(0) for csol in varset])
        return BigMPInit(DualVars(*stacked_dual_list))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append([c_init[:, -1].unsqueeze(1) for c_init in varset])
        return BigMPInit(DualVars(*stacked_dual_list))