"""
Files specific to the "saddle point frank wolfe" derivation for the Anderson relaxation.
"""

import torch
import copy
from plnn.explp_solver import anderson_optimization, bigm_optimization
from plnn.proxlp_solver import utils
import itertools
import math


def compute_primal_dual_gap(additional_coeffs, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts,
                            primal_vars, dual_vars, dual_bounds, precision=torch.float):
    # Compute the primal-dual gap for the saddle point problem (at convergence, this will be 0).
    add_coeff = next(iter(additional_coeffs.values()))
    primal_obj = utils.bdot(weights[-1].backward(add_coeff), primal_vars.xt[-1]) + \
          utils.bdot(add_coeff, weights[-1].get_bias())

    # for the alpha and beta terms, I need to compute the LMOs, and then inner_grad_lmo
    for lay_idx in range(1, len(dual_vars.alpha) - 1):
        alphak_condgrad, grad_alphak = dual_vars.alphak_grad_lmo(lay_idx, weights, primal_vars, precision)
        primal_obj += utils.bdot(alphak_condgrad, grad_alphak)

    for lay_idx in range(1, len(dual_vars.sum_beta)):
        _, _, _, _, atom_grad, _ = dual_vars.betak_grad_lmo(
            lay_idx, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision,
            only_grad=True)
        primal_obj += (dual_vars.beta_M[lay_idx] * torch.clamp(atom_grad, 0, None)).\
            sum(dim=tuple(range(2, atom_grad.dim())))

    return primal_obj - dual_bounds


def spfw_block_k_step(lay_idx, weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, l_preacts, u_preacts,
                      dual_vars, primal_vars, iter, step_size_dict, precision=torch.float, fixed_M=False,
                      max_iters=None):
    """
    Given list of layers, primal variables (instance of SaddlePrimalVars), pre and post activation bounds,
    perform a SP-FW step on a primal-dual block of the form (duals_k, primals_k, primals_{k-1}).
    :param step_size_dict: contains information on which step size rule to use, and with which parameters
    The update is performed in place on dual_vars.
    """

    # reference current layer variables via shorter names
    alpha_k = dual_vars.alpha[lay_idx]
    new_alphak_M = dual_vars.alpha_M[lay_idx]
    sum_betak = dual_vars.sum_beta[lay_idx]
    sum_WkIbetak = dual_vars.sum_Wp1Ibetap1[lay_idx - 1]
    sum_Wk1mIubetak = dual_vars.sum_W1mIubeta[lay_idx]
    sum_WkIlbetak = dual_vars.sum_WIlbeta[lay_idx]
    new_betak_M = dual_vars.beta_M[lay_idx]
    batch_shape = sum_betak.shape[:2]
    unsqueezedk_shape = (*batch_shape, *((1,) * (sum_betak.dim() - 2)))
    unsqueezedkm1_shape = (*batch_shape, *((1,) * (sum_WkIbetak.dim() - 2)))
    c_fixed_M = fixed_M or (iter + 1) % 100 != 0  # allow for checks only rather rarely

    # compute alpha_k's gradient and its LMO over it.
    alphak_condgrad, grad_alphak = dual_vars.alphak_grad_lmo(lay_idx, weights, primal_vars, precision)

    M_condgrad, W_I_M, WIlM, W1mIuM, atom_grad, WI = dual_vars.betak_grad_lmo(
        lay_idx, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision)

    # compute gradient and LMO for primals on this layer and the previous.
    xk_cd, xk_grad = primal_vars.xk_grad_lmo(lay_idx, dual_vars, clbs, cubs)
    zk_cd, zk_grad = primal_vars.zk_grad_lmo(lay_idx, dual_vars, precision)
    xkm1_cd, xkm1_grad = primal_vars.xk_grad_lmo(lay_idx-1, dual_vars, clbs, cubs)
    # for primals, the direction won't depend on M, so can compute it already.
    xk_diff = xk_cd - primal_vars.xt[lay_idx]
    xkm1_diff = xkm1_cd - primal_vars.xt[lay_idx - 1]
    zk_diff = zk_cd - primal_vars.zt[lay_idx - 1]

    # the variable M is problematic w/ SP as there is no monotonicity guarantee. Just increase once, check rarely.
    #  Must be really stuck on the boundary, to increase M.
    max_nb_of_ceilraise = 2
    for _ in range(max_nb_of_ceilraise):
        # If M needs to be fixed, skip the dynamical increase. Otherwise, check every 500 SP-FW iterations.
        alphak_diff = alphak_condgrad - alpha_k
        diff_sum = M_condgrad - sum_betak
        diff_sum_WIbeta = W_I_M - sum_WkIbetak
        diff_sum_WIu = W1mIuM - sum_Wk1mIubetak
        diff_sum_WIl = WIlM - sum_WkIlbetak

        # This is virtually unaffected by the primal subgradient initialization.
        if step_size_dict["type"] == "fw":
            step_size = (torch.ones((*alphak_condgrad.shape[:2], 1), device=alphak_condgrad.device, dtype=precision) *
                (1 / (step_size_dict['fw_start'] + iter))).view(unsqueezedk_shape)
        else:
            # w/o subgradient primal initialization, this underperforms. W/ it, it's much better.
            alpha_inner_grad_direction, alpha_dir_sqnorm = dual_vars.alphak_direction_info(alphak_diff, grad_alphak)
            beta_inner_grad_direction, beta_dir_sqnorm = dual_vars.betak_direction_info(lay_idx, weights, primal_vars,
                                                                                        diff_sum, atom_grad)
            primkm1_inner_grad_dir, primkm1_dir_sqrnorm = \
                primal_vars.primalsk_direction_info(lay_idx - 1, xkm1_diff, xkm1_grad)
            primk_inner_grad_dir, primk_dir_sqrnorm = \
                primal_vars.primalsk_direction_info(lay_idx, xk_diff, xk_grad, zk_diff=zk_diff, zk_grad=zk_grad)

            inner_grad_direction = alpha_inner_grad_direction + beta_inner_grad_direction - \
                                   (primkm1_inner_grad_dir + primk_inner_grad_dir)
            direction_norm = torch.sqrt(alpha_dir_sqnorm + beta_dir_sqnorm + primkm1_dir_sqrnorm + primk_dir_sqrnorm)
            step_size = gradlike_step_size(inner_grad_direction, direction_norm, iter, max_iters, step_size_dict).\
                view(unsqueezedk_shape)

        if c_fixed_M:
            break

        # If we get too close to M for alpha or beta, we increase its value.
        alpha_close_to_cap, beta_close_to_cap = adjust_artificial_caps_k(
            lay_idx, weights, alpha_k, alphak_diff, sum_betak, diff_sum, alphak_condgrad, M_condgrad, W_I_M, W1mIuM,
            WIlM, WI, new_alphak_M, new_betak_M, step_size, precision)

        if not (alpha_close_to_cap.any() or beta_close_to_cap.any()):
            break

    # Perform the SP-FW with the obtained directions.
    dual_vars.update_duals_from_alphak(lay_idx,  weights, alpha_k.addcmul(step_size, alphak_diff))
    dual_vars.update_duals_from_betak(
        lay_idx,
        weights,
        sum_betak.addcmul(step_size, diff_sum),
        sum_WkIbetak.addcmul(step_size.view(unsqueezedkm1_shape), diff_sum_WIbeta),
        sum_Wk1mIubetak.addcmul(step_size, diff_sum_WIu),
        sum_WkIlbetak.addcmul(step_size, diff_sum_WIl)
    )
    primal_vars.update_primals_from_primalsk(lay_idx - 1,
        primal_vars.xt[lay_idx - 1].addcmul(step_size.view(unsqueezedkm1_shape), xkm1_diff))
    primal_vars.update_primals_from_primalsk(
        lay_idx,
        primal_vars.xt[lay_idx].addcmul(step_size, xk_diff),
        primal_vars.zt[lay_idx - 1].addcmul(step_size, zk_diff)
    )


def spfw_full_step(weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, l_preacts, u_preacts, dual_vars,
                   primal_vars, iter, step_size_dict, precision=torch.float, fixed_M=False, max_iters=None):
    """
    Given list of layers, primal variables (instance of SaddlePrimalVars), pre and post activation bounds,
    perform a SP-FW step on all variables at once.
    :param step_size_dict: contains information on which step size rule to use, and with which parameters
    The update is performed in place on dual_vars.
    """
    batch_shape = dual_vars.sum_beta[0].shape[:2]
    c_fixed_M = fixed_M or (iter + 1) % 100 != 0  # allow for checks only rather rarely
    # Get LMOs and gradient information.
    dual_cd, grad_alpha, beta_atom_grad, WIs = dual_vars.dual_grad_lmo(
        weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision, fixed_M=c_fixed_M,
        store_grad=step_size_dict["type"] != "fw")
    primal_cd, x_grad, z_grad = primal_vars.primals_grad_lmo(dual_vars, clbs, cubs, precision,
        store_grad=step_size_dict["type"] != "fw")

    # the variable M is problematic w/ SP as there is no monotonicity guarantee. Just increase once, check rarely.
    #  Must be really stuck on the boundary, to increase M.
    max_nb_of_ceilraise = 2
    for _ in range(max_nb_of_ceilraise):
        # If M needs to be fixed, skip the dynamical increase. Otherwise, check every 500 SP-FW iterations.

        # This is virtually unaffected by the primal subgradient initialization.
        if step_size_dict["type"] == "fw":
            step_size = torch.ones(
                (*dual_vars.alpha[0].shape[:2], 1), device=dual_vars.alpha[0].device, dtype=precision) * \
                (1 / (step_size_dict['fw_start'] + iter))
        else:
            # w/o subgradient primal initialization, this underperforms. W/ it, it's much better.
            step_size = compute_full_gradlike_step_size(
                weights, dual_vars, dual_cd, grad_alpha, beta_atom_grad, primal_vars, primal_cd, x_grad, z_grad, iter,
                max_iters, step_size_dict)

        if c_fixed_M:
            break

        # If we get too close to M for alpha or beta, we increase its value.
        alpha_close_to_cap = torch.zeros((1,), device=dual_vars.alpha[0].device, dtype=torch.bool)
        beta_close_to_cap = torch.zeros((1,), device=dual_vars.sum_beta[0].device, dtype=torch.bool)
        for lay_idx in range(1, len(dual_cd.alpha) - 1):
            step_size = step_size.view((*batch_shape, *((1,) * (dual_cd.sum_beta[lay_idx].dim() - 2))))
            alphak_diff = dual_cd.alpha[lay_idx] - dual_vars.alpha[lay_idx]
            diff_sum = dual_cd.sum_beta[lay_idx] - dual_vars.sum_beta[lay_idx]
            c_alpha_close_to_cap, c_beta_close_to_cap = adjust_artificial_caps_k(
                lay_idx, weights, dual_vars.alpha[lay_idx], alphak_diff, dual_vars.sum_beta[lay_idx],
                diff_sum, dual_cd.alpha[lay_idx], dual_cd.sum_beta[lay_idx], dual_cd.sum_Wp1Ibetap1[lay_idx-1],
                dual_cd.sum_W1mIubeta[lay_idx], dual_cd.sum_WIlbeta[lay_idx], WIs[lay_idx-1],
                dual_vars.alpha_M[lay_idx], dual_vars.beta_M[lay_idx], step_size, precision)
            alpha_close_to_cap |= c_alpha_close_to_cap.any()
            beta_close_to_cap |= c_beta_close_to_cap.any()

        if not (alpha_close_to_cap.any() or beta_close_to_cap.any()):
            break

    # Perform the SP-FW step with the obtained directions.
    step_size = step_size.view((*batch_shape, *((1,) * (dual_cd.sum_beta[0].dim() - 2))))
    primal_vars.update_primals_from_primalsk(0, (1 - step_size) * primal_vars.xt[0] + step_size * primal_cd.xt[0])
    for lay_idx in range(1, len(dual_cd.alpha) - 1):
        step_size = step_size.view((*batch_shape, *((1,) * (dual_cd.sum_beta[lay_idx].dim() - 2))))
        unsqueezedkm1_shape = (*batch_shape, *((1,) * (dual_cd.sum_Wp1Ibetap1[lay_idx-1].dim() - 2)))
        dual_vars.update_duals_from_alphak(
            lay_idx,  weights, (1 - step_size) * dual_vars.alpha[lay_idx] + step_size * dual_cd.alpha[lay_idx])
        dual_vars.update_duals_from_betak(
            lay_idx,
            weights,
            (1 - step_size) * dual_vars.sum_beta[lay_idx] + step_size * dual_cd.sum_beta[lay_idx],
            (1 - step_size).view(unsqueezedkm1_shape) * dual_vars.sum_Wp1Ibetap1[lay_idx-1] +
                step_size.view(unsqueezedkm1_shape) * dual_cd.sum_Wp1Ibetap1[lay_idx-1],
            (1 - step_size) * dual_vars.sum_W1mIubeta[lay_idx] + step_size * dual_cd.sum_W1mIubeta[lay_idx],
            (1 - step_size) * dual_vars.sum_WIlbeta[lay_idx] + step_size * dual_cd.sum_WIlbeta[lay_idx]
        )
        primal_vars.update_primals_from_primalsk(
            lay_idx,
            (1 - step_size) * primal_vars.xt[lay_idx] + step_size * primal_cd.xt[lay_idx],
            (1 - step_size) * primal_vars.zt[lay_idx-1] + step_size * primal_cd.zt[lay_idx-1]
        )


def gradlike_step_size(inner_grad_direction, direction_norm, iter, max_iters, step_size_dict):
    """
    Simulate a linearly decreasing step size (between two values init_step and fin_step) for gradient descent, for a
    given FW direction.
    This is done by normalizing the decreasing step size by multiplying with the cosine distance between the current
    direction and the gradient, and the norm of the gradient.
    :param inner_grad_direction: inner product between gradient and current direction (batch_size x 1)
    :param direction_norm: norm of the direction (batch_size x 1)
    :return: the [0,1]-clamped step size (batch_size x 1)
    """
    init_step = step_size_dict["initial_step_size"]
    fin_step = step_size_dict["final_step_size"]
    step_size = torch.ones_like(inner_grad_direction) * \
                (init_step + (iter / max_iters) * (fin_step - init_step)) * inner_grad_direction / direction_norm
    step_size[direction_norm == 0] = 0
    step_size.clamp_(0, 1)
    return step_size


def compute_full_gradlike_step_size(weights, dual_vars, dual_cd, grad_alpha, beta_atom_grad,
                                    primal_vars, primal_cd, x_grad, z_grad, iter, max_iters, step_size_dict):
    """
    Given the list of network weights, current dual and primal variables (dual_vars, primal_vars) along with the current
    conditional gradient (dual_cd, primal_cd) as instances of SaddleDualVars and SaddlePrimalVars, specifications
    for the current gradients of primals/duals as lists of tensors (x_grad, z_grad, beta_atom_grad) and specifications
    for gradlike_step_size (iter, max_iters, step_size_dict), simulate a linearly decreasing step size for gradient
    descent, on the full set of primal and dual variables (no blocks).
    :return: the [0,1]-clamped step size (batch_size x 1)
    """
    xkm1_diff = primal_cd.xt[0] - primal_vars.xt[0]
    prim0_inner_grad_dir, prim0_dir_sqrnorm = primal_vars.primalsk_direction_info(0, xkm1_diff, x_grad[0])
    inner_grad_direction = - prim0_inner_grad_dir
    direction_norm = prim0_dir_sqrnorm

    for lay_idx in range(1, len(dual_cd.alpha) - 1):
        alphak_diff = dual_cd.alpha[lay_idx] - dual_vars.alpha[lay_idx]
        diff_sum = dual_cd.sum_beta[lay_idx] - dual_vars.sum_beta[lay_idx]
        xk_diff = primal_cd.xt[lay_idx] - primal_vars.xt[lay_idx]
        zk_diff = primal_cd.zt[lay_idx - 1] - primal_vars.zt[lay_idx - 1]

        alpha_inner_grad_direction, alpha_dir_sqnorm = dual_vars.alphak_direction_info(alphak_diff, grad_alpha[lay_idx-1])
        beta_inner_grad_direction, beta_dir_sqnorm = dual_vars.betak_direction_info(lay_idx, weights, primal_vars,
                                                                                    diff_sum, beta_atom_grad[lay_idx-1])

        primk_inner_grad_dir, primk_dir_sqrnorm = \
            primal_vars.primalsk_direction_info(lay_idx, xk_diff, x_grad[lay_idx], zk_diff=zk_diff,
                                                zk_grad=z_grad[lay_idx-1])

        inner_grad_direction += alpha_inner_grad_direction + beta_inner_grad_direction - primk_inner_grad_dir
        direction_norm += alpha_dir_sqnorm + beta_dir_sqnorm + primk_dir_sqrnorm

    direction_norm = torch.sqrt(direction_norm)
    step_size = gradlike_step_size(inner_grad_direction, direction_norm, iter, max_iters, step_size_dict)
    return step_size


def adjust_artificial_caps_k(lay_idx, weights, alpha_k, alphak_diff, sum_betak, diff_sum, alphak_condgrad, M_condgrad,
                             W_I_M, W1mIuM, WIlM, WI, new_alphak_M, new_betak_M, step_size, precision):
    """
    Given the current dual variables, the LMO directions and the current M values associated with layer lay_idx,
    determine if taking a step of size step_size will too close or out of the artificial boundary. If this is the case,
    increase M and re-compute the LMO directions accordingly (operates in place).
    :return: two tensors indicating whether alpha and beta are close/out of the cap (respectively)
    """
    lin_k = weights[lay_idx-1]
    # If we get too close to M for alpha or beta, we increase its value.
    alpha_close_to_cap = (alpha_k.addcmul(step_size, alphak_diff) >= 0.99 * new_alphak_M)
    beta_close_to_cap = (sum_betak.addcmul(step_size, diff_sum) >= 0.99 * new_betak_M)

    if alpha_close_to_cap.any().item():
        guaranteed_increase_factor = 2.0
        # If M is different for each batch entry, change only the entries close to the cap.
        if len(new_alphak_M) > 1:
            mul_factor = (alpha_close_to_cap.type(precision) * max(step_size.max(), 1) * guaranteed_increase_factor +
                          (~alpha_close_to_cap).type(precision))
        else:
            mul_factor = max(step_size.max(), 1) * guaranteed_increase_factor
        alphak_condgrad[:] = mul_factor * alphak_condgrad
        new_alphak_M[:] = mul_factor * new_alphak_M

    if beta_close_to_cap.any().item():
        guaranteed_increase_factor = 2.0
        # If M is different for each batch entry, change only the entries close to the cap.
        if len(new_alphak_M) > 1:
            mul_factor = (beta_close_to_cap.type(precision) * max(step_size.max(), 1) * guaranteed_increase_factor +
                          (~beta_close_to_cap).type(precision))
        else:
            mul_factor = max(step_size.max(), 1) * guaranteed_increase_factor
        masked_op = anderson_optimization.MaskedLinearOp(lin_k) if type(weights[lay_idx-1]) is utils.LinearOp \
            else anderson_optimization.MaskedConvOp(lin_k, W_I_M, M_condgrad)
        masked_op.set_WI(WI)
        M_condgrad[:] = mul_factor * M_condgrad
        W_I_M[:] = masked_op.backward(M_condgrad)
        W1mIuM[:] = mul_factor * W1mIuM
        WIlM[:] = mul_factor * WIlM
        new_betak_M[:] = mul_factor * new_betak_M

    return alpha_close_to_cap, beta_close_to_cap


def primals_subgrad_initializer(weights, masked_ops, clbs, cubs, nubs, l_checks, u_checks, l_preacts, u_preacts,
                                dual_vars, primal_class, params, primal_init=None):
    """
    ADAM subgradient descent for the primal problem (via the Lagrangian) of the saddle point formulation.
    Starts by operating on big-M's dual variables only, then moves on to include all Anderson dual variables.
    Used to initialize the primals for Anderson SP-FW.
    """
    if primal_init is not None:
        primal_vars = primal_init
    else:
        primal_vars = primal_class.mid_box_initialization(dual_vars, clbs, cubs)

    init_step_size = params['initial_step_size']
    final_step_size = params['final_step_size']

    # Adam-related quantities.
    exp_avg = primal_vars.zero_like()
    exp_avg_sq = primal_vars.zero_like()
    beta_1 = params['betas'][0]
    beta_2 = params['betas'][1]

    # Do a certain number of iterations using bigm's dual oracle, then switch to Anderson (resetting step sizes).
    n_bigm_iters = params["nb_bigm_iter"]
    n_anderson_iters = params["nb_anderson_iter"]
    for outer_it in itertools.count():
        if outer_it > n_bigm_iters + n_anderson_iters:
            break

        # Reset Adam stats when we switch from big-M to Anderson.
        if outer_it == n_bigm_iters+1:
            exp_avg = primal_vars.zero_like()
            exp_avg_sq = primal_vars.zero_like()

        bigm_flag = outer_it <= n_bigm_iters
        primal_vars_subg = compute_primal_subgradient(weights, primal_vars, dual_vars, masked_ops, nubs, l_checks,
                                                      u_checks, l_preacts, u_preacts, bigm_flag)

        if outer_it <= n_bigm_iters:
            n_current_iters = outer_it
            step_size = init_step_size + ((n_current_iters + 1) / n_bigm_iters) * (final_step_size - init_step_size)
        else:
            n_current_iters = outer_it - n_bigm_iters
            step_size = init_step_size + ((n_current_iters + 1) / n_anderson_iters) * (final_step_size - init_step_size)

        # Perform an Adam step
        bias_correc1 = 1 - beta_1 ** (n_current_iters+1)
        bias_correc2 = 1 - beta_2 ** (n_current_iters+1)
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta_1).add_(1 - beta_1, primal_vars_subg)
        exp_avg_sq.mul_(beta_2).addcmul_(1 - beta_2, primal_vars_subg, primal_vars_subg)
        denom = (exp_avg_sq.sqrt().div_cte_(math.sqrt(bias_correc2))).add_cte_(1e-8)
        step_size = step_size / bias_correc1
        primal_vars.projected_linear_combination(step_size, exp_avg.div(denom), clbs, cubs)

        # normal subgradient ascent
        # primal_vars.projected_linear_combination(step_size, primal_vars_subg, clbs, cubs)

    return primal_vars


def compute_primal_subgradient(weights, primal_vars, dual_vars, masked_ops, nubs, l_checks, u_checks, l_preacts,
                               u_preacts, bigm_flag, precision=torch.float):
    """
    Given the network layers, post- and pre-activation bounds as lists of tensors, and primal variables as PrimalVars,
    compute the subgradient of the primal objective, with the duals constrained to the artificial simplices.
    This is done for all primal variables at once.
    Used to initialize the primal variables for Anderson SP-FW.
    :param bigm_flag: whether to use bigm's duals (or Andersons, much more expensive)
    :return: SaddlePrimalVars instance representing the subgradient for the primal variables
    """
    xt_subg = [None] * len(dual_vars.sum_beta)
    zt_subg = [None] * (len(dual_vars.sum_beta) - 1)

    alphak_maxp1 = dual_vars.alpha[-1]
    W_I_maxp1 = torch.zeros_like(dual_vars.alpha[-2])
    for lay_idx in range(len(dual_vars.sum_beta)-1, -1, -1):
        # compute and store the subgradients.
        lin_kp1 = weights[lay_idx]
        if lay_idx == 0:
            xt_subg[0] = -lin_kp1.backward(alphak_maxp1) + W_I_maxp1
        else:
            lin_k = weights[lay_idx - 1]
            # solve the inner problems.
            alphak_max, _ = dual_vars.alphak_grad_lmo(lay_idx, weights, primal_vars, precision)
            M_max, W_I_max, WIlmax, W1mIumax, _, _ = dual_vars.betak_grad_lmo(
                lay_idx, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision,
                bigm_only=bigm_flag)

            xt_subg[lay_idx] = alphak_max - lin_kp1.backward(alphak_maxp1) - M_max + W_I_maxp1
            zt_subg[lay_idx-1] = M_max * lin_k.get_bias() + WIlmax + W1mIumax

            alphak_maxp1 = alphak_max
            W_I_maxp1 = W_I_max
    return primal_vars.__class__(xt_subg, zt_subg)


class SaddleDualVars(anderson_optimization.DualVars):
    """
    Class representing the dual variables for the "saddle point frank wolfe" deivation. These are
    alpha_0, alpha_1, beta (through its sufficient statistics), and their functions f and g.
    The norms of alpha and beta are kept for the purposes of changing the simplex size.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for all variables except alpha_1.
    """

    def __init__(self, alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, alpha_M=None, beta_M=None):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        :param alpha_M: (beta_M defined analogously) upper cap for the artificial simplex imposed on the dual variables.
            (iterator containing a cap per layer)
        """
        super().__init__(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs)
        self.alpha_M = [1e2] * len(alpha) if alpha_M is None else alpha_M
        self.beta_M = [1e2] * len(sum_beta) if beta_M is None else beta_M

    @staticmethod
    def from_super_class(super_instance, alpha_M=None, beta_M=None):
        """
        Return an instance of this class from an instance of the super class.
        """
        return SaddleDualVars(super_instance.alpha, super_instance.sum_beta, super_instance.sum_Wp1Ibetap1,
                            super_instance.sum_W1mIubeta, super_instance.sum_WIlbeta, super_instance.fs,
                              super_instance.gs, alpha_M=alpha_M, beta_M=beta_M)

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size, alpha_M=None, beta_M=None):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        base_duals = anderson_optimization.DualVars.naive_initialization(weights, additional_coeffs, device, input_size)
        return SaddleDualVars.from_super_class(base_duals, alpha_M=alpha_M, beta_M=beta_M)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, clbs, cubs, lower_bounds, upper_bounds, bigm_M_factor=5.0):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual variables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        :param bigm_M_factor: constant factor of the max big-m variables to use for M
        """
        base_duals, primals = anderson_optimization.DualVars.bigm_initialization(bigm_duals, weights, clbs, cubs,
                                                                                 lower_bounds, upper_bounds)

        # Let's infer M values from the big-M initialization.
        alpha_M, beta_M = SaddleDualVars.get_M_from_init(base_duals.alpha, base_duals.sum_beta, M_factor=bigm_M_factor)

        return SaddleDualVars.from_super_class(base_duals, alpha_M=alpha_M, beta_M=beta_M), \
               SaddlePrimalVars.from_super_class(primals)

    @staticmethod
    def cut_initialization(cut_DualVars, cut_PrimalVars, cut_M_factor=1.2):
        """
        Initializing from the cut algorithm
        """
        alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs = cut_DualVars.as_saddlep_initialization()

        # Let's infer M values from the cuts initialization.
        alpha_M, beta_M = SaddleDualVars.get_M_from_init(alpha, sum_beta, M_factor=cut_M_factor)

        return SaddleDualVars(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, alpha_M=alpha_M,
                              beta_M=beta_M), \
               SaddlePrimalVars.from_super_class(cut_PrimalVars)

    @staticmethod
    def get_M_from_init(init_alphas, init_sum_betas, M_factor=1.1):
        """
        Given values for alpha and sums of beta values to use as initialisation, infer M values.
        We choose as M value (artificial upper cap) the corresponding dual value from the initialisation.
        (performs better than the max only, which creates an unnecessarily large feasible region.
        If this value is 0, we 1/100 of the max value across the batch entry.
        """
        alpha_M = []
        for alphak in init_alphas:
            max_alphak, _ = torch.max(alphak.view(*alphak.shape[:2], -1), dim=2)
            max_alphak = max_alphak.view((*alphak.shape[:2], *((1,) * (alphak.dim() - 2)))).expand_as(alphak)
            alphak_almostzero = (alphak <= 1e-8)
            alphak_cap = torch.where(alphak_almostzero, max_alphak / 100, M_factor * alphak)
            alpha_M.append(alphak_cap)

        beta_M = []
        for sum_betak in init_sum_betas:
            max_betak, _ = torch.max(sum_betak.view(*sum_betak.shape[:2], -1), dim=2)
            max_betak = max_betak.view((*sum_betak.shape[:2], *((1,) * (sum_betak.dim() - 2)))).expand_as(sum_betak)
            sum_betak_almostzero = (sum_betak <= 1e-8)
            sum_betak_cap = torch.where(sum_betak_almostzero, max_betak / 100, M_factor * sum_betak)
            beta_M.append(sum_betak_cap)
        return alpha_M, beta_M

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return SaddleDualVars(
            copy.deepcopy(self.alpha), copy.deepcopy(self.sum_beta), copy.deepcopy(self.sum_Wp1Ibetap1),
            copy.deepcopy(self.sum_W1mIubeta), copy.deepcopy(self.sum_WIlbeta), copy.deepcopy(self.fs),
            copy.deepcopy(self.gs), alpha_M=copy.deepcopy(self.alpha_M), beta_M=copy.deepcopy(self.beta_M))

    def update_duals_from_alphak(self, lay_idx, weights, new_alpha_k, new_alphak_M=None):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        super().update_duals_from_alphak(lay_idx, weights, new_alpha_k)
        if new_alphak_M is not None:
            self.alpha_M[lay_idx] = new_alphak_M

    def update_duals_from_betak(self, lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak,
                                new_sum_WkIlbetak, new_betak_M=None):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        super().update_duals_from_betak(lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak,
                                        new_sum_WkIlbetak)
        if new_betak_M is not None:
            self.beta_M[lay_idx] = new_betak_M

    def alphak_grad_lmo(self, lay_idx, weights, primal_vars, precision):
        """
        Given list of layers and çurrent primal variables (instance of SaddlePrimalVars),
        compute and return the linear minimization oracle of alpha_k with its gradient, and the gradient itself.
        """
        # Compute the gradient over alphak
        grad_alphak = weights[lay_idx - 1].forward(primal_vars.xt[lay_idx-1]) - primal_vars.xt[lay_idx]

        # Let's compute the best atom in the dictionary according to the LMO.
        # If grad_alphak > 0, lmo_alphak = M
        lmo_alphak = (grad_alphak > 0).type(precision) * self.alpha_M[lay_idx]
        return lmo_alphak, grad_alphak

    def alphak_direction_info(self, alphak_diff, grad_alphak):
        """
        Given the current LMO direction (n.b. not atom) of alphas, and the gradient over alpha_k,
        compute the inner product of the direction at layer lay_idx with the gradient, and the direction's squared norm.
        """
        alpha_inner_grad_direction = utils.bdot(alphak_diff, grad_alphak).unsqueeze(-1)
        alpha_dir_sqnorm = utils.bl2_norm(alphak_diff).unsqueeze(-1)
        return alpha_inner_grad_direction, alpha_dir_sqnorm

    def betak_grad_lmo(self, lay_idx, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars,
                       precision, only_grad=False, bigm_only=False):
        """
        Given list of layers, primal variables (instance of SaddlePrimalVars), pre and post activation bounds,
        compute and return the linear minimization oracle of beta_k with its gradient, and the gradient itself (over
        the non-zero entries of the conditional gradient only).
        The LMO is expressed in terms of the four sufficient statistics for beta.
        :param bigm_only: if True, the LMO is going to be computed only the two big-M constraints only (much cheaper).
        """

        # reference current layer variables via shorter names
        lin_k = weights[lay_idx - 1]
        zk = primal_vars.zt[lay_idx-1]
        xk = primal_vars.xt[lay_idx]
        xkm1 = primal_vars.xt[lay_idx-1]
        M = self.beta_M[lay_idx]
        l_preact = l_preacts[lay_idx].unsqueeze(1)
        u_preact = u_preacts[lay_idx].unsqueeze(1)

        # Compute the gradients for the Big-M variables.
        beta0_grad = xk - zk * u_preact
        beta1_grad = xk - lin_k.forward(xkm1) + (1 - zk) * l_preact

        if not bigm_only:
            # Compute the optimal weight mask from the exponential family...
            masked_op, Istar_k, exp_k_grad, unfolded_zk, l_check, u_check = self.anderson_oracle(
                lay_idx, weights, masked_ops, nubs, l_checks, u_checks, primal_vars, do_intermediates=False)

            # Let's compute the actual LMO via the sufficient statistics.
            lmo_is_beta0 = ((beta0_grad > beta1_grad) & (beta0_grad > exp_k_grad))
            lmo_is_beta1 = (beta1_grad > beta0_grad) & (beta1_grad > exp_k_grad)
            # gradient of the selected outer corner of the simplex
            atom_grad = torch.where(lmo_is_beta0, beta0_grad, torch.where(lmo_is_beta1, beta1_grad, exp_k_grad))
        else:
            # Let's compute the actual LMO via the sufficient statistics.
            lmo_is_beta0 = beta0_grad > beta1_grad
            lmo_is_beta1 = beta1_grad > beta0_grad
            atom_grad = torch.where(lmo_is_beta0, beta0_grad, beta1_grad)

        M_atom_mask = M * (atom_grad >= 0).type(precision)  # compared with 0 to do FW

        if not only_grad:
            # express the LMO through its sufficient statistics.
            beta_sum_lmo = M_atom_mask
            if not bigm_only:
                if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                    atom_beta0_check = lin_k.unfold_output(lmo_is_beta0, gather=self.and_set[lay_idx - 1])
                    atom_beta1_check = lin_k.unfold_output(lmo_is_beta1, gather=self.and_set[lay_idx - 1])
                else:
                    atom_beta0_check = lmo_is_beta0
                    atom_beta1_check = lmo_is_beta1

                atom_I = torch.where(
                    atom_beta0_check.unsqueeze(3), torch.zeros_like(Istar_k),
                    torch.where(atom_beta1_check.unsqueeze(3), torch.ones_like(Istar_k), Istar_k))
                masked_op.set_mask(atom_I)
                beta_WI_lmo = masked_op.backward(M_atom_mask)
                beta_WIl_lmo = torch.where(lmo_is_beta1, l_preact - lin_k.get_bias(),
                    masked_op.forward(l_check, no_unsqueeze=True, add_bias=False)) * M_atom_mask
                beta_W1mIu_lmo = torch.where(lmo_is_beta0, u_preact - lin_k.get_bias(),
                    nubs[lay_idx - 1].unsqueeze(1) - masked_op.forward(u_check, no_unsqueeze=True)) * M_atom_mask
                WI = masked_op.WI
            else:
                beta_WI_lmo = lin_k.backward(M_atom_mask * lmo_is_beta1.type(precision))
                beta_WIl_lmo = lmo_is_beta1.type(precision) * (l_preact - lin_k.get_bias()) * M_atom_mask
                beta_W1mIu_lmo = lmo_is_beta0.type(precision) * (u_preact - lin_k.get_bias()) * M_atom_mask
                WI = None
        else:
            beta_sum_lmo = None; beta_WI_lmo = None; beta_WIl_lmo = None; beta_W1mIu_lmo = None; WI = None

        return beta_sum_lmo, beta_WI_lmo, beta_WIl_lmo, beta_W1mIu_lmo, atom_grad, WI

    def dual_grad_lmo(self, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision,
                      fixed_M=False, store_grad=False):
        """
        Given list of layers, primal variables (instance of SaddlePrimalVars), pre and post activation bounds,
        compute and return the full conditional gradient of the duals, and the gradient itself (for beta, this is the
        gradient over the non-zero entries of the conditional gradient only).
        :return: conditional gradient as SaddleDualVars, lists of tensors (starting from k=1) for grad_alpha,
            beta_atom_grad, WI
        """
        cd_alpha = [torch.zeros_like(self.alpha[0])]
        cd_sum_beta = [torch.zeros_like(self.sum_beta[0])]
        cd_sum_Wp1Ibetap1 = []
        cd_sum_W1mIubeta = [torch.zeros_like(self.sum_W1mIubeta[0])]
        cd_sum_WIlbeta = [torch.zeros_like(self.sum_WIlbeta[0])]
        grad_alpha = []; beta_atom_grads = []; WIs = []
        for lay_idx in range(1, len(self.alpha) - 1):
            # compute alpha_k's gradient and its LMO over it.
            alphak_condgrad, grad_alphak = self.alphak_grad_lmo(lay_idx, weights, primal_vars, precision)
            cd_alpha.append(alphak_condgrad)
            grad_alpha.append(grad_alphak) if store_grad else grad_alpha.append(None)

            M_condgrad, W_I_M, WIlM, W1mIuM, atom_grad, WI = self.betak_grad_lmo(
                lay_idx, weights, masked_ops, nubs, l_checks, u_checks, l_preacts, u_preacts, primal_vars, precision)
            cd_sum_beta.append(M_condgrad)
            cd_sum_Wp1Ibetap1.append(W_I_M)
            cd_sum_W1mIubeta.append(W1mIuM)
            cd_sum_WIlbeta.append(WIlM)
            beta_atom_grads.append(atom_grad) if store_grad else beta_atom_grads.append(None)
            WIs.append(WI) if (not fixed_M) or store_grad else WIs.append(None)
        cd_sum_Wp1Ibetap1.append(torch.zeros_like(self.sum_Wp1Ibetap1[-1]))

        return SaddleDualVars(cd_alpha, cd_sum_beta, cd_sum_Wp1Ibetap1, cd_sum_W1mIubeta, cd_sum_WIlbeta, None, None), \
               grad_alpha, beta_atom_grads, WIs

    def betak_inner_grad(self, lay_idx, weights, primal_vars):
        """
        Given the primal variables, compute the inner product of the current betak iterate with its
        gradient.
        """
        xk = primal_vars.xt[lay_idx]
        xkm1 = primal_vars.xt[lay_idx-1]
        zk = primal_vars.zt[lay_idx-1]

        inner_beta_sum = (self.sum_beta[lay_idx] * (xk - weights[lay_idx - 1].get_bias() * zk)).\
            sum(dim=tuple(range(2, self.sum_beta[lay_idx].dim())))
        inner_WI = (-self.sum_Wp1Ibetap1[lay_idx - 1] * xkm1).\
            sum(dim=tuple(range(2, self.sum_Wp1Ibetap1[lay_idx - 1].dim())))
        inner_WIl = ((1 - zk) * self.sum_WIlbeta[lay_idx]).\
            sum(dim=tuple(range(2, self.sum_WIlbeta[lay_idx].dim())))
        inner_W1mIu = (-zk * self.sum_W1mIubeta[lay_idx]).\
            sum(dim=tuple(range(2, self.sum_W1mIubeta[lay_idx].dim())))
        return (inner_beta_sum + inner_WI + inner_WIl + inner_W1mIu).unsqueeze(-1)

    def betak_direction_info(self, lay_idx, weights, primal_vars, diff_sum, atom_grad):
        """
        Given the primal variables, the current LMO direction (n.b. not atom) expressed as sums over exponential betas,
        and the gradient over the optimal atoms according to the LMO,
        compute the inner product of the direction at layer lay_idx with the gradient, and the direction's squared norm.
        """
        # the inner product is computed as the difference of inner products with the atoms and current point
        beta_inner_grad_direction = (torch.clamp(atom_grad, 0, None) * self.beta_M[lay_idx]).\
            sum(dim=tuple(range(2, atom_grad.dim()))).unsqueeze(-1)-self.betak_inner_grad(lay_idx, weights, primal_vars)
        beta_dir_sqnorm = utils.bl2_norm(diff_sum).unsqueeze(-1)  # TODO: this is a lower bound
        return beta_inner_grad_direction, beta_dir_sqnorm


class SaddlePrimalVars(anderson_optimization.PrimalVars):
    """
    Class representing the primal variables xt, zt for the "saddle point frank wolfe" derivation.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for xt and 1 to n-1 for zt.
    """

    @staticmethod
    def from_super_class(super_instance):
        """
        Return an instance of this class from an instance of the super class.
        """
        return SaddlePrimalVars(super_instance.xt, super_instance.zt)

    @staticmethod
    def mid_box_initialization(dual_vars, clbs, cubs):
        """
        Initialize the primal variables (anchor points) to the mid-point of the box constraints (halfway through each
        variable's lower and upper bounds).
        """
        primals = anderson_optimization.PrimalVars.mid_box_initialization(dual_vars, clbs, cubs)
        return SaddlePrimalVars.from_super_class(primals)

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return SaddlePrimalVars(copy.deepcopy(self.xt), copy.deepcopy(self.zt))

    @staticmethod
    def xk_grad_lmo(lay_idx, dual_vars, clbs, cubs):
        """
        Given list of post-activation bounds and çurrent dual variables (instance of SaddleDualVars),
        compute and return the linear minimization oracle of x_k its gradients, and the gradient itself.
        """
        xk_grad = -dual_vars.fs[lay_idx]
        xk_lmo = torch.where(xk_grad >= 0, clbs[lay_idx].unsqueeze(1), cubs[lay_idx].unsqueeze(1))
        return xk_lmo, xk_grad

    @staticmethod
    def zk_grad_lmo(lay_idx, dual_vars, precision):
        """
        Given çurrent dual variables (instance of SaddleDualVars),
        compute and return the linear minimization oracle of z_k its gradients, and the gradient itself.
        """
        zk_grad = -dual_vars.gs[lay_idx - 1]
        if lay_idx > 0:
            z_k_lmo = (zk_grad <= 0).type(precision)
        else:
            # g_k is defined from 1 to n - 1.
            z_k_lmo = None
        return z_k_lmo, zk_grad

    def primals_grad_lmo(self, dual_vars, clbs, cubs, precision, store_grad=False):
        """
        Given dual variables (instance of SaddleDualVars), post activation bounds,
        compute and return the full conditional gradient of the primals, and the gradient itself.
        :return: conditional gradient as SaddlePrimalVars, lists of tensors for grad_x, grad_z
        """
        xs = []; zs = []
        x_grad = []; z_grad = []
        xk_cd, xk_grad = self.xk_grad_lmo(0, dual_vars, clbs, cubs)
        xs.append(xk_cd)
        x_grad.append(xk_grad) if store_grad else x_grad.append(None)
        for lay_idx in range(1, len(dual_vars.alpha) - 1):
            xk_cd, xk_grad = self.xk_grad_lmo(lay_idx, dual_vars, clbs, cubs)
            xs.append(xk_cd)
            x_grad.append(xk_grad)
            zk_cd, zk_grad = self.zk_grad_lmo(lay_idx, dual_vars, precision)
            zs.append(zk_cd)
            z_grad.append(zk_grad) if store_grad else z_grad.append(None)
        return SaddlePrimalVars(xs, zs), x_grad, z_grad

    @staticmethod
    def primalsk_direction_info(lay_idx, xk_diff, xk_grad, zk_diff=None, zk_grad=None):
        """
        Given the current LMO direction (n.b. not atom) of primals, and the gradient over x_k and z_k,
        compute the inner product of the direction at layer lay_idx with the gradient, and the direction's squared norm.
        """
        primal_inner_grad_dir = utils.bdot(xk_diff, xk_grad).unsqueeze(-1)
        primal_dir_sqrnorm = utils.bl2_norm(xk_diff)
        if lay_idx > 0 and zk_diff is not None:
            primal_inner_grad_dir += utils.bdot(zk_diff, zk_grad).unsqueeze(-1)
            primal_dir_sqrnorm += utils.bl2_norm(zk_diff)
        return primal_inner_grad_dir, primal_dir_sqrnorm.unsqueeze(-1)

    def update_primals_from_primalsk(self, lay_idx, new_xk, new_zk=None):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        self.xt[lay_idx] = new_xk
        if lay_idx > 0 and new_zk is not None:
            self.zt[lay_idx-1] = new_zk


class SaddleAndersonInit(anderson_optimization.AndersonPInit):
    """
    Parent Init class for Anderson-relaxation-based solvers.
    """
    def __init__(self, parent_duals, parent_primals):
        # parent_duals are the dual values (instance of DualVars from bigm_optimization) at parent termination
        # parent_primals are the primal values (instance of SaddlePrimalVars) at parent termination
        super().__init__(parent_duals)
        self.primals = parent_primals

    def to_cpu(self):
        # Move content to cpu
        super().to_cpu()
        for varname in self.primals.__dict__:
            self.primals.__dict__[varname] = [cvar.cpu() for cvar in self.primals.__dict__[varname]]

    def to_device(self, device):
        # Move content to device "device"
        super().to_device(device)
        for varname in self.primals.__dict__:
            self.primals.__dict__[varname] = [cvar.to(device) for cvar in self.primals.__dict__[varname]]

    def as_stack(self, stack_size):
        # Repeat the content of this parent init to form a stack of size "stack_size"
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(self.do_stack_list(varset, stack_size))
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append(self.do_stack_list(varset, stack_size))
        return SaddleAndersonInit(bigm_optimization.DualVars(*stacked_dual_list), SaddlePrimalVars(*stacked_primal_list))

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        super().set_stack_parent_entries(parent_solution, batch_idx)
        for varname in self.primals.__dict__:
            for x_idx in range(len(self.primals.__dict__[varname])):
                self.set_parent_entries(self.primals.__dict__[varname][x_idx],
                                        parent_solution.primals.__dict__[varname][x_idx], batch_idx)

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(self.get_entry_list(varset, batch_idx))
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append(self.get_entry_list(varset, batch_idx))
        return SaddleAndersonInit(bigm_optimization.DualVars(*stacked_dual_list), SaddlePrimalVars(*stacked_primal_list))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.beta_0, self.duals.beta_1, self.duals.fs,
                            self.duals.gs, self.duals.alpha_back, self.duals.beta_1_back]
        for varset in constructor_vars:
            stacked_dual_list.append(self.lb_only_list(varset))
        stacked_primal_list = []
        for varset in [self.primals.xt, self.primals.zt]:
            stacked_primal_list.append(self.lb_only_list(varset))
        return SaddleAndersonInit(bigm_optimization.DualVars(*stacked_dual_list), SaddlePrimalVars(*stacked_primal_list))