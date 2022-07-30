import torch
import copy
from plnn_bounds.proxlp_solver.utils import bdot
import math


class ByPairsDecomposition:

    def __init__(self, init_method):
        initializers = {'KW': self.get_initial_kw_solution,
                        'naive': self.get_initial_naive_solution}
        assert init_method in initializers, "Unknown initialization type"
        self.initial_dual_solution = initializers[init_method]

    @staticmethod
    def get_optim_primal(weights, additional_coeffs,
                         lower_bounds, upper_bounds,
                         dual_vars):
        rhos = dual_vars.rhos
        batch_size = rhos[0].shape[0]

        zahats = []
        zbhats = []

        # TODO: Is there some problem if the additional coeffs are for the
        # first few layers?

        # Optimize for the first layer because it is special
        lin_0 = torch.zeros_like(lower_bounds[0])
        lin_hat_1 = -rhos[0]
        za_0, zahat_1 = opt_lin_domain(weights[0], lin_0, lin_hat_1,
                                       lower_bounds[0], upper_bounds[0])
        zahats.append(zahat_1)

        # Optimize for all the other layers
        last_idx = len(weights)-1
        for lay_idx, (layer, lb_k, ub_k) in enumerate(zip(weights,
                                                          lower_bounds,
                                                          upper_bounds)):
            if lay_idx == 0:
                # We already dealt with it just above
                continue

            lin_hat_k = rhos[lay_idx-1]
            if lay_idx == last_idx:
                lin_hat_kp1 = 0
            else:
                lin_hat_kp1 = -rhos[lay_idx]
            if (lay_idx+1) in additional_coeffs:
                lin_hat_kp1 += additional_coeffs[lay_idx+1]

            zbhat_k, zahat_kp1 = opt_relu_pair(layer, lin_hat_k, lin_hat_kp1,
                                               lb_k, ub_k)
            zbhats.append(zbhat_k)
            zahats.append(zahat_kp1)

        primal = PrimalVarSet(zahats, zbhats, za_0)

        return primal

    @staticmethod
    def get_optim_primal_layer(lay_idx, n_layers, layer, additional_coeffs, lb_k, ub_k, dual_vars):

        # Run get_optim_primal_layer only on subproblem k=lay_idx. Returns the
        # computed partial conditional gradients (zahat_kp1, zbhat_k)
        last_idx = n_layers - 1

        if lay_idx == 0:
            # Optimize for the first layer because it is special
            lin_0 = torch.zeros_like(lb_k)
            lin_hat_1 = -dual_vars.primal_grad[0]
            za_0, zahat_1 = opt_lin_domain(layer, lin_0, lin_hat_1,
                                           lb_k, ub_k)
            return SubproblemCondGrad(lay_idx, zahat_1, None, z0=za_0)

        # Optimize for all the other layers
        lin_hat_k = dual_vars.primal_grad[lay_idx-1]
        if lay_idx == last_idx:
            lin_hat_kp1 = 0
        else:
            lin_hat_kp1 = -dual_vars.primal_grad[lay_idx]
        if (lay_idx+1) in additional_coeffs:
            lin_hat_kp1 += additional_coeffs[lay_idx+1]

        # TODO The last layer could be treated differently and we would
        # obtain a better result. Think about it and update?
        zbhat_k, zahat_kp1 = opt_relu_pair(layer, lin_hat_k, lin_hat_kp1,
                                           lb_k, ub_k)

        return SubproblemCondGrad(lay_idx, zahat_kp1, zbhat_k)

    @staticmethod
    def get_initial_kw_solution(weights, additional_coeffs,
                                lower_bounds, upper_bounds):
        assert len(additional_coeffs) > 0
        rhos = []

        final_lay_idx = len(weights)
        if final_lay_idx in additional_coeffs:
            # There is a coefficient on the output of the network
            rho = -additional_coeffs[final_lay_idx]
            lay_idx = final_lay_idx
        else:
            # There is none. Just identify the shape from the additional coeffs
            add_coeff = next(iter(additional_coeffs.values()))
            batch_size = add_coeff.shape[0]
            device = lower_bounds[-1].device

            lay_idx = final_lay_idx -1
            while lay_idx not in additional_coeffs:
                lay_shape = lower_bounds[lay_idx].shape
                rhos.append(torch.zeros((batch_size,) + lay_shape,
                                        device=device))
                lay_idx -= 1
            # We now reached the time where lay_idx has an additional coefficient
            rho = -additional_coeffs[lay_idx]
            rhos.append(torch.zeros_like(rho))
        lay_idx -= 1

        while lay_idx > 0:
            lay = weights[lay_idx]
            lbda = lay.backward(rho)

            lbs = lower_bounds[lay_idx]
            ubs = upper_bounds[lay_idx]

            scale = ubs / (ubs - lbs)
            scale.masked_fill_(lbs > 0, 1)
            scale.masked_fill_(ubs < 0, 0)

            rho = scale * lbda
            rhos.append(rho)
            lay_idx -= 1

        rhos.reverse()

        return DualVarSet(rhos)

    @staticmethod
    def get_initial_naive_solution(weights, additional_coeffs,
                                   lower_bounds, upper_bounds):
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[0]
        device = lower_bounds[-1].device

        rhos = []
        for lay_idx in range(1, len(weights)):
            lay_shape = lower_bounds[lay_idx].shape
            rhos.append(torch.zeros((batch_size,) + lay_shape,
                                    device=device))

        return DualVarSet(rhos)


class PrimalVarSet:
    def __init__(self, zahats, zbhats, z0):
        self.zahats = zahats
        self.zbhats = zbhats
        self.z0 = z0

    def as_dual_subgradient(self):
        rho_eq = []
        for zahat, zbhat in zip(self.zahats, self.zbhats):
            rho_eq.append(zbhat - zahat)
        return DualVarSet(rho_eq)

    def get_layer_subgradient(self, lay_idx):
        """
        Returns the subgradient for layer lay_idx (as a tensor of shape batch_size x layer width)
        """
        return self.zbhats[lay_idx] - self.zahats[lay_idx]

    def weighted_combination(self, other, coeff):
        new_zahats = []
        new_zbhats = []

        # Need to fix how many dim we expand depending on network size
        coeffs = []
        for zahat in self.zahats:
            nb_coeff_expands = (zahat.dim() - 1)
            coeffs.append(coeff.view((coeff.shape[0],) + (1,)*nb_coeff_expands))

        for zahat, ozahat, coeffd in zip(self.zahats, other.zahats, coeffs):
            new_zahats.append(zahat + coeffd * (ozahat - zahat))
        for zbhat, ozbhat, coeffd in zip(self.zbhats, other.zbhats, coeffs):
            new_zbhats.append(zbhat + coeffd * (ozbhat - zbhat))

        coeff0 = coeff.view((coeff.shape[0],) + (1,) * (self.z0.dim() - 1))
        new_z0 = self.z0 + coeff0 * (other.z0 - self.z0)

        return PrimalVarSet(new_zahats, new_zbhats, new_z0)

    def assert_subproblems_feasible(self, weights, final_coeffs,
                                    lower_bounds, upper_bounds):
        # TODO: implement this
        pass

    def weighted_combination_subproblem(self, subproblem, coeff):
        # Perform a weighted combination on the zahats and zbhats that correspond to subproblem k.

        k = subproblem.k
        o_zahat = subproblem.zahat_kp1
        o_zbhat = subproblem.zbhat_k

        coeffd_a = coeff.view((coeff.shape[0],) + (1,) * (self.zahats[k].dim() - 1))
        self.zahats[k] = self.zahats[k] + coeffd_a * (o_zahat - self.zahats[k])

        if k > 0:
            coeffd_b = coeff.view((coeff.shape[0],) + (1,) * (self.zbhats[k - 1].dim() - 1))
            self.zbhats[k-1] = self.zbhats[k-1] + coeffd_b * (o_zbhat - self.zbhats[k-1])
        else:
            coeff0 = coeff.view((coeff.shape[0],) + (1,) * (self.z0.dim() - 1))
            self.z0 = self.z0 + coeff0 * (subproblem.z0 - self.z0)
        return self


class DualVarSet:
    def __init__(self, rhos):
        self.rhos = rhos
        self.momentum = [torch.zeros_like(rho) for rho in rhos]
        self.temp_momentum = [torch.zeros_like(rho) for rho in rhos]
        self.primal_grad = [torch.zeros_like(rho) for rho in rhos]

    def bdot(self, other):
        val = 0
        for rho, orho in zip(self.rhos, other.rhos):
            val += bdot(rho, orho)
        return val

    def add_(self, step_size, to_add):
        for rho, rho_step in zip(self.rhos, to_add.rhos):
            rho.add_(step_size, rho_step)
        return self

    def add_cte_(self, cte):
        for rho in self.rhos:
            rho.add_(cte)
        return self

    def addcmul_(self, coeff, to_add1, to_add2):
        for rho, rho1, rho2 in zip(self.rhos, to_add1.rhos, to_add2.rhos):
            rho.addcmul_(coeff, rho1, rho2)
        return self

    def addcdiv_(self, coeff, num, denom):
        for rho, num_rho, denom_rho in zip(self.rhos, num.rhos, denom.rhos):
            rho.addcdiv_(coeff, num_rho, denom_rho)
        return self

    def div_cte_(self, denom):
        for rho in self.rhos:
            rho.div_(denom)
        return self

    def mul_(self, coeff):
        for rho in self.rhos:
            rho.mul_(coeff)
        return self

    def zero_like(self):
        new_rhos = []
        for rho in self.rhos:
            new_rhos.append(torch.zeros_like(rho))
        return DualVarSet(new_rhos)

    def add(self, to_add, step_size):
        new_rhos = []
        for rho, rho_step in zip(self.rhos, to_add.rhos):
            new_rhos.append(rho + step_size * rho_step)
        return DualVarSet(new_rhos)

    def subtract(self, step_size, to_subtract):
        new_rhos = []
        for rho, rho_step in zip(self.rhos, to_subtract.rhos):
            new_rhos.append(rho - step_size * rho_step)
        return DualVarSet(new_rhos)

    def sqrt(self):
        new_rhos = [rho.sqrt() for rho in self.rhos]
        return DualVarSet(new_rhos)

    def clone(self):
        new_rhos = [r.clone() for r in self.rhos]
        return DualVarSet(new_rhos)

    def weighted_squared_norm(self, eta):
        val = 0
        batch_size = self.rhos[0].shape[0]
        for rho in self.rhos:
            val += eta * rho.view(batch_size, -1).pow(2).sum(dim=-1)
        return val

    def assert_zero(self):
        for rho in self.rhos:
            assert rho.abs().max() == 0

    def update_from_anchor_points(self, anchor_point, primal_vars, eta, lay_idx="all", acceleration_dict=None):
        """
        Given the anchor point (DualVarSet instance) and primal vars (as PrimalVarSet instance), compute and return the
        updated dual variables (anchor points) with their
        closed-form from KKT conditions. The update is performed in place.
         lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = list(range(min(len(primal_vars.zahats), len(primal_vars.zbhats))))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            if acceleration_dict:
                if acceleration_dict['momentum'] and acceleration_dict['momentum'] != 0:
                    # use momentum
                    c_momentum = acceleration_dict['momentum'] * self.momentum[lay_idx] + (1 / eta) * primal_vars.get_layer_subgradient(lay_idx)
                    self.temp_momentum[lay_idx] = c_momentum  # keep track of the last momentum state
                    self.rhos[lay_idx] = anchor_point.rhos[lay_idx] + c_momentum
                    self.primal_grad[lay_idx] = self.rhos[lay_idx] - acceleration_dict['momentum'] * self.momentum[lay_idx]
                elif acceleration_dict['momentum'] == 0:
                    # normal proximal update
                    self.rhos[lay_idx] = anchor_point.rhos[lay_idx] + (1 / eta) * primal_vars.get_layer_subgradient(
                        lay_idx)
                    self.primal_grad[lay_idx] = self.rhos[lay_idx]
            else:
                # normal proximal update
                self.rhos[lay_idx] = anchor_point.rhos[lay_idx] + (1/eta) * primal_vars.get_layer_subgradient(lay_idx)
                self.primal_grad[lay_idx] = self.rhos[lay_idx]

    def update_acceleration(self, acceleration_dict=None):
        # update momentum to its last stored temporary version used for primal gradients. To be done before updating the
        # proximal terms. Does the same with nesterov.
        if acceleration_dict:
            if acceleration_dict['momentum'] != 0:
                self.momentum = [temp_mom.clone() for temp_mom in self.temp_momentum]

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return DualVarSet(copy.deepcopy(self.rhos))


class SubproblemCondGrad:
    # Contains the variables corresponding to a single subproblem conditional gradient computation
    def __init__(self, k, zahat_kp1, zbhat_k, z0=None):
        self.k = k
        self.zahat_kp1 = zahat_kp1
        self.zbhat_k = zbhat_k
        self.z0 = z0  # non-None only for the first layer

    def proximal_optimal_step_size_subproblem(self, additional_coeffs, dual_vars, primal_vars, n_layers, eta):
        # Compute proximal_optimal_step_size knowing that only the conditional gradient of subproblem k was updated.

        k = self.k
        zahat = self.zahat_kp1
        zbhat = self.zbhat_k

        a_diff = zahat - primal_vars.zahats[k]

        if k == 0:
            upper = bdot(dual_vars.primal_grad[0], a_diff)
            lower = (1/eta) * a_diff.view(a_diff.shape[0], -1).pow(2).sum(dim=-1)
        else:
            b_diff = primal_vars.zbhats[k - 1] - zbhat
            upper = bdot(dual_vars.primal_grad[k-1], b_diff)
            lower = (1/eta) * b_diff.view(b_diff.shape[0], -1).pow(2).sum(dim=-1)

            if k != (n_layers-1):
                upper += bdot(dual_vars.primal_grad[k], a_diff)
                lower += (1/eta) * a_diff.view(a_diff.shape[0], -1).pow(2).sum(dim=-1)
            if (k+1) in additional_coeffs:
                upper += bdot(additional_coeffs[k+1], primal_vars.zahats[k] - zahat)

        opt_step_size = torch.where(lower > 0, upper / lower, torch.zeros_like(lower))
        # Set to 0 the 0/0 entries.
        up_mask = upper == 0
        low_mask = lower == 0
        sum_mask = up_mask + low_mask
        opt_step_size[sum_mask > 1] = 0
        opt_step_size = torch.clamp(opt_step_size, min=0, max=1)

        decrease = -0.5 * lower * opt_step_size.pow(2) + upper * opt_step_size

        return opt_step_size, decrease


def opt_lin_domain(layer, lin_k, lin_hat_kp1, lb_k, ub_k):
    lin_eq = lin_k + layer.backward(lin_hat_kp1)

    pos_coeff = (lin_eq > 0)
    za_k = torch.where(pos_coeff, lb_k, ub_k)
    zahat_kp1 = layer.forward(za_k)

    return za_k, zahat_kp1


def opt_relu_pair(layer, lin_hat_k, lin_hat_kp1, lb_k, ub_k):
    if lin_hat_kp1 is 0:
        kp1_k_eq = torch.zeros_like(lin_hat_k)
    else:
        kp1_k_eq = layer.backward(lin_hat_kp1)

    # Passing ReLUs
    lin_eq = lin_hat_k + kp1_k_eq
    passing_zhatk = torch.where(lin_eq > 0, lb_k, ub_k)

    # Blocking ReLUs
    blocking_zhatk = torch.where(lin_hat_k > 0, lb_k, ub_k)

    # Ambiguous ReLUs
    pos_kp1keq = torch.clamp(kp1_k_eq, 0, None)
    neg_kp1keq = torch.clamp(kp1_k_eq, None, 0)
    s_k = ub_k / (ub_k - lb_k)

    amb_eq_val = torch.stack([
        lin_hat_k*lb_k + neg_kp1keq * s_k * lb_k,
        lin_hat_k*ub_k + neg_kp1keq * s_k * ub_k + pos_kp1keq * ub_k,
        torch.zeros_like(neg_kp1keq)
    ])
    _, choice = torch.min(amb_eq_val, 0)
    amb_zhatk = torch.where(choice==0, lb_k, ub_k)
    amb_zhatk.masked_fill_(choice==2, 0)

    zhat_k = torch.where(lb_k >= 0,
                         passing_zhatk,
                         torch.where(ub_k <= 0,
                                     blocking_zhatk,
                                     amb_zhatk))

    relu_zhat_k = torch.clamp(zhat_k, 0, None)
    non_amb_zk = relu_zhat_k
    amb_zk = torch.where(kp1_k_eq >= 0, relu_zhat_k, s_k*(zhat_k-lb_k))
    z_k = torch.where((lb_k >= 0) | (ub_k <= 0),
                      non_amb_zk,
                      amb_zk)

    zhat_kp1 = layer.forward(z_k)

    return zhat_k, zhat_kp1


class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """

    def __init__(self, rhos):
        """
        Given rhos to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1 = []
        self.temp_m1 = []
        # second moments
        self.m2 = []
        self.temp_m2 = []
        for rho in rhos:
            self.m1.append(torch.zeros_like(rho))
            self.temp_m1.append(torch.zeros_like(rho))
            self.m2.append(torch.zeros_like(rho))
            self.temp_m2.append(torch.zeros_like(rho))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.outer_it = 1

    def update_moments_take_step(self, lay_idx, eta, dual_vars, anchor_vars, primal_vars):
        """
        Update the ADAM moments given the subgradients, and normal gd step size, then take the step from dual_vars.
        Update performed in place on dual_vars.
        """
        # Update the ADAM moments.
        self.temp_m1[lay_idx] = self.m1[lay_idx].mul(self.beta1).add(
            1-self.beta1,
            primal_vars.get_layer_subgradient(lay_idx))
        self.temp_m2[lay_idx] = self.m2[lay_idx].mul(self.beta2).addcmul(
            1 - self.beta2, primal_vars.get_layer_subgradient(lay_idx),
            primal_vars.get_layer_subgradient(lay_idx))

        bias_correc1 = 1 - self.beta1 ** (self.outer_it)
        bias_correc2 = 1 - self.beta2 ** (self.outer_it)
        corrected_step_size = (1/eta) * math.sqrt(bias_correc2) / bias_correc1

        # Take the projected (non-negativity constraints) step.
        rho_step = self.temp_m1[lay_idx] / (self.temp_m2[lay_idx].sqrt() + self.epsilon)
        dual_vars.rhos[lay_idx] = anchor_vars.rhos[lay_idx] + corrected_step_size * rho_step

    def update_adam_stats(self):
        """
        Update the internal ADAM stats to the last temporary ones (called before updating the proximal terms)
        """
        self.m1 = [c_temp_m1.clone() for c_temp_m1 in self.temp_m1]
        self.m2 = [c_temp_m2.clone() for c_temp_m2 in self.temp_m2]
        self.outer_it += 1
