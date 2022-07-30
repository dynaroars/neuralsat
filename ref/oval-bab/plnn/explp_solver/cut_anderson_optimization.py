import torch
from plnn.proxlp_solver import utils
import math
import copy
from plnn.explp_solver import anderson_optimization


class CutDualVars(anderson_optimization.DualVars):
    """
    Class representing the dual variables alpha, beta_0, and beta_1, and their functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for beta_0, for indices 0 to n for
    the others.
    """
    def __init__(self, alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, beta_list=None, I_list=[]):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        alpha_back and beta_1_back are lists of the backward passes of alpha and beta_1. Useful to avoid
        re-computing them.
        """
        super().__init__(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs)
        self.beta_list = beta_list
        self.I_list = I_list

    @staticmethod
    def from_super_class(super_instance, beta_list=None, I_list=None):
        """
        Return an instance of this class from an instance of the super class.
        """
        return CutDualVars(super_instance.alpha, super_instance.sum_beta, super_instance.sum_Wp1Ibetap1,
                           super_instance.sum_W1mIubeta, super_instance.sum_WIlbeta, super_instance.fs,
                           super_instance.gs, beta_list=beta_list, I_list=I_list)

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        base_duals = anderson_optimization.DualVars.naive_initialization(weights, additional_coeffs, device, input_size)

        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        beta_a = []  # Indexed from 0 to n-1, the first is always zero
        beta_list = []
        I_list=[]
        # Build also the shortcut terms f and g

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device, dtype=add_coeff.dtype)
        # Insert the dual variables for the box bound
        fixed_0_inpsize = zero_tensor(input_size)
        beta_a.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(beta_a[-1])[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            beta_a.append(zero_tensor(nb_outputs))
            I_list.append([])
            beta_list.append([])

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])
        beta_a.append(torch.zeros_like(alpha[-1]))

        return CutDualVars.from_super_class(base_duals, beta_list=beta_list, I_list=I_list)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, additional_coeffs, device, input_size, clbs, cubs, lower_bounds,
                            upper_bounds, opt_args):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual variables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, xt, zt, beta_list, I_list = \
            bigm_duals.as_cut_initialization(weights, clbs, cubs, lower_bounds, upper_bounds)

        base_duals, primals = CutDualVars(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, beta_list=beta_list, I_list=I_list), \
                              CutPrimalVars(xt, zt)

        return base_duals, primals

    def as_saddlep_initialization(self):
        """
        Given the network layers and pre-activation bounds as lists of tensors,
        compute and return the corresponding initialization of the explp (Anderson) variables from the instance of this
        class.
        """
        return self.alpha, self.sum_beta, self.sum_Wp1Ibetap1, self.sum_W1mIubeta, self.sum_WIlbeta, self.fs, self.gs

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return CutDualVars(
            copy.deepcopy(self.alpha), copy.deepcopy(self.sum_beta), copy.deepcopy(self.sum_Wp1Ibetap1),
            copy.deepcopy(self.sum_W1mIubeta), copy.deepcopy(self.sum_WIlbeta), copy.deepcopy(self.fs),
            copy.deepcopy(self.gs), beta_list=copy.deepcopy(self.beta_list), I_list=copy.deepcopy(self.I_list))

    def dualbigmk_grad(self, lay_idx, weights, primal_vars, l_preacts, u_preacts):
        """
        Given list of layers, preactivation bounds, çurrent primal variables (instance of CutrimalVars),
        compute and return the gradient of bigm dual variables related to layer k.
        """
        xk_hat = weights[lay_idx - 1].forward(primal_vars.xt[lay_idx-1])
        xk = primal_vars.xt[lay_idx]
        zk = primal_vars.zt[lay_idx - 1]
        alpha_subgrad = xk_hat - xk
        beta_0_subg = xk - zk * u_preacts[lay_idx].unsqueeze(1)
        beta_1_subg = xk + (1 - zk) * l_preacts[lay_idx].unsqueeze(1) - xk_hat

        return alpha_subgrad, beta_0_subg, beta_1_subg

    def update_from_step(self, weights, dual_vars_subg, lay_idx="all"):
        """
        Given the network pre-activation bounds as lists of tensors, all dual variables (and their functions f and g)
        lay_idx are the layers (int or list) for which to perform the update. "all" means update all
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.fs))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)
        for lay_idx in lay_to_iter:
            if lay_idx > 0:
                self.update_duals_from_alphak(lay_idx, weights, dual_vars_subg.alpha[lay_idx])
                self.update_duals_from_betak(
                    lay_idx, weights, dual_vars_subg.sum_beta[lay_idx], dual_vars_subg.sum_Wp1Ibetap1[lay_idx-1],
                    dual_vars_subg.sum_W1mIubeta[lay_idx], dual_vars_subg.sum_WIlbeta[lay_idx])


class DualADAMStats:
    """
    class storing (and containing operations for) the ADAM statistics for the dual variables.
    they are stored as lists of tensors, for ReLU indices from 1 to n-1.
    """
    def __init__(self, sum_beta, beta1=0.9, beta2=0.999):
        """
        Given beta_0 to copy the dimensionality from, initialize all ADAM stats to 0 tensors.
        """
        # first moments
        self.m1_alpha = []
        self.m1_sum_beta = []
        # second moments
        self.m2_alpha = []
        self.m2_sum_beta = []
        for lay_idx in range(1, len(sum_beta)):
            self.m1_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m2_alpha.append(torch.zeros_like(sum_beta[lay_idx]))
            self.m1_sum_beta.append([torch.zeros_like(sum_beta[lay_idx]), torch.zeros_like(sum_beta[lay_idx])])
            self.m2_sum_beta.append([torch.zeros_like(sum_beta[lay_idx]), torch.zeros_like(sum_beta[lay_idx])])

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def bigm_adam_initialization(self, sum_beta, bigm_adam_stats, beta1=0.9, beta2=0.999):
        # first moments
        self.m1_alpha = []
        self.m1_sum_beta = []
        # second moments
        self.m2_alpha = []
        self.m2_sum_beta = []
        for lay_idx in range(1, len(sum_beta)):
            self.m1_alpha.append(bigm_adam_stats.m1_alpha[lay_idx-1])
            self.m2_alpha.append(bigm_adam_stats.m2_alpha[lay_idx-1])
            self.m1_sum_beta.append([bigm_adam_stats.m1_beta_0[lay_idx - 1], bigm_adam_stats.m1_beta_1[lay_idx - 1]])
            self.m2_sum_beta.append([bigm_adam_stats.m2_beta_0[lay_idx - 1], bigm_adam_stats.m2_beta_1[lay_idx - 1]])

        self.coeff1 = beta1
        self.coeff2 = beta2
        self.epsilon = 1e-8

    def active_set_adam_step(self, weights, masked_ops, step_size, outer_it, dual_vars, primal_vars, clbs, cubs, nubs,
                             l_preacts, u_preacts, l_checks, u_checks, opt_args, precision=torch.float):
        """
        Take a projected ADAM step on the active set of dual variables, possibly adding new variables to the active set.
        Update performed in place on dual_vars.
        """
        new_alpha = []
        new_sum_beta = []
        new_sum_WkIbeta = []
        new_sum_Wk1mIubeta = []
        new_sum_WkIlbeta = []

        new_alpha.append(torch.zeros_like(dual_vars.alpha[0]))
        new_sum_beta.append(torch.zeros_like(dual_vars.sum_beta[0]))
        new_sum_Wk1mIubeta.append(torch.zeros_like(dual_vars.sum_W1mIubeta[0]))
        new_sum_WkIlbeta.append(torch.zeros_like(dual_vars.sum_WIlbeta[0]))

        # Inner minimization (primals) at layer 0.
        x0, _ = CutPrimalVars.primalsk_min(0, dual_vars, clbs, cubs, precision)
        primal_vars.update_primals_from_primalsk(0, x0, None)

        for lay_idx in range(1, len(dual_vars.sum_beta)):

            # Inner minimization (primals) at layer k.
            new_xk, new_zk = CutPrimalVars.primalsk_min(lay_idx, dual_vars, clbs, cubs, precision)
            primal_vars.update_primals_from_primalsk(lay_idx, new_xk, new_zk)

            # Compute big-M supergradients.
            alphak_subgrad, beta0k_subg, beta1k_subg = dual_vars.dualbigmk_grad(
                lay_idx, weights, primal_vars, l_preacts, u_preacts)

            # Update the ADAM moments.
            self.m1_alpha[lay_idx-1].mul_(self.coeff1).add_(1-self.coeff1, alphak_subgrad)
            self.m2_alpha[lay_idx-1].mul_(self.coeff2).addcmul_(1 - self.coeff2, alphak_subgrad, alphak_subgrad)

            bias_correc1 = 1 - self.coeff1 ** (outer_it + 1)
            bias_correc2 = 1 - self.coeff2 ** (outer_it + 1)
            corrected_step_size = step_size * math.sqrt(bias_correc2) / bias_correc1

            # Take the projected (non-negativity constraints) step.
            alpha_step_size = self.m1_alpha[lay_idx-1] / (self.m2_alpha[lay_idx-1].sqrt() + self.epsilon)
            new_alpha_k = torch.clamp(dual_vars.alpha[lay_idx] + corrected_step_size * alpha_step_size, 0, None)
            new_alpha.append(new_alpha_k)

            # Easily access this layer's operands.
            lin_k = weights[lay_idx - 1]
            masked_op = masked_ops[lay_idx - 1]
            zk = primal_vars.zt[lay_idx-1]
            xk = primal_vars.xt[lay_idx]
            xkm1 = primal_vars.xt[lay_idx-1]
            l_preact = l_preacts[lay_idx].unsqueeze(1)
            u_preact = u_preacts[lay_idx].unsqueeze(1)
            l_check = l_checks[lay_idx-1]
            u_check = u_checks[lay_idx-1]

            if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                # Unfold the convolutional inputs into matrices containing the parts (slices) of the input forming the
                # convolution output.
                unfolded_xkm1 = lin_k.unfold_input(xkm1, gather=dual_vars.and_set[lay_idx - 1])  # input space unfolding

            new_sum_betak = torch.zeros_like(dual_vars.sum_beta[lay_idx])
            new_sum_WkIbetak = torch.zeros_like(dual_vars.sum_Wp1Ibetap1[lay_idx - 1])
            new_sum_Wk1mIubetak = torch.zeros_like(dual_vars.sum_W1mIubeta[lay_idx])
            new_sum_WkIlbetak = torch.zeros_like(dual_vars.sum_WIlbeta[lay_idx])

            Ik_list = dual_vars.I_list[lay_idx-1]
            for ik_index in range(len(Ik_list)):
                if ik_index == 0:
                    # Update beta_0 moments and take projected step.
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, beta0k_subg)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(
                        1 - self.coeff2, beta0k_subg, beta0k_subg)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / \
                                         (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size *
                                                   sum_beta_step_size, 0, None).type(precision)
                    beta_WIl = 0
                    beta_W1mIu = (u_preact - lin_k.get_bias()) * new_sum_betak_ik
                    beta_WI = 0
                elif ik_index == 1:
                    # Update beta_1 moments and take projected step.
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, beta1k_subg)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(
                        1 - self.coeff2, beta1k_subg, beta1k_subg)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / \
                                         (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size *
                                                   sum_beta_step_size, 0, None).type(precision)
                    beta_WIl = (l_preact - lin_k.get_bias()) * new_sum_betak_ik
                    beta_W1mIu = 0
                    beta_WI = lin_k.backward(new_sum_betak_ik)
                else:
                    # Update beta_I_k moments and take projected step (for variables in the active set).
                    masked_op.set_mask(Ik_list[ik_index])
                    WI_xkm1 = masked_op.forward(unfolded_xkm1 if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]
                                                else xkm1, add_bias=False)
                    nub_WIu = nubs[lay_idx - 1].unsqueeze(1) - masked_op.forward(
                        u_check, no_unsqueeze=True, add_bias=False)
                    W1mIu = nub_WIu - lin_k.get_bias()
                    WIl = masked_op.forward(l_check, no_unsqueeze=True, add_bias=False)
                    exp_k_grad = xk - WI_xkm1 + (1 - zk) * WIl - zk * nub_WIu
                    self.m1_sum_beta[lay_idx-1][ik_index].mul_(self.coeff1).add_(1-self.coeff1, exp_k_grad)
                    self.m2_sum_beta[lay_idx-1][ik_index].mul_(self.coeff2).addcmul_(
                        1 - self.coeff2, exp_k_grad, exp_k_grad)
                    sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index] / \
                                         (self.m2_sum_beta[lay_idx-1][ik_index].sqrt() + self.epsilon)
                    new_sum_betak_ik = torch.clamp(dual_vars.beta_list[lay_idx-1][ik_index] + corrected_step_size *
                                                   sum_beta_step_size, 0, None).type(precision)

                    # TODO: adapt to linears as well
                    if dual_vars.and_set[lay_idx - 1] is not None and type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                        # Zero parts of the input space we are not using the Anderson relaxation on.
                        new_sum_betak_ik = \
                            lin_k.zero_scatter_folded_out(new_sum_betak_ik, dual_vars.and_set[lay_idx - 1])

                    beta_WIl = WIl * new_sum_betak_ik
                    beta_W1mIu = W1mIu * new_sum_betak_ik
                    beta_WI = masked_op.backward(new_sum_betak_ik)

                dual_vars.beta_list[lay_idx-1][ik_index] = new_sum_betak_ik
                # update pre-computed passes.
                new_sum_betak = new_sum_betak + new_sum_betak_ik
                new_sum_WkIbetak = new_sum_WkIbetak + beta_WI
                new_sum_Wk1mIubetak = new_sum_Wk1mIubetak + beta_W1mIu
                new_sum_WkIlbetak = new_sum_WkIlbetak + beta_WIl

            # Add new variables to the active set, and take the corresponding projected ADAM step for them.
            if (len(Ik_list) <= opt_args['max_cuts'] and outer_it % opt_args['cut_frequency'] < opt_args['cut_add']):

                # Compute most violated constraint at current primal minimizer.
                masked_op, Istar_k, exp_k_grad, WIl, W1mIu, _ = dual_vars.anderson_oracle(
                    lay_idx, weights, masked_ops, nubs, l_checks, u_checks, primal_vars,
                    random_mask=opt_args['random_cuts'])

                self.m1_sum_beta[lay_idx-1].append(torch.zeros_like(dual_vars.sum_beta[lay_idx]))
                self.m2_sum_beta[lay_idx-1].append(torch.zeros_like(dual_vars.sum_beta[lay_idx]))
                self.m1_sum_beta[lay_idx-1][ik_index+1].mul_(self.coeff1).add_(1-self.coeff1, exp_k_grad)
                self.m2_sum_beta[lay_idx-1][ik_index+1].mul_(self.coeff2).addcmul_(
                    1 - self.coeff2, exp_k_grad, exp_k_grad)

                sum_beta_step_size = self.m1_sum_beta[lay_idx-1][ik_index+1] / \
                                     (self.m2_sum_beta[lay_idx-1][ik_index+1].sqrt() + self.epsilon)
                M_atom_mask = torch.clamp(corrected_step_size * sum_beta_step_size, 0, None).type(precision)

                # TODO: adapt to linears as well
                if dual_vars.and_set[lay_idx - 1] is not None and type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                    # Zero parts of the input space we are not using the Anderson relaxation on.
                    M_atom_mask = lin_k.zero_scatter_folded_out(M_atom_mask, dual_vars.and_set[lay_idx - 1])

                dual_vars.beta_list[lay_idx-1].append(M_atom_mask)
                dual_vars.I_list[lay_idx-1].append(Istar_k)

                beta_WI_lmo = masked_op.backward(M_atom_mask)
                beta_WIl_lmo = WIl * M_atom_mask
                beta_W1mIu_lmo = W1mIu * M_atom_mask
                # update pre-computed backward passes.
                new_sum_betak = new_sum_betak + M_atom_mask
                new_sum_WkIbetak = new_sum_WkIbetak + beta_WI_lmo
                new_sum_Wk1mIubetak = new_sum_Wk1mIubetak + beta_W1mIu_lmo
                new_sum_WkIlbetak = new_sum_WkIlbetak + beta_WIl_lmo

            new_sum_beta.append(new_sum_betak)
            new_sum_WkIbeta.append(new_sum_WkIbetak)
            new_sum_Wk1mIubeta.append(new_sum_Wk1mIubetak)
            new_sum_WkIlbeta.append(new_sum_WkIlbetak)

        return CutDualVars(new_alpha, new_sum_beta, new_sum_WkIbeta, new_sum_Wk1mIubeta, new_sum_WkIlbeta, None, None,
                           beta_list=None, I_list=[])


class CutPrimalVars(anderson_optimization.PrimalVars):

    def __init__(self, xt, zt):
        """
        Given the primal vars as lists of tensors (of correct length), initialize the class with these.
        """
        self.xt = xt
        self.zt = zt

    @staticmethod
    def from_super_class(super_instance):
        """
        Return an instance of this class from an instance of the super class.
        """
        return CutPrimalVars(super_instance.xt, super_instance.zt)

    @staticmethod
    def mid_box_initialization(dual_vars, clbs, cubs):
        """
        Initialize the primal variables (anchor points) to the mid-point of the box constraints (halfway through each
        variable's lower and upper bounds).
        """
        primals = anderson_optimization.PrimalVars.mid_box_initialization(dual_vars, clbs, cubs)
        return CutPrimalVars.from_super_class(primals)

    def copy(self):
        """
        deep-copy the current instance
        :return: the copied class instance
        """
        return CutPrimalVars(copy.deepcopy(self.xt), copy.deepcopy(self.zt))

    @staticmethod
    def primalsk_min(lay_idx, dual_vars, clbs, cubs, precision):
        """
        Given list of post-activation bounds and çurrent dual variables (instance of SaddleDualVars),
        compute and return the linear minimization oracle of x_k and z_k with their gradients, and the gradients
        themselves. (done jointly as they are independent, so no Gauss-Seidel effect)
        """
        x_k_lmo = torch.where(dual_vars.fs[lay_idx] >= 0, cubs[lay_idx].unsqueeze(1), clbs[lay_idx].unsqueeze(1))
        if lay_idx > 0:
            z_k_lmo = (dual_vars.gs[lay_idx - 1] >= 0).type(precision)
        else:
            # g_k is defined from 1 to n - 1.
            z_k_lmo = None
        return x_k_lmo, z_k_lmo

    def update_primals_from_primalsk(self, lay_idx, new_xk, new_zk):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        self.xt[lay_idx] = new_xk
        if lay_idx > 0:
            self.zt[lay_idx-1] = new_zk
