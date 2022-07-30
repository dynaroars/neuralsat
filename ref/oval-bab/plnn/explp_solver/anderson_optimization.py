import itertools
import torch
from plnn.proxlp_solver import utils
from plnn.branch_and_bound.utils import ParentInit


def compute_bounds(dual_vars, weights, clbs, cubs, new_fs=None, new_gs=None):
    """
    Compute the problem bounds, given the dual variables (instance of DualVars), their sufficient statistics,
    intermediate bounds (clbs, cubs) (as lists of tensors) and network layers (weights, LinearOp, ConvOp classes from
    proxlp_solver.utils).
    Dual variables are tensors of size opt_layer_width x layer shape, the intermediate bounds lack opt_layer_width.
    :param new_fs: as new_gs, allows for fs and gs functions to be decoupled from the passed dual variables.
    :return: a tensor of bounds, of size 2 x n_neurons of the layer to optimize. The first half is the negative of the
    upper bound of each neuron, the second the lower bound.
    """

    fs = dual_vars.fs if new_fs is None else new_fs
    gs = dual_vars.gs if new_gs is None else new_gs

    bounds = 0
    for lin_k, alpha_k in zip(weights, dual_vars.alpha[1:]):
        b_k = lin_k.get_bias()
        bounds += utils.bdot(alpha_k, b_k)

    for f_k, cl_k, cu_k in zip(fs, clbs, cubs):
        bounds -= utils.bdot(torch.clamp(f_k, 0, None), cu_k.unsqueeze(1))
        bounds -= utils.bdot(torch.clamp(f_k, None, 0), cl_k.unsqueeze(1))

    for g_k in gs:
        bounds -= torch.clamp(g_k, 0, None).sum(dim=tuple(range(2, g_k.dim())))  # z to 1

    for sum_WIlbeta_k in dual_vars.sum_WIlbeta:
        bounds += sum_WIlbeta_k.sum(dim=tuple(range(2, sum_WIlbeta_k.dim())))

    return bounds


def precompute_l_u_check(weights, dual_vars, primal_vars, clbs, cubs):
    # Given duals, primals, weights, and postact boudns, precompute l_check, u_check, and instances of MaskedOp classes.
    masked_ops, u_checks, l_checks = [], [], []
    for lay_idx in range(1, len(dual_vars.sum_beta)):
        lin_k = weights[lay_idx - 1]
        cl_km1 = clbs[lay_idx - 1]
        cu_km1 = cubs[lay_idx - 1]
        if type(weights[lay_idx - 1]) not in [utils.ConvOp, utils.BatchConvOp]:
            masked_op = MaskedLinearOp(lin_k)
            cu_km1 = utils.apply_transforms(lin_k.pre_transform, cu_km1)
            cl_km1 = utils.apply_transforms(lin_k.pre_transform, cl_km1)
            u_checks.append(torch.where(lin_k.weights > 0, cu_km1.unsqueeze(1), cl_km1.unsqueeze(1)).unsqueeze(1))
            l_checks.append(torch.where(lin_k.weights > 0, cl_km1.unsqueeze(1), cu_km1.unsqueeze(1)).unsqueeze(1))
            masked_ops.append(masked_op)
        else:
            masked_op = MaskedConvOp(
                lin_k, primal_vars.xt[lay_idx - 1], dual_vars.sum_beta[lay_idx],
                gather=dual_vars.and_set[lay_idx - 1])
            unfolded_cu_km1 = lin_k.unfold_input(cu_km1.unsqueeze(1), gather=dual_vars.and_set[lay_idx - 1]).unsqueeze(2)
            unfolded_cl_km1 = lin_k.unfold_input(cl_km1.unsqueeze(1), gather=dual_vars.and_set[lay_idx - 1]).unsqueeze(2)
            u_checks.append(torch.where(
                (masked_op.unfolded_W_k > 0).unsqueeze(-1), unfolded_cu_km1, unfolded_cl_km1))
            l_checks.append(torch.where(
                (masked_op.unfolded_W_k > 0).unsqueeze(-1), unfolded_cl_km1, unfolded_cu_km1))
            masked_ops.append(masked_op)
    return masked_ops, u_checks, l_checks


class DualVars:
    """
    Class representing the base dual (relaxation) variables for the Anderson relaxation. These are alpha_0, alpha_1,
    beta (through its sufficient statistics), and their  functions f and g.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for all variables except alpha_1.
    """
    def __init__(self, alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs):
        """
        Given the dual vars as lists of tensors (of correct length) along with their computed functions, initialize the
        class with these.
        """
        self.alpha = alpha
        self.sum_beta = sum_beta
        self.sum_Wp1Ibetap1 = sum_Wp1Ibetap1
        self.sum_W1mIubeta = sum_W1mIubeta
        self.sum_WIlbeta = sum_WIlbeta
        self.fs = fs
        self.gs = gs

        self.alpha_norm = [0] * len(sum_beta)
        self.beta_norm = [0] * len(sum_beta)
        self.update_norms()

        # TODO: handle this (set of neurons for which to use the Anderson constraints).
        self.and_set = [None] * len(sum_beta)

    @staticmethod
    def naive_initialization(weights, additional_coeffs, device, input_size):
        """
        Given parameters from the optimize function, initialize the dual vairables and their functions as all 0s except
        some special corner cases. This is equivalent to initialising with naive interval propagation bounds.
        """
        add_coeff = next(iter(additional_coeffs.values()))
        batch_size = add_coeff.shape[:2]

        alpha = []  # Indexed from 0 to n, the last is constrained to the cost function, first is zero
        sum_beta = []  # Indexed from 0 to n-1, first is always zero
        sum_W1mIubeta = []  # Indexed from 0 to n-1, first is always zero
        sum_WIlbeta = []  # Indexed from 0 to n-1, first is always zero
        sum_Wp1Ibetap1 = []  # Indexed from 0 to n-1, but represent the next beta

        # Build also the shortcut terms f and g
        fs = []  # Indexed from 0 to n-1
        gs = []  # Indexed from 1 to n-1

        # Fill in the variable holders with variables, all initiated to zero
        zero_tensor = lambda size: torch.zeros((*batch_size, *size), device=device)
        # Insert the dual variables for the box bound
        sum_Wp1Ibetap1.append(zero_tensor(input_size))
        fs.append(zero_tensor(input_size))
        fixed_0_inpsize = zero_tensor(input_size)
        sum_beta.append(fixed_0_inpsize)
        sum_W1mIubeta.append(fixed_0_inpsize)
        sum_WIlbeta.append(fixed_0_inpsize)
        alpha.append(fixed_0_inpsize)
        for lay_idx, layer in enumerate(weights[:-1]):
            nb_outputs = layer.get_output_shape(sum_beta[-1])[2:]

            # Initialize the dual variables
            alpha.append(zero_tensor(nb_outputs))
            sum_beta.append(zero_tensor(nb_outputs))
            sum_W1mIubeta.append(zero_tensor(nb_outputs))
            sum_WIlbeta.append(zero_tensor(nb_outputs))
            sum_Wp1Ibetap1.append(zero_tensor(nb_outputs))

            # Initialize the shortcut terms
            fs.append(zero_tensor(nb_outputs))
            gs.append(zero_tensor(nb_outputs))

        # Add the fixed values that can't be changed that comes from above
        alpha.append(additional_coeffs[len(weights)])

        # Adjust the fact that the last term for the f shorcut is not zero,
        # because it depends on alpha.
        fs[-1] = -weights[-1].backward(additional_coeffs[len(weights)])

        return DualVars(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs)

    @staticmethod
    def bigm_initialization(bigm_duals, weights, clbs, cubs, lower_bounds, upper_bounds):
        """
        Given bigm dual variables, network weights, post/pre-activation lower and upper bounds,
        initialize the Anderson dual vairables and their functions to the corresponding values of the bigm duals.
        Additionally, it returns the primal variables corresponding to the inner bigm minimization with those dual
        variables.
        """
        # alphas and betas are in the bigm relaxation and their values come from there
        alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs, xt, zt = \
            bigm_duals.as_explp_initialization(weights, clbs, cubs, lower_bounds, upper_bounds)

        return DualVars(alpha, sum_beta, sum_Wp1Ibetap1, sum_W1mIubeta, sum_WIlbeta, fs, gs), PrimalVars(xt, zt)

    def zero_dual_vars(self, weights, additional_coeffs):
        """
        Set all the dual variables to 0 (and treat their functions accordingly).
        """
        for tensor in itertools.chain(self.alpha[:-1], self.sum_beta, self.sum_W1mIubeta, self.sum_WIlbeta,
                                      self.sum_Wp1Ibetap1, self.fs, self.gs):
            tensor.zero_()
        self.fs[-1].copy_(-weights[-1].backward(additional_coeffs[len(weights)]))

    def update_norms(self, lay_idx="all"):
        """
        Given the current values of alphas and betas, update the internally stored norms alpha_norm and beta_norm.
        :return:
        """
        if lay_idx == "all":
            lay_to_iter = range(len(self.sum_beta))
        else:
            lay_to_iter = [lay_idx] if type(lay_idx) is int else list(lay_idx)

        for lay_idx in lay_to_iter:
            self.alpha_norm[lay_idx] = self.alpha[lay_idx].\
                sum(dim=tuple(range(2, self.alpha[lay_idx].dim()))).unsqueeze(-1)
            self.beta_norm[lay_idx] = self.sum_beta[lay_idx].\
                sum(dim=tuple(range(2, self.sum_beta[lay_idx].dim()))).unsqueeze(-1)

    def update_duals_from_alphak(self, lay_idx, weights, new_alpha_k):
        """
        Given new values for alphas at layer lay_idx, update the dual variables and their functions.
        """
        self.fs[lay_idx] += (new_alpha_k - self.alpha[lay_idx])
        self.fs[lay_idx - 1] -= weights[lay_idx - 1].backward(new_alpha_k - self.alpha[lay_idx])
        self.alpha[lay_idx] = new_alpha_k
        self.alpha_norm[lay_idx] = self.alpha[lay_idx].\
            sum(dim=tuple(range(2, self.alpha[lay_idx].dim()))).unsqueeze(-1)

    def update_duals_from_betak(self, lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak,
                               new_sum_WkIlbetak):
        """
        Given new values for beta sufficient statistics at layer lay_idx, update the dual variables and their functions.
        """
        self.fs[lay_idx - 1] += (new_sum_WkIbetak - self.sum_Wp1Ibetap1[lay_idx - 1])
        self.fs[lay_idx] -= (new_sum_betak - self.sum_beta[lay_idx])
        self.gs[lay_idx - 1] += ((new_sum_betak - self.sum_beta[lay_idx]) * weights[lay_idx - 1].get_bias()
                + (new_sum_Wk1mIubetak - self.sum_W1mIubeta[lay_idx])
                + (new_sum_WkIlbetak - self.sum_WIlbeta[lay_idx]))
        self.sum_beta[lay_idx] = new_sum_betak
        self.sum_W1mIubeta[lay_idx] = new_sum_Wk1mIubetak
        self.sum_WIlbeta[lay_idx] = new_sum_WkIlbetak
        self.sum_Wp1Ibetap1[lay_idx - 1] = new_sum_WkIbetak
        self.beta_norm[lay_idx] = self.sum_beta[lay_idx].\
            sum(dim=tuple(range(2, self.sum_beta[lay_idx].dim()))).unsqueeze(-1)

    def update_duals_from_alphak_betak(self, lay_idx, weights, new_alpha_k, new_sum_betak, new_sum_WkIbetak,
                                     new_sum_Wk1mIubetak, new_sum_WkIlbetak):
        """
        Given new values for alpha and beta sufficient statistics at layer lay_idx, update the dual variables and their
        functions.
        """
        self.update_duals_from_alphak(lay_idx, weights, new_alpha_k)
        self.update_duals_from_betak(lay_idx, weights, new_sum_betak, new_sum_WkIbetak, new_sum_Wk1mIubetak,
                                     new_sum_WkIlbetak)

    def anderson_oracle(self, lay_idx, weights, masked_ops, nubs, l_checks, u_checks, primal_vars,
                        do_intermediates=True, random_mask=False):
        """
        Given list of layers, primal variables (instance of SaddlePrimalVars), post activation bounds,
        compute and return the output of the Anderson oracle over the exponential family of beta variables.
        If random_mask is True, the mask is sampled by tossing a coin for each binary entry. (hence, the Anderson
         oracle is not used)
        Returns the optimal mask, along with the corresponding gradient and relevant intermediate computations.
        """
        lin_k = weights[lay_idx - 1]
        zk = primal_vars.zt[lay_idx - 1]
        xk = primal_vars.xt[lay_idx]
        xkm1 = primal_vars.xt[lay_idx - 1]
        masked_op = masked_ops[lay_idx - 1]
        l_check = l_checks[lay_idx - 1]
        u_check = u_checks[lay_idx - 1]

        # Compute the optimal weight mask from the exponential family...
        if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
            unfolded_xkm1 = lin_k.unfold_input(xkm1, gather=self.and_set[lay_idx - 1])  # input space unfolding
            unfolded_zk = lin_k.unfold_output(zk, gather=self.and_set[lay_idx - 1])  # output space unfolding
            if not random_mask:
                d_in = unfolded_zk.unsqueeze(-2) * (u_check - l_check) + l_check
                d_in -= unfolded_xkm1.unsqueeze(2)
            else:
                xkm1_shape = unfolded_xkm1.shape[-2:]
        else:
            # TODO: gather adaptation missing for linear layers
            unfolded_zk = None
            # Fully connected layer.
            xkm1 = utils.apply_transforms(lin_k.pre_transform, xkm1)
            if not random_mask:
                d_in = zk.unsqueeze(3) * (u_check - l_check) + l_check
                d_in -= xkm1.unsqueeze(2)
            else:
                xkm1_shape = xkm1.shape[-1:]
            xkm1 = utils.apply_transforms(lin_k.pre_transform, xkm1, inverse=True)

        if not random_mask:
            # Compute the mask from the Anderson oracle.
            d = masked_op.unmasked_multiply(d_in)
            Istar_k = (d >= 0).type(torch.bool if d.is_cuda else torch.uint8)  # torch.where notimplemented for cpu-bool
        else:
            # Sample a mask uniformly from {0, 1}.
            zk_shape = (unfolded_zk if type(lin_k) in [utils.ConvOp, utils.BatchConvOp] else zk).shape[:3]
            Istar_k = torch.randint(0, 2, zk_shape + xkm1_shape, device=xk.device, dtype=torch.bool)

        if do_intermediates or random_mask:
            masked_op.set_mask(Istar_k)
            # ... and its gradient
            WI_xkm1 = masked_op.forward(unfolded_xkm1 if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]
                                        else xkm1, add_bias=False)
            nub_WIu = nubs[lay_idx - 1].unsqueeze(1) - masked_op.forward(u_check, no_unsqueeze=True, add_bias=False)
            W1mIu = nub_WIu - lin_k.get_bias()
            WIl = masked_op.forward(l_check, no_unsqueeze=True, add_bias=False)
            exp_k_grad = xk - WI_xkm1 + (1 - zk) * WIl - zk * nub_WIu

            return masked_op, Istar_k, exp_k_grad, WIl, W1mIu, unfolded_zk
        else:
            # TODO: missing gather adaptation
            d *= Istar_k
            if type(lin_k) in [utils.ConvOp, utils.BatchConvOp]:
                exp_k_grad = d.sum(dim=-2).view_as(self.sum_beta[lay_idx])
            else:
                exp_k_grad = d.sum(dim=-1)
            exp_k_grad += (xk - zk * nubs[lay_idx - 1].unsqueeze(1))
            return masked_op, Istar_k, exp_k_grad, unfolded_zk, l_check, u_check


class PrimalVars:
    """
    Class representing the primal variables xt, zt.
    They are stored as lists of tensors, for ReLU indices from 0 to n-1 for xt and 1 to n-1 for zt.
    """

    def __init__(self, xt, zt):
        """
        Given the primal vars as lists of tensors (of correct length), initialize the class with these.
        """
        self.xt = xt
        self.zt = zt

    @staticmethod
    def mid_box_initialization(dual_vars, clbs, cubs):
        """
        Initialize the primal variables to the mid-point of the box constraints (halfway through each
        variable's lower and upper bounds).
        """
        xt = []
        zt = []
        for lay_idx, layer in enumerate(dual_vars.sum_beta):
            # Initialize primals to their middle points in the box domains
            init_value = ((clbs[lay_idx] + cubs[lay_idx]) / 2).unsqueeze(1)
            xt.append(init_value.expand_as(dual_vars.sum_beta[lay_idx]).clone())
            if lay_idx > 0:
                zt.append(0.5 * torch.ones_like(dual_vars.sum_beta[lay_idx]))

        return PrimalVars(xt, zt)

    def projected_linear_combination(self, step_size, other, clbs, cubs):
        # Take projected linear combination of self and other (instance of this class).
        for lay_idx in range(len(self.xt)):
            self.xt[lay_idx].add_(step_size, other.xt[lay_idx])
            self.xt[lay_idx] = torch.min(torch.max(self.xt[lay_idx], clbs[lay_idx].unsqueeze(1)),
                                         cubs[lay_idx].unsqueeze(1))
            if lay_idx > 0:
                self.zt[lay_idx-1].add_(step_size, other.zt[lay_idx - 1])
                self.zt[lay_idx-1].clamp_(0, 1)

    def add_(self, step_size, to_add):
        for xk, other_xk in zip(self.xt, to_add.xt):
            xk.add_(step_size, other_xk)
        for zk, other_zk in zip(self.zt, to_add.zt):
            zk.add_(step_size, other_zk)
        return self

    def add_cte_(self, cte):
        for xk in self.xt:
            xk.add_(cte)
        for zk in self.zt:
            zk.add_(cte)
        return self

    def addcmul_(self, coeff, to_add1, to_add2):
        for xk, other_xk1, other_xk2 in zip(self.xt, to_add1.xt, to_add2.xt):
            xk.addcmul_(coeff, other_xk1, other_xk2)
        for zk, other_zk1, other_zk2 in zip(self.zt, to_add1.zt, to_add2.zt):
            zk.addcmul_(coeff, other_zk1, other_zk2)
        return self

    def addcdiv_(self, coeff, num, denom):
        for xk, num_xk, denom_xk in zip(self.xt, num.xt, denom.xt):
            xk.addcdiv_(coeff, num_xk, denom_xk)
        for zk, num_zk, denom_zk in zip(self.zt, num.zt, denom.zt):
            zk.addcdiv_(coeff, num_zk, denom_zk)
        return self

    def div_cte_(self, denom):
        for xk in self.xt:
            xk.div_(denom)
        for zk in self.zt:
            zk.div_(denom)
        return self

    def mul_(self, coeff):
        for xk in self.xt:
            xk.mul_(coeff)
        for zk in self.zt:
            zk.mul_(coeff)
        return self

    def zero_like(self):
        new_xt = []
        new_zt = []
        for xk in self.xt:
            new_xt.append(torch.zeros_like(xk))
        for zk in self.zt:
            new_zt.append(torch.zeros_like(zk))
        return self.__class__(new_xt, new_zt)

    def sqrt(self):
        new_xt = [xk.sqrt() for xk in self.xt]
        new_zt = [zk.sqrt() for zk in self.zt]
        return self.__class__(new_xt, new_zt)

    def clone(self):
        new_xt = [xk.clone() for xk in self.xt]
        new_zt = [zk.clone() for zk in self.zt]
        return self.__class__(new_xt, new_zt)

    def div(self, to_divide):
        new_xt = []
        new_zt = []
        for xk, other_xk in zip(self.xt, to_divide.xt):
            new_xt.append(xk.div(other_xk))
        for zk, other_zk in zip(self.zt, to_divide.zt):
            new_zt.append(zk.div(other_zk))
        return self.__class__(new_xt, new_zt)


class MaskedLinearOp:
    """
    Implements forward/backward masked linear operator.
    mask is the weights mask for the layer (a batch of weight matrices). lin_k is the underlying LinearOp.
    """
    def __init__(self, lin_k):
        self.lin_k = lin_k

    def set_mask(self, mask):
        # Assumes any pre_transform has already been applied to the mask
        unsqueezed_weights = self.lin_k.weights.unsqueeze(1) if type(self.lin_k) is utils.BatchLinearOp else \
            self.lin_k.weights.view((1, 1, *self.lin_k.weights.shape))
        self.WI = (unsqueezed_weights * mask)

    def set_WI(self, WI):
        self.WI = WI

    def unmasked_multiply(self, input):
        # Assumes any pre_transform has already been applied to the input
        unsqueezed_weights = self.lin_k.weights.unsqueeze(1) if type(self.lin_k) is utils.BatchLinearOp else \
            self.lin_k.weights.view((1,1,*self.lin_k.weights.shape))
        output = (unsqueezed_weights * input)
        return output

    def forward(self, input, no_unsqueeze=False, add_bias=True):
        # no_unsqueeze=True assumes any pre_transform in self.lin_k has already been applied
        if self.lin_k.pre_transform is not None and (not no_unsqueeze):
            input = utils.apply_transforms(self.lin_k.pre_transform, input)

        if not no_unsqueeze:
            input = input.unsqueeze(2)
        output = (self.WI * input).sum(dim=-1)

        if add_bias:
            output += self.lin_k.get_bias()
        return output

    def backward(self, input):
        back_out = (self.WI * input.unsqueeze(-1)).sum(dim=-2)
        if self.lin_k.pre_transform is not None:
            back_out = utils.apply_transforms(self.lin_k.pre_transform, back_out, inverse=True)
        return back_out


class MaskedConvOp:
    """
    Implements forward/backward masked convolutional operator, relying on unfolding and folding the convolutional
    operator. mask is the weights mask that operates in the unfolded space, lin_k is the underlying ConvOp,
     in_ex/out_ex is an input/output example to retrieve the shapes.
    Gather means that this masked convolutional operator will be applied only on a subset of the out neurons.
    """
    def __init__(self, lin_k, in_ex, out_ex, gather=None):
        self.lin_k = lin_k
        self.out_ex = out_ex
        self.in_spat_shape = utils.apply_transforms(lin_k.pre_transform, in_ex).shape[-2:]
        self.unfolded_W_k = self.lin_k.unfold_weights(gather=gather)
        self.do_gather = (gather is not None)
        self.gather = gather

        if self.lin_k.prescale is not None:
            self.unfolded_prescale = self.lin_k.unfold_input(self.lin_k.prescale.unsqueeze(1), gather=gather,
                                                             no_pre_transform=True)
        else:
            self.unfolded_prescale = None

    def set_mask(self, mask):
        # Assumes any pre_transform has already been applied to the mask
        batch_shape = mask.shape[:2]
        view_shape = batch_shape + self.unfolded_W_k.shape[-2:] + (1,) if self.do_gather else \
            (1, 1, *self.unfolded_W_k.shape, 1)
        self.WI = self.unfolded_W_k.view(view_shape) * mask

    def set_WI(self, WI):
        self.WI = WI

    def unmasked_multiply(self, inp):
        # Assumes any pre_transform has already been applied to the input
        if self.lin_k.prescale is not None:
            inp = inp * self.unfolded_prescale.unsqueeze(-3)
        batch_shape = inp.shape[:2]
        view_shape = batch_shape + self.unfolded_W_k.shape[-2:] + (1,) if self.do_gather else \
            (1, 1, *self.unfolded_W_k.shape, 1)
        output = self.unfolded_W_k.view(view_shape) * inp
        return output

    def forward(self, input, unfold_input=False, no_unsqueeze=False, add_bias=True):
        # In case of gather (masked operator only over selected neurons), fill the missing output entries with 0s
        # Assumes any pre_transform in self.lin_k has already been applied (via the input unfolding)
        if unfold_input:
            input = self.lin_k.unfold_input(input, gather=self.gather)
        if self.lin_k.prescale is not None:
            c_prescale = self.unfolded_prescale.unsqueeze(-3) if no_unsqueeze else \
                self.unfolded_prescale
            input = input * c_prescale

        if not no_unsqueeze:
            input = input.unsqueeze(2)

        output = (self.WI * input).sum(dim=-2)
        if self.do_gather:
            batch_shape = self.out_ex.shape[:2]
            # Fill with zeros missing dimensions.
            output_holder = torch.zeros(
                (self.out_ex.shape[:2] + (utils.prod(self.out_ex.shape[2:]),)),
                device=self.out_ex.device, dtype=self.out_ex.dtype)
            output = output_holder.scatter_(-1, self.gather[2], output.view(*batch_shape, -1))

        output = output.view_as(self.out_ex)

        if add_bias:
            output += self.lin_k.get_bias()
        return output

    def backward(self, input):

        input = self.lin_k.unfold_output(input)
        if self.do_gather:
            # Use only relevant subset of the output space.
            restr_channel, restr_spatial = self.gather[0].shape[-1], self.gather[1].shape[-1]
            input = input.view((*input.shape[:2], -1)).gather(-1, self.gather[2]).view(
                (*input.shape[:2], restr_channel, restr_spatial))

        back_out = (self.WI * input.unsqueeze(-2)).sum(dim=-3)

        if self.do_gather:
            # Fill missing (not computed) pieces of the input space with zeros, recovering the (o_2*o_3) shape
            # before folding again.
            output_holder = torch.zeros(
                (back_out.shape[:3] + (utils.prod(self.out_ex.shape[3:]),)),
                device=back_out.device, dtype=back_out.dtype)
            kernel_shape = back_out.shape[2]
            back_out = output_holder.scatter_(-1, self.gather[1].unsqueeze(-2).expand(
                (self.gather[1].shape[:-1] + (kernel_shape,) + self.gather[1].shape[-1:])), back_out)

        back_out = self.lin_k.fold_unfolded_input(back_out, self.in_spat_shape)

        if self.lin_k.prescale is not None:
            back_out = back_out * self.lin_k.prescale.unsqueeze(1)
        if self.lin_k.pre_transform is not None:
            back_out = utils.apply_transforms(self.lin_k.pre_transform, back_out, inverse=True)
        return back_out


class AndersonPInit(ParentInit):
    """
    Parent Init class for Anderson-relaxation-based solvers.
    """
    def __init__(self, parent_duals):
        # parent_duals are the dual values (instance of DualVars) at parent termination
        self.duals = parent_duals

    def to_cpu(self):
        # Move content to cpu.
        for varname in self.duals.__dict__:
            self.duals.__dict__[varname] = [cvar.cpu() for cvar in self.duals.__dict__[varname]]

    def to_device(self, device):
        # Move content to device "device"
        for varname in self.duals.__dict__:
            self.duals.__dict__[varname] = [cvar.to(device) for cvar in self.duals.__dict__[varname]]

    def as_stack(self, stack_size):
        # Repeat (copies) the content of this parent init to form a stack of size "stack_size"
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.sum_beta, self.duals.sum_Wp1Ibetap1, self.duals.sum_W1mIubeta,
                self.duals.sum_WIlbeta, self.duals.fs, self.duals.gs]
        for varset in constructor_vars:
            stacked_dual_list.append(self.do_stack_list(varset, stack_size))
        return AndersonPInit(DualVars(*stacked_dual_list))

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        for varname in self.duals.__dict__:
            for x_idx in range(len(self.duals.__dict__[varname])):
                self.set_parent_entries(self.duals.__dict__[varname][x_idx],
                                        parent_solution.duals.__dict__[varname][x_idx], batch_idx)

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.sum_beta, self.duals.sum_Wp1Ibetap1, self.duals.sum_W1mIubeta,
                            self.duals.sum_WIlbeta, self.duals.fs, self.duals.gs]
        for varset in constructor_vars:
            stacked_dual_list.append(self.get_entry_list(varset, batch_idx))
        return AndersonPInit(DualVars(*stacked_dual_list))

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        # this operation makes sense only in the BaB context (single output neuron), when both lb and ub where computed.
        assert self.duals.alpha[0].shape[1] == 2

        stacked_dual_list = []
        constructor_vars = [self.duals.alpha, self.duals.sum_beta, self.duals.sum_Wp1Ibetap1, self.duals.sum_W1mIubeta,
                            self.duals.sum_WIlbeta, self.duals.fs, self.duals.gs]
        for varset in constructor_vars:
            stacked_dual_list.append(self.lb_only_list(varset))
        return AndersonPInit(DualVars(*stacked_dual_list))

    def get_bounding_score(self, x_idx):
        # Get a score for which intermediate bound to tighten on layer x_idx (larger is better)
        # For Anderson, only using information from the bigm formulation.
        # The explanation for this scoring is: "the magnitude of the multipliers for the activation's upper constraints"
        scores = self.duals.beta_0[x_idx] + self.duals.beta_1[x_idx]
        if scores.dim() > 3:
            scores = scores.view(*scores.shape[:2], -1)
        lb_index = 0 if scores.shape[1] == 1 else 1
        return scores[:, lb_index]
