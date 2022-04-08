import math
import time
import torch
from torch.nn import functional as F


def bdot(elt1, elt2):
    # Batch dot product
    return (elt1 * elt2).view(*elt1.shape[:2], -1).sum(-1)


def bl2_norm(bv):
    return (bv * bv).view(*bv.shape[:2], -1).sum(-1)


def prod(elts):
    if type(elts) in [int, float]:
        return elts
    else:
        prod = 1
        for elt in elts:
            prod *= elt
        return prod

def compute_output_padding(l_0, layer):
    # the following is done because when stride>1, conv2d maps multiple output shapes to the same input shape.
    # https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d
    w_1 = layer.weight
    (output_padding_0, output_padding_1) = (0, 0)
    if (l_0.shape[2] + 2*layer.padding[0] - layer.dilation[0]*(w_1.shape[2]-1) -1) % layer.stride[0] !=0:
        h_out_0 = math.floor((l_0.shape[2] + 2*layer.padding[0] - layer.dilation[0]*(w_1.shape[2]-1) -1) / layer.stride[0] + 1)
        output_padding_0 = l_0.shape[2] - 1 + 2*layer.padding[0] - layer.dilation[0]*(w_1.shape[2]-1) - (h_out_0 - 1)*layer.stride[0]
    if (l_0.shape[3] + 2*layer.padding[1] - layer.dilation[1]*(w_1.shape[3]-1) -1) % layer.stride[1] !=0:
        h_out_1 = math.floor((l_0.shape[3] + 2*layer.padding[1] - layer.dilation[1]*(w_1.shape[3]-1) -1) / layer.stride[1] + 1)
        output_padding_1 = l_0.shape[3] - 1 + 2*layer.padding[1] - layer.dilation[1]*(w_1.shape[3]-1) - (h_out_1 - 1)*layer.stride[1]

    return (output_padding_0, output_padding_1)

class LinearOp:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]
        self.flatten_from_shape = None
        self.preshape = (self.in_features,)
        self.postshape = (self.out_features,)

    def normalize_outrange(self, lbs, ubs):
        inv_range = 1.0 / (ubs - lbs)
        self.bias = inv_range * (2 * self.bias - ubs - lbs)
        self.weights = 2 * inv_range.unsqueeze(1) * self.weights

    def forward(self, inp):
        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)

        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        forw_out = torch.bmm(
            inp,
            self.weights.t().unsqueeze(0).expand((inp.shape[0], self.weights.shape[1], self.weights.shape[0]))
        )
        forw_out += self.bias.view((1,) * (inp.dim() - 1) + self.weights.shape[:1])

        return forw_out

    def forward_no_bias(self, inp):
        # same as above, but without adding the bias. Used in SA-L1
        # turns the inp into column vector for matmul if applicable
        if len(inp.shape) == 3:
            forw_out = self.weights.unsqueeze(0).unsqueeze(0) @ inp.unsqueeze(-1)
            forw_out.squeeze_(-1)
        else:
            forw_out = self.weights.unsqueeze(0).unsqueeze(0) @ inp
        return forw_out


    def interval_forward(self, lb_in, ub_in):
        if self.flatten_from_shape is not None:
            lb_in = lb_in.view(lb_in.shape[0], -1)
            ub_in = ub_in.view(ub_in.shape[0], -1)

        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = LinearOp(pos_wt, self.bias)
        neg_lay = LinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb_in.unsqueeze(1)) + neg_lay.forward(ub_in.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub_in.unsqueeze(1)) + neg_lay.forward(lb_in.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    def backward(self, out):

        # IMPORTANT: batch matmul is bugged, bmm + expand needs to be used, instead
        # https://discuss.pytorch.org/t/unexpected-huge-memory-cost-of-matmul/41642
        back_inp = torch.bmm(
            out,
            self.weights.unsqueeze(0).expand((out.shape[0], self.weights.shape[0], self.weights.shape[1]))
        )

        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def subset_backward(self, out, subset):

        # same as backward, but only pick a subset of rows - used in subset L1
        # subset is a list of indices
        # weights = torch.stack([self.weights[subset[batch, :], :] for batch in range(out.shape[0])])
        weights = torch.gather(self.weights.unsqueeze(0).expand(subset.shape[0], self.out_features, self.in_features),
                               1, subset.unsqueeze(-1).expand(subset.shape[0], subset.shape[1], self.weights.shape[-1]))
        back_inp = torch.bmm(
            out,
            weights
        )

        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def subset_backward_non_batch(self, out, subset):

        # same as backward, but only pick a subset of rows - used in subset L1
        # subset is a list of indices
        back_inp = torch.bmm(
            out,
            self.weights.unsqueeze(0).expand((out.shape[0], self.weights.shape[0], self.weights.shape[1]))[:, subset, :]
        )

        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def get_output_shape(self, in_shape):
        """
        Return the output shape (as tuple) given the input shape. The input shape is the shape will influence the output
        shape.
        """
        return (*in_shape[:2], self.out_features)

    def get_bias(self):
        """
        Return the bias with the correct unsqueezed shape.
        """
        return self.bias.view((1, 1, *self.bias.shape))

    def get_unconditioned_bias(self):
        """
        Return the bias (without first layer pre-conditioning) with the correct unsqueezed shape.
        As this class does not deal with the first layer, which is conditioned, returns the only bias.
        """
        return self.get_bias()

    def sum_column(self, batch_shape, additional_coeffs=None):
        # returns the sum of each column in the right shape. Used in L1 decomposition code
        weights = self.weights.unsqueeze(1).repeat((*batch_shape,) + (1, 1))
        # multiply additional_coeffs if applicable
        if additional_coeffs is not None:
            weights = additional_coeffs.unsqueeze(-2) @ weights
        return weights.sum(-2)

    def __repr__(self):
        return f'<Linear: {self.in_features} -> {self.out_features}>'

    def flatten_from(self, shape):
        self.flatten_from_shape = shape


class BatchLinearOp(LinearOp):
    """
    Exactly same interface and functionality as LinearOp, but batch of weights and biases.
    Ignores flatten from shape as this will never be used as a final layer
    """
    def __init__(self, weights, bias, unconditioned_bias):
        self.weights = weights
        self.bias = bias
        # stored for context where W_1 is absent but b_1 is not (hence the normalization is mathematically wrong)
        self.unconditioned_bias = unconditioned_bias
        self.batch_size = weights.shape[0]
        self.out_features = weights.shape[1]
        self.in_features = weights.shape[2]
        self.flatten_from_shape = None
        self.preshape = (self.in_features,)
        self.postshape = (self.out_features,)

    def forward(self, inp):
        if self.flatten_from_shape is not None:
            inp = inp.view(*inp.shape[:2], -1)
        domain_batch = inp.shape[0]
        layer_batch = inp.shape[1]
        # TODO: there must be a more memory-efficient way to perform this
        forw_out = (self.weights.unsqueeze(1) @ inp.unsqueeze(-1)).view(
            (domain_batch, layer_batch, self.weights.shape[1], 1)).squeeze(-1)
        forw_out += self.bias.unsqueeze(1)
        return forw_out

    def forward_no_bias(self, inp):
        # same as above, but without adding the bias. Used in SA-L1
        # turns the inp into column vector for matmul if applicable
        if len(inp.shape) == 3:
            forw_out = self.weights.unsqueeze(1) @ inp.unsqueeze(-1)
            forw_out.squeeze_(-1)
        else:
            forw_out = self.weights.unsqueeze(1) @ inp
        return forw_out

    def backward(self, out):
        # back_inp = out @ self.weights
        # TODO: there must be a more memory-efficient way to perform this
        back_inp = (out.unsqueeze(1) @ self.weights.unsqueeze(1)).squeeze(1)
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def subset_backward(self, out, subset):
        # weights = torch.stack([self.weights.unsqueeze(1)[batch, :, subset[batch, :], :] for batch in range(out.shape[0])], dim=0)
        weights = torch.gather(self.weights.unsqueeze(1), 2,
                               subset.unsqueeze(1).unsqueeze(-1).expand(subset.shape[0], 1, subset.shape[1], self.weights.shape[-1]))
        # assert (weights == gather).all()
        back_inp = (out.unsqueeze(1) @ weights).squeeze(1)
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def subset_backward_non_batch(self, out, subset):
        back_inp = (out.unsqueeze(1) @ self.weights.unsqueeze(1)[:, :, subset, :]).squeeze(1)
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0], out.shape[1]) + self.flatten_from_shape)
        return back_inp

    def interval_forward(self, lb_in, ub_in):
        if self.flatten_from_shape is not None:
            lb_in = lb_in.view(lb_in.shape[0], -1)
            ub_in = ub_in.view(ub_in.shape[0], -1)

        pos_wt = torch.clamp(self.weights, 0, None)
        neg_wt = torch.clamp(self.weights, None, 0)
        pos_lay = BatchLinearOp(pos_wt, self.bias)
        neg_lay = BatchLinearOp(neg_wt, torch.zeros_like(self.bias))
        lb_out = (pos_lay.forward(lb_in.unsqueeze(1)) + neg_lay.forward(ub_in.unsqueeze(1))).squeeze(1)
        ub_out = (pos_lay.forward(ub_in.unsqueeze(1)) + neg_lay.forward(lb_in.unsqueeze(1))).squeeze(1)

        return lb_out, ub_out

    def get_bias(self):
        """
        Return the bias with the correct unsqueezed shape.
        """
        return self.bias.unsqueeze(1)

    def get_unconditioned_bias(self):
        """
        Return the bias (without first layer pre-conditioning) with the correct unsqueezed shape.
        """
        return self.unconditioned_bias.view((1, 1, *self.unconditioned_bias.shape))

    def sum_column(self, batch_shape, additional_coeffs=None):
        # returns the sum of each column in the right shape. Used in L1 decomposition code
        # make sure the required batch shape is correct
        assert self.weights.shape[0] == batch_shape[0]
        weights = self.weights.unsqueeze(1).repeat(1, batch_shape[1], 1, 1)
        if additional_coeffs is not None:
            weights = additional_coeffs @ weights
        return weights.sum(-2)


    def __repr__(self):
        return f'{self.batch_size} x <Linear: {self.in_features} -> {self.out_features}>'


class ConvOp:
    def __init__(self, weights, bias,
                 stride, padding, dilation, groups, output_padding=0):
        self.weights = weights
        if bias.dim() == 1:
            self.bias = bias.view(-1, 1, 1)
        else:
            self.bias = bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]
        self.kernel_height = weights.shape[2]
        self.kernel_width = weights.shape[3]

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.output_padding = output_padding

        self.prescale = None

        self.preshape = None
        self.postshape = None

    def add_prerescaling(self, prescale):
        if self.prescale is None:
            self.prescale = prescale
        else:
            self.prescale = self.prescale * prescale

    def forward(self, inp):
        if self.prescale is not None:
            inp = inp * self.prescale

        if inp.dim() == 5:
            # batch over multiple domains
            domain_batch_size = inp.shape[0]
            batch_size = inp.shape[1]
            unfolded_in = inp.view(domain_batch_size * batch_size, *inp.shape[2:])
            fold_back = True
        else:
            unfolded_in = inp
            fold_back = False

        out = F.conv2d(unfolded_in, self.weights, None, self.stride, self.padding, self.dilation, self.groups)

        if fold_back:
            out = out.view(domain_batch_size, batch_size, *out.shape[1:])
        out += self.bias.view(*((1,) * (inp.dim() - 3)), *self.bias.shape)
        if self.preshape is None:
            # Write down the shape of the inputs/outputs of this network.
            # The assumption is that this will remain constant (fixed input size)
            self.preshape = inp.shape[1:]
            self.postshape = out.shape[1:]
        return out

    def interval_forward(self, lb_in, ub_in):
        if self.prescale is not None:
            lb_in = lb_in * self.prescale
            ub_in = ub_in * self.prescale

        pos_wts = torch.clamp(self.weights, 0, None)
        neg_wts = torch.clamp(self.weights, None, 0)

        unbiased_lb_out = (F.conv2d(lb_in, pos_wts, None, self.stride, self.padding, self.dilation, self.groups)
                           + F.conv2d(ub_in, neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        unbiased_ub_out = (F.conv2d(ub_in, pos_wts, None,
                                    self.stride, self.padding,
                                    self.dilation, self.groups)
                           + F.conv2d(lb_in, neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        lb_out = unbiased_lb_out + self.bias.unsqueeze(0)
        ub_out = unbiased_ub_out + self.bias.unsqueeze(0)
        if self.preshape is None:
            # Write down the shape of the inputs/outputs of this network.
            # The assumption is that this will remain constant (fixed input size)
            self.preshape = lb_in.shape[1:]
            self.postshape = ub_in.shape[1:]
        return lb_out, ub_out


    def backward(self, out):
        if out.dim() == 5:
            # batch over multiple domains
            domain_batch_size = out.shape[0]
            batch_size = out.shape[1]
            unfolded_out = out.reshape(domain_batch_size * batch_size, *out.shape[2:])
            fold_back = True
        else:
            unfolded_out = out
            fold_back = False
        inp = F.conv_transpose2d(unfolded_out, self.weights, None,
                                 stride=self.stride, padding=self.padding,
                                 output_padding=self.output_padding, groups=self.groups,
                                 dilation=self.dilation)
        if fold_back:
            inp = inp.view(domain_batch_size, batch_size, *inp.shape[1:])
        if self.prescale is not None:
            inp = inp * self.prescale
        return inp

    def _check_backward(self, inp):
        # Check that we get a good implementation of backward / forward as
        # transpose from each other.
        assert inp.dim() == 4, "Make sure that you test with a batched input"
        inp = torch.randn_like(inp)
        through_forward = self.forward(inp) - self.bias
        nb_outputs = prod(through_forward.shape[1:])
        targets = torch.eye(nb_outputs, device=self.weights.device)
        targets = targets.view((nb_outputs,) + through_forward.shape[1:])

        cost_coeffs = self.backward(targets)

        out = (cost_coeffs.unsqueeze(0) * inp.unsqueeze(1)).sum(4).sum(3).sum(2)
        out = out.view(*through_forward.shape)

        diff = (out - through_forward).abs().max()

    def equivalent_linear(self, inp):

        assert inp.dim() == 4, "No batched input"

        zero_inp = torch.zeros_like(inp).unsqueeze(0)
        eq_b = self.forward(zero_inp).squeeze(0)
        out_shape = eq_b.shape
        nb_outputs = prod(out_shape)
        targets = torch.eye(nb_outputs, device=self.weights.device)
        targets = targets.view((nb_outputs,) + out_shape)

        eq_W = self.backward(targets).view((nb_outputs, -1))
        eq_b = eq_b.view(-1)

        eq_lin = LinearOp(eq_W, eq_b)

        # # CHECKING
        # nb_samples = 1000
        # rand_inp = torch.randn((nb_samples,) + inp.shape)
        # rand_out = self.forward(rand_inp)
        # flat_randinp = rand_inp.view(nb_samples, -1)
        # flat_randout = eq_lin.forward(flat_randinp)
        # error = (rand_out.view(-1) - flat_randout.view(-1)).abs().max()
        # print(f"Convolution to Linear error: {error}")
        # assert error < 1e-5

        return eq_lin

    def get_output_shape(self, in_shape):
        """
        Return the output shape (as tuple) given the input shape.
        Assumes that in_shape has five dimensions (the first two are the batch size).
        """
        c_out = self.out_features
        h_out = (in_shape[3] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_height - 1) - 1)/self.stride[0] + 1
        h_out = math.floor(h_out)
        w_out = (in_shape[4] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_width - 1) - 1)/self.stride[1] + 1
        w_out = math.floor(w_out)
        return (*in_shape[:2], c_out, h_out, w_out)

    def unfold_input(self, inp, gather=None):
        """
        Unfold an input vector reflecting the actual slices in the convolutional operator.
        See https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        If gather is available, returns only selected indices of the unfolded vector (contained in the gather tensor).
        As the selected indices will vary across channels and batch entries, the vector is expanded across those
        dimensions.
        If gather is available, returns only selected indices of the unfolded vector (contained in the gather tensor).
        As the selected indices will vary across channels and batch entries, the vector is expanded across those
        dimensions.
        """
        if inp.dim() == 5:
            # batch over multiple domains
            domain_batch_size = inp.shape[0]
            batch_size = inp.shape[1]
            in_4dim = inp.view(domain_batch_size * batch_size, *inp.shape[2:])
            fold_back = True
        else:
            in_4dim = inp
            fold_back = False

        unfolded_inp = torch.nn.functional.unfold(
            in_4dim, (self.kernel_height, self.kernel_width), dilation=self.dilation,
            padding=self.padding, stride=self.stride)

        if fold_back:
            unfolded_inp = unfolded_inp.view(domain_batch_size, batch_size, *unfolded_inp.shape[1:])
        if gather is not None:
            # Select given entries of the unfolded vector, across spatial dimensions.
            kernel_shape = unfolded_inp.shape[2]
            batch_size = gather[1].shape[1]
            unfolded_inp = unfolded_inp.expand(((domain_batch_size, batch_size) + unfolded_inp.shape[2:]))
            unfolded_inp = unfolded_inp.gather(-1, gather[1].unsqueeze(-2).expand(
                (gather[1].shape[:-1] + (kernel_shape,) + gather[1].shape[-1:])))
        return unfolded_inp

    def unfold_weights(self, gather=None):
        """
        Unfold the weights to go with the actual slices in the convolutional operator. (see unfold_input)
        See https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        returns a view
        If gather is available, returns only selected indices of the unfolded vector (contained in the gather tensor).
        """
        unfolded_weights = self.weights.reshape(self.weights.shape[0], -1)
        if gather is not None:
            batch_shape = gather[0].shape[:2]
            kernel_shape = unfolded_weights.shape[-1]
            unfolded_weights = unfolded_weights.expand(batch_shape + unfolded_weights.shape)
            unfolded_weights = unfolded_weights.gather(-2, gather[0].unsqueeze(-1).expand(
                (gather[0].shape + (kernel_shape,))))
        return unfolded_weights

    def unfold_output(self, out, gather=None):
        """
        Unfold a vector representing the convolutional output, reflecting the format of unfolded inputs/weights
        See functions unfold_input and unfold_weights.
        If gather is available, returns only selected indices of the unfolded vector (contained in the gather tensor).
        """
        batch_channel_shape = out.shape[:3]  # linearize everything that's not batches or channels
        unfolded_out = out.view((*batch_channel_shape, -1))
        if gather is not None:
            # Restrict over channels and spatial dimensions.
            restr_channel, restr_spatial = gather[0].shape[-1], gather[1].shape[-1]
            unfolded_out = unfolded_out.view((*out.shape[:2], -1)).gather(-1, gather[2]).view(
                (*out.shape[:2], restr_channel, restr_spatial))
        return unfolded_out

    def fold_unfolded_input(self, unfolded_inp, folded_inp_spat_shape):
        """
        Fold a vector unfolded with unfold_input.
        :param folded_inp_spat_shape: the spatial shape of the desired output
        """
        if unfolded_inp.dim() == 4:
            # batch over multiple domains
            domain_batch_size = unfolded_inp.shape[0]
            batch_size = unfolded_inp.shape[1]
            in_3dim = unfolded_inp.view(domain_batch_size * batch_size, *unfolded_inp.shape[2:])
            fold_back = True
        else:
            in_3dim = unfolded_inp
            fold_back = False

        folded_inp = torch.nn.functional.fold(
            in_3dim, folded_inp_spat_shape, (self.kernel_height, self.kernel_width), dilation=self.dilation,
            padding=self.padding, stride=self.stride)

        if fold_back:
            folded_inp = folded_inp.view(domain_batch_size, batch_size, *folded_inp.shape[1:])

        return folded_inp

    def zero_scatter_folded_out(self, out, indices):
        """
        Scatter with zeros at indices a vector in the (folded) output space of this vector.
        """
        out_vector = torch.zeros_like(out)
        extended_indices = indices[2]
        unfolded_out = out.view((*out.shape[:2], -1))
        out_vector.view((*out.shape[:2], -1)).scatter_(-1, extended_indices, unfolded_out.gather(-1, extended_indices))
        return out_vector

    def get_bias(self):
        """
        Return the bias with the correct unsqueezed shape.
        """
        return self.bias.view((1, 1, *self.bias.shape))

    def get_unconditioned_bias(self):
        """
        Return the bias (without first layer pre-conditioning) with the correct unsqueezed shape.
        As this class does not deal with the first layer, which is conditioned, returns the only bias.
        """
        return self.get_bias()

    def __repr__(self):
        return f"<Conv[{self.kernel_height}, {self.kernel_width}]: {self.in_features} -> {self.out_features}"


class BatchConvOp(ConvOp):
    """
    Exactly same interface and functionality as ConvOp, but batch of weights and biases.
    Ignores flatten from shape as this will never be used as a final layer
    """

    def __init__(self, weights, bias, unconditioned_bias,
                 stride, padding, dilation, groups, output_padding=0):
        self.weights = weights
        if bias.dim() == 1:
            self.bias = bias.view(-1, 1, 1)
        else:
            self.bias = bias
        # stored for context where W_1 is absent but b_1 is not (hence the normalization is mathematically wrong)
        if unconditioned_bias.dim() == 1:
            self.unconditioned_bias = unconditioned_bias.view(-1, 1, 1)
        else:
            self.unconditioned_bias = unconditioned_bias
        self.out_features = weights.shape[0]
        self.in_features = weights.shape[1]
        self.kernel_height = weights.shape[2]
        self.kernel_width = weights.shape[3]
        self.batch_size = bias.shape[0]

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.output_padding = output_padding

        self.prescale = None

        self.preshape = None
        self.postshape = None

    def forward(self, inp):
        if self.prescale is not None:
            inp = inp * self.prescale.unsqueeze(1).expand_as(inp)

        if inp.dim() == 5:
            # batch over multiple domains
            domain_batch_size = inp.shape[0]
            batch_size = inp.shape[1]
            unfolded_in = inp.view(domain_batch_size * batch_size, *inp.shape[2:])
            fold_back = True
        else:
            unfolded_in = inp
            fold_back = False

        out = F.conv2d(unfolded_in, self.weights, None,
                       self.stride, self.padding, self.dilation, self.groups)

        if fold_back:
            out = out.view(domain_batch_size, batch_size, *out.shape[1:])

        out += self.bias.unsqueeze(1).expand_as(out)
        if self.preshape is None:
            # Write down the shape of the inputs/outputs of this network.
            # The assumption is that this will remain constant (fixed input size)
            self.preshape = inp.shape[1:]
            self.postshape = out.shape[1:]
        return out

    def interval_forward(self, lb_in, ub_in):
        if self.prescale is not None:
            lb_in = lb_in * self.prescale
            ub_in = ub_in * self.prescale

        pos_wts = torch.clamp(self.weights, 0, None)
        neg_wts = torch.clamp(self.weights, None, 0)

        unbiased_lb_out = (F.conv2d(lb_in, pos_wts, None,
                                   self.stride, self.padding,
                                   self.dilation, self.groups)
                           + F.conv2d(ub_in, neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        unbiased_ub_out = (F.conv2d(ub_in, pos_wts, None,
                                    self.stride, self.padding,
                                    self.dilation, self.groups)
                           + F.conv2d(lb_in, neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        lb_out = unbiased_lb_out + self.bias
        ub_out = unbiased_ub_out + self.bias
        if self.preshape is None:
            # Write down the shape of the inputs/outputs of this network.
            # The assumption is that this will remain constant (fixed input size)
            self.preshape = lb_in.shape[1:]
            self.postshape = ub_in.shape[1:]
        return lb_out, ub_out

    def backward(self, out):
        if out.dim() == 5:
            # batch over multiple domains
            domain_batch_size = out.shape[0]
            batch_size = out.shape[1]
            unfolded_out = out.view(domain_batch_size * batch_size, *out.shape[2:])
            fold_back = True
        else:
            unfolded_out = out
            fold_back = False
        inp = F.conv_transpose2d(unfolded_out, self.weights, None,
                                 stride=self.stride, padding=self.padding,
                                 output_padding=self.output_padding, groups=self.groups,
                                 dilation=self.dilation)
        if fold_back:
            inp = inp.view(domain_batch_size, batch_size, *inp.shape[1:])
        if self.prescale is not None:
            inp = inp * self.prescale.unsqueeze(1)
        return inp

    def _check_backward(self, inp):
        # Check that we get a good implementation of backward / forward as
        # transpose from each other.
        assert inp.dim() == 4, "Make sure that you test with a batched input"
        inp = torch.randn_like(inp)
        through_forward = self.forward(inp) - self.bias
        nb_outputs = prod(through_forward.shape[1:])
        targets = torch.eye(nb_outputs, device=self.weights.device)
        targets = targets.view((nb_outputs,) + through_forward.shape[1:])

        cost_coeffs = self.backward(targets)

        out = (cost_coeffs.unsqueeze(0) * inp.unsqueeze(1)).sum(4).sum(3).sum(2)
        out = out.view(*through_forward.shape)

        diff = (out - through_forward).abs().max()

    def equivalent_linear(self, inp):
        if inp.dim() == 2:
            # This is a flat input to a convolutional network.
            # I'll assume that this was because we linearized
            # some previous convolutions.

            # Figure out the number of channels
            inp_channels = self.weights.shape[1]
            # WARNING: assumption that everything is square
            size = math.sqrt(inp.size(0) / inp_channels)
            assert int(size) == size
            inp = inp.view(inp_channels, size, size)

        assert inp.dim() == 4, "No batched input"

        zero_inp = torch.zeros_like(inp).unsqueeze(0)
        eq_b = self.forward(zero_inp).squeeze(0)
        out_shape = eq_b.shape
        nb_outputs = prod(out_shape)
        targets = torch.eye(nb_outputs, device=self.weights.device)
        targets = targets.view((nb_outputs,) + out_shape)

        eq_W = self.backward(targets).view((nb_outputs, -1))
        eq_b = eq_b.view(-1)

        eq_lin = LinearOp(eq_W, eq_b)

        # # CHECKING
        # nb_samples = 1000
        # rand_inp = torch.randn((nb_samples,) + inp.shape)
        # rand_out = self.forward(rand_inp)
        # flat_randinp = rand_inp.view(nb_samples, -1)
        # flat_randout = eq_lin.forward(flat_randinp)
        # error = (rand_out.view(-1) - flat_randout.view(-1)).abs().max()
        # print(f"Convolution to Linear error: {error}")
        # assert error < 1e-5

        return eq_lin

    def get_bias(self):
        """
        Return the bias with the correct unsqueezed shape.
        """
        return self.bias.unsqueeze(1)

    def get_unconditioned_bias(self):
        """
        Return the bias (without first layer pre-conditioning) with the correct unsqueezed shape.
        """
        return self.unconditioned_bias.view((1, 1, *self.unconditioned_bias.shape))

    def __repr__(self):
        return f"{self.batch_size} x <Conv[{self.kernel_height}, {self.kernel_width}]: {self.in_features} -> {self.out_features}"


def get_relu_mask(lb, ub):
    # given a layer's lower and upper bounds (tensors), return a relu mask, which stores which relus are ambiguous.
    # 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
    passing = (lb >= 0)
    blocking = (ub <= 0)
    ambiguous = (~passing & ~blocking)
    return passing.type(torch.float) * 1 + ambiguous.type(torch.float) * (-1)


def create_final_coeffs_slice(start_batch_index, end_batch_index, batch_size, nb_out, tensor_example, node_layer_shape,
                              node=(-1, None), upper_bound=False):
    # Given indices and specifications (batch size for BaB, number of output neurons, example of used tensor and shape
    # of current last layer), create a slice of final_coeffs for dual iterative solvers (auxiliary fake last layer
    # variables indicating which neurons to compute bounds for)
    is_full_batch = (node[1] is None)
    is_part_batch = type(node[1]) is list or torch.is_tensor(node[1])
    if not is_full_batch:
        if is_part_batch:
            fixed_len = node[1].shape[1]
            if end_batch_index <= fixed_len:
                # Only upper bounds
                c_neuron_list = node[1][:, start_batch_index:end_batch_index]
                slice_coeffs = torch.zeros((batch_size, c_neuron_list.shape[1], nb_out),
                                           device=tensor_example.device, dtype=tensor_example.dtype)
                slice_coeffs.scatter_(2, c_neuron_list.unsqueeze(2), -1)
            elif start_batch_index >= fixed_len:
                # Only lower bounds
                c_neuron_list = node[1][:, start_batch_index-fixed_len:end_batch_index-fixed_len]
                slice_coeffs = torch.zeros((batch_size, c_neuron_list.shape[1], nb_out),
                                           device=tensor_example.device, dtype=tensor_example.dtype)
                slice_coeffs.scatter_(2, c_neuron_list.unsqueeze(2), 1)
            else:
                # Both lower and upper bounds
                c_neuron_list = node[1][:, start_batch_index-fixed_len:]
                slice_coeffs = torch.zeros((batch_size, c_neuron_list.shape[1], nb_out),
                                           device=tensor_example.device, dtype=tensor_example.dtype)
                slice_coeffs.scatter_(2, c_neuron_list.unsqueeze(2), -1)
                c_neuron_list = node[1][:, :end_batch_index-fixed_len]
                slice_coeffs2 = torch.zeros((batch_size, c_neuron_list.shape[1], nb_out),
                                            device=tensor_example.device, dtype=tensor_example.dtype)
                slice_coeffs2.scatter_(2, c_neuron_list.unsqueeze(2), 1)
                slice_coeffs = torch.cat([slice_coeffs, slice_coeffs2], dim=1)
            # Re-put slice_coeffs in the proper shape.
            slice_coeffs = slice_coeffs.view((batch_size, slice_coeffs.size(1),) + node_layer_shape)
        else:
            # Optimise over single neuron.
            slice_coeffs = torch.zeros_like(tensor_example).unsqueeze(1)
            if tensor_example.dim() == 2:
                slice_coeffs[:, 0, node[1]] = -1 if upper_bound else 1
            elif tensor_example.dim() == 4:
                slice_coeffs[:, 0, node[1][0], node[1][1], node[1][2]] = -1 if upper_bound else 1
            else:
                raise NotImplementedError
    else:
        slice_indices = list(range(start_batch_index, end_batch_index))
        slice_coeffs = torch.zeros((len(slice_indices), nb_out),
                                   device=tensor_example.device, dtype=tensor_example.dtype)
        slice_diag = slice_coeffs.diagonal(start_batch_index)
        slice_diag[:] = -torch.ones_like(slice_diag)
        slice_diag = slice_coeffs.diagonal(start_batch_index - nb_out)
        slice_diag[:] = torch.ones_like(slice_diag)
        slice_coeffs = slice_coeffs.expand((batch_size, *slice_coeffs.shape))
        slice_coeffs = slice_coeffs.view((batch_size, slice_coeffs.size(1),) + node_layer_shape)
    return slice_coeffs


class OptimizationTrace:
    """
    Logger for neural network bounds optimization, associated to a single bounds computation.
    Contains a number of dictionaries (indexed by the network layer the optimization refers to) containing quantities
    that describe the optimization.

    bounds_progress_per_layer: dictionary of lists for the evolution of the computed batch of bounds over the a subset of
        the iterations. These bounds might be associated to upper (stored as their negative, in the first half of the
        vector) and lower bounds.
    time_progress_per_layer: dictionary of lists which store the elapsed time associated to each of the iterations
        logged in the lists above.
    """
    def __init__(self):
        self.bounds_progress_per_layer = {}
        self.time_progress_per_layer = {}
        self.cumulative_logging_time = 0

    def start_timing(self):
        self.start_timer = time.time()

    def add_point(self, layer_idx, bounds, logging_time=None):
        # add the bounds at the current optimization state, measuring time as well
        # logging_time allows to subtract the time used for the logging computations
        if logging_time is not None:
            self.cumulative_logging_time += logging_time
        c_time = time.time() - self.start_timer - self.cumulative_logging_time
        if layer_idx in self.bounds_progress_per_layer:
            self.bounds_progress_per_layer[layer_idx].append(bounds)
        else:
            self.bounds_progress_per_layer[layer_idx] = [bounds]
        if layer_idx in self.time_progress_per_layer:
            self.time_progress_per_layer[layer_idx].append(c_time)
        else:
            self.time_progress_per_layer[layer_idx] = [c_time]

    def get_last_layer_bounds_means_trace(self, first_half_only_as_ub=False, second_half_only=False):
        """
        Get the evolution over time of the average of the last layer bounds.
        :param first_half_only_as_ub: assuming that the first half of the batches contains upper bounds, flip them and
            count only those in the average
        :return: list of singleton tensors
        """
        last_layer = sorted(self.bounds_progress_per_layer.keys())[-1]
        if first_half_only_as_ub:
            if self.bounds_progress_per_layer[last_layer][0].dim() > 1:
                bounds_trace = [-bounds[:, :int(bounds.shape[1] / 2)].mean() for bounds in
                                self.bounds_progress_per_layer[last_layer]]
            else:
                bounds_trace = [-bounds[:int(len(bounds) / 2)].mean() for bounds in
                                self.bounds_progress_per_layer[last_layer]]
        elif second_half_only:
            if self.bounds_progress_per_layer[last_layer][0].dim() > 1:
                bounds_trace = [bounds[:, int(bounds.shape[1] / 2):].mean() for bounds in
                                self.bounds_progress_per_layer[last_layer]]
            else:
                bounds_trace = [bounds[int(len(bounds) / 2):].mean() for bounds in
                                self.bounds_progress_per_layer[last_layer]]
        else:
            bounds_trace = [bounds.mean() for bounds in self.bounds_progress_per_layer[last_layer]]
        return bounds_trace

    def get_last_layer_time_trace(self):
        last_layer = sorted(self.time_progress_per_layer.keys())[-1]
        return self.time_progress_per_layer[last_layer]


class ProxOptimizationTrace(OptimizationTrace):
    """
    Logger for neural network bounds optimization, associated to a single bounds computation done via proximal methods.
    Contains a number of dictionaries (indexed by the network layer the optimization refers to) containing quantities
    that describe the optimization.

    bounds_progress_per_layer: dictionary of lists for the evolution of the computed batch of bounds over the a subset
        of the iterations. These bounds might be associated to upper (stored as their negative, in the first half of the
        vector) and lower bounds.
    objs_progress_per_layer: dictionary of lists for the evolution of the computed batch of objectives over the a subset
        of the iterations. These objectives might be associated to upper (stored in the first half of the
        vector) and lower bound computations.
    time_progress_per_layer: dictionary of lists which store the elapsed time associated to each of the iterations
        logged in the lists above.
    """

    def __init__(self):
        super().__init__()
        self.objs_progress_per_layer = {}

    def add_proximal_point(self, layer_idx, bounds, objs, logging_time=None):
        # add the bounds and objective at the current optimization state, measuring time as well
        self.add_point(layer_idx, bounds, logging_time=logging_time)
        if layer_idx in self.objs_progress_per_layer:
            self.objs_progress_per_layer[layer_idx].append(objs)
        else:
            self.objs_progress_per_layer[layer_idx] = [objs]

    def get_last_layer_objs_means_trace(self):
        """
        Get the evolution over time of the average of the last layer objectives.
        :return: list of singleton tensors
        """
        last_layer = sorted(self.objs_progress_per_layer.keys())[-1]
        objs_trace = self.objs_progress_per_layer[last_layer]
        return objs_trace
