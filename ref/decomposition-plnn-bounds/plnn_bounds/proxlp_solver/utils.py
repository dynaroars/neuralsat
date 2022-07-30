import math
import time
import torch
from abc import ABC
from torch.nn import functional as F

def bdot(elt1, elt2):
    # Batch dot product
    return (elt1 * elt2).view(elt1.shape[0], -1).sum(-1)

def bl2_norm(bv):
    return (bv * bv).view(bv.shape[0], -1).sum(-1)

def prod(elts):
    if type(elts) in [int, float]:
        return elts
    else:
        prod = 1
        for elt in elts:
            prod *= elt
        return prod

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

    def add_prerescaling(self, pre_scales):
        if self.flatten_from_shape is not None:
            # Do the flattening
            pre_scales = pre_scales.view(-1)
        self.weights = self.weights * pre_scales.unsqueeze(0)

    def forward(self, inp):
        if self.flatten_from_shape is not None:
            inp = inp.view(inp.shape[0], -1)
        return inp @ self.weights.t() + self.bias

    def interval_forward(self, lb_in, ub_in):
        if self.flatten_from_shape is not None:
            lb_in = lb_in.view(-1)
            ub_in = ub_in.view(-1)
        pos_wt = torch.clamp(self.weights.t(), 0, None)
        neg_wt = torch.clamp(self.weights.t(), None, 0)
        lb_out = (lb_in @ pos_wt + ub_in @ neg_wt + self.bias)
        ub_out = (lb_in @ neg_wt + ub_in @ pos_wt + self.bias)
        return lb_out, ub_out

    def backward(self, out):
        back_inp = out @ self.weights
        if self.flatten_from_shape is not None:
            back_inp = back_inp.view((out.shape[0],) + self.flatten_from_shape)
        return back_inp

    def get_output_shape(self, in_shape):
        """
        Return the output shape (as tuple) given the input shape. The input shape is necessary only for compatibility
        with convolutional layers.
        """
        return in_shape[0], self.out_features

    def __repr__(self):
        return f'<Linear: {self.in_features} -> {self.out_features}>'

    def flatten_from(self, shape):
        self.flatten_from_shape = shape

class ConvOp:

    def __init__(self, weights, bias,
                 stride, padding, dilation, groups):
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

        self.prescale = None
        self.postscale = None

        self.preshape = None
        self.postshape = None

    def normalize_outrange(self, lbs, ubs):
        if self.postscale is None:
            self.bias = (self.bias - (lbs + ubs)/2) / (ubs - lbs)
            self.postscale = 1 / (ubs - lbs)
        else:
            raise Exception("Not yet fought through")

    def add_prerescaling(self, prescale):
        if self.prescale is None:
            self.prescale = prescale
        else:
            self.prescale = self.prescale * prescale

    def forward(self, inp):
        if self.prescale is not None:
            inp = inp * self.prescale
        out = F.conv2d(inp, self.weights, None,
                       self.stride, self.padding, self.dilation, self.groups)
        if self.postscale is not None:
            out = out * self.postscale
        out += self.bias.unsqueeze(0)
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

        unbiased_lb_out = (F.conv2d(lb_in.unsqueeze(0), pos_wts, None,
                                   self.stride, self.padding,
                                   self.dilation, self.groups)
                           + F.conv2d(ub_in.unsqueeze(0), neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        unbiased_ub_out = (F.conv2d(ub_in.unsqueeze(0), pos_wts, None,
                                    self.stride, self.padding,
                                    self.dilation, self.groups)
                           + F.conv2d(lb_in.unsqueeze(0), neg_wts, None,
                                      self.stride, self.padding,
                                      self.dilation, self.groups))
        if self.postscale is not None:
            unbiased_lb_out = unbiased_lb_out * self.postscale
            unbiased_ub_out = unbiased_ub_out * self.postscale
        lb_out = unbiased_lb_out + self.bias.unsqueeze(0)
        ub_out = unbiased_ub_out + self.bias.unsqueeze(0)
        if self.preshape is None:
            # Write down the shape of the inputs/outputs of this network.
            # The assumption is that this will remain constant (fixed input size)
            self.preshape = lb_in.shape[1:]
            self.postshape = ub_in.shape[1:]
        return lb_out.squeeze(0), ub_out.squeeze(0)


    def backward(self, out):
        if self.postscale is not None:
            out = out * self.postscale
        inp = F.conv_transpose2d(out, self.weights, None,
                                 stride=self.stride, padding=self.padding,
                                 output_padding=0, groups=self.groups,
                                 dilation=self.dilation)
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

        assert inp.dim() == 3, "No batched input"

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
        Assumes that in_shape has four dimensions (the first is the batch size).
        """
        c_out = self.out_features
        h_out = (in_shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_height - 1) - 1)/self.stride[0] + 1
        h_out = math.floor(h_out)
        w_out = (in_shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_width - 1) - 1)/self.stride[1] + 1
        w_out = math.floor(w_out)
        return in_shape[0], c_out, h_out, w_out

    def unfold_input(self, inp):
        """
        Unfold an input vector reflecting the actual slices in the convolutional operator.
        See https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        """
        unfolded_inp = torch.nn.functional.unfold(
            inp, (self.kernel_height, self.kernel_width), dilation=self.dilation,
            padding=self.padding, stride=self.stride)
        return unfolded_inp

    def unfold_weights(self):
        """
        Unfold the weights to go with the actual slices in the convolutional operator. (see unfold_input)
        See https://pytorch.org/docs/stable/nn.html#torch.nn.Unfold
        returns a view
        """
        return self.weights.view(self.weights.shape[0], -1)

    def unfold_output(self, out):
        """
        Unfold a vector representing the convolutional output, reflecting the format of unfolded inputs/weights
        See functions unfold_input and unfold_weights.
        """
        batch_channel_shape = out.shape[:2]  # linearize everything that's not batches or channels
        unfolded_out = out.view((*batch_channel_shape, -1))
        return unfolded_out

    def fold_unfolded_input(self, unfolded_inp, folded_inp_spat_shape):
        """
        Fold a vector unfolded with unfold_input.
        :param folded_inp_spat_shape: the spatial shape of the desired output
        """
        folded_inp = torch.nn.functional.fold(
            unfolded_inp, folded_inp_spat_shape, (self.kernel_height, self.kernel_width), dilation=self.dilation,
            padding=self.padding, stride=self.stride)
        return folded_inp

    def __repr__(self):
        return f"<Conv[{self.kernel_height}, {self.kernel_width}]: {self.in_features} -> {self.out_features}"


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

    def get_last_layer_bounds_means_trace(self, first_half_only_as_ub=False):
        """
        Get the evolution over time of the average of the last layer bounds.
        :param first_half_only_as_ub: assuming that the first half of the batches contains upper bounds, flip them and
            count only those in the average
        :return: list of singleton tensors
        """
        last_layer = sorted(self.bounds_progress_per_layer.keys())[-1]
        if first_half_only_as_ub:
            bounds_trace = [-bounds[:int(len(bounds) / 2)].mean() for bounds in
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
