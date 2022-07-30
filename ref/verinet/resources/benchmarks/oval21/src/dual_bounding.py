from torch import nn
from torch.nn import functional as F
import time, math
import torch
from src.utils import BatchLinearOp, BatchConvOp, get_relu_mask, compute_output_padding, create_final_coeffs_slice, \
    LinearOp, ConvOp, prod


class DualBounding:
    """
    Class implementing basic methods re-used by other dual solvers (e.g., creating the layer classes, sub-batching
    over bounds computations, etc).
    """

    def __init__(self, layers):
        self.layers = layers
        self.net = nn.Sequential(*layers)
        for param in self.net.parameters():
            param.requires_grad = False
        self.optimize = None
        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []

    @staticmethod
    def build_first_conditioned_layer(l_0, u_0, layer, no_conv=False):
        w_1 = layer.weight
        b_1 = layer.bias

        pos_w1 = torch.clamp(w_1, 0, None)
        neg_w1 = torch.clamp(w_1, None, 0)

        if isinstance(layer, nn.Linear):

            l_1 = l_0.view(l_0.shape[0], -1) @ pos_w1.t() + u_0.view(u_0.shape[0], -1) @ neg_w1.t() + b_1
            u_1 = u_0.view(u_0.shape[0], -1) @ pos_w1.t() + l_0.view(l_0.shape[0], -1) @ neg_w1.t() + b_1

            # Build the "conditioned" first layer
            range_0 = (u_0 - l_0)

            # range_1 = (u_1 - l_1)
            # cond_w_1 = (1/range_1).unsqueeze(1) * w_1 * range_0
            # cond_b_1 = (1/range_1) * (2 * b_1 - (u_1 + l_1) + w_1 @ (u_0 + l_0))
            cond_w_1 = w_1.unsqueeze(0) * 0.5 * range_0.unsqueeze(1)
            b0_sum = (u_0 + l_0)
            cond_b_1 = b_1 + 0.5 * b0_sum.view(b0_sum.shape[0], -1) @ w_1.t()

            cond_layer = BatchLinearOp(cond_w_1, cond_b_1, b_1)
        elif isinstance(layer, nn.Conv2d):
            l_1 = (F.conv2d(l_0, pos_w1, b_1, layer.stride, layer.padding, layer.dilation, layer.groups)
                   + F.conv2d(u_0, neg_w1, None,
                              layer.stride, layer.padding,
                              layer.dilation, layer.groups))
            u_1 = (F.conv2d(u_0, pos_w1, b_1,
                            layer.stride, layer.padding,
                            layer.dilation, layer.groups)
                   + F.conv2d(l_0, neg_w1, None,
                              layer.stride, layer.padding,
                              layer.dilation, layer.groups))

            range_0 = (u_0 - l_0) / 2
            out_bias = F.conv2d((u_0 + l_0) / 2, w_1, b_1,
                                layer.stride, layer.padding,
                                layer.dilation, layer.groups)

            output_padding = compute_output_padding(l_0, layer)  # can comment this to recover old behaviour

            cond_layer = BatchConvOp(w_1, out_bias, b_1,
                                     layer.stride, layer.padding,
                                     layer.dilation, layer.groups, output_padding)
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

            output_padding = compute_output_padding(prev_ub, layer)  # can comment this to recover old behaviour
            obj_layer = ConvOp(w_kp1, b_kp1,
                               layer.stride, layer.padding,
                               layer.dilation, layer.groups, output_padding)
            if no_conv:
                obj_layer_orig = obj_layer
                obj_layer = obj_layer.equivalent_linear(orig_shape_prev_ub)
        else:
            obj_layer = LinearOp(w_kp1, b_kp1)

        if isinstance(obj_layer, LinearOp) and (prev_ub.dim() > 2):
            # This is the first LinearOp,
            # We need to include the flattening
            obj_layer.flatten_from(prev_ub.shape[1:])

        return obj_layer, obj_layer_orig

    def compute_lower_bound(self, node=(-1, None), upper_bound=False, counterexample_verification=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the (list of, one per self.lower_bounds.shape[0]) list of
              gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        '''
        additional_coeffs = {}
        current_lbs = self.lower_bounds[node[0]].clone()
        current_ubs = self.upper_bounds[node[0]].clone()
        if current_lbs.dim() == 0:
            current_lbs = current_lbs.unsqueeze(0)
        node_layer_shape = current_lbs.shape[1:]
        batch_size = current_lbs.shape[0]
        self.opt_time_per_layer = []

        lay_to_opt = len(self.lower_bounds) + node[0] if node[0] < 0 else node[0]
        is_full_batch = (node[1] is None)
        is_part_batch = type(node[1]) is list or torch.is_tensor(node[1])
        # with batchification, we need to optimize over all layers in any case, as otherwise the tensors of
        # different sizes should be kept as a list (slow)
        # Optimize all the bounds
        nb_out = prod(node_layer_shape)

        start_opt_time = time.time()
        # if the resulting batch size from parallelizing over the output neurons boundings is too large, we need
        # to divide into sub-batches
        if is_full_batch:
            neuron_batch_size = nb_out * 2
        else:
            if is_part_batch:
                if type(node[1]) is list:
                    assert type(node[1][0][0]) is int, "For conv neurons, provide list of indices in linearised space"
                    tens_node1 = torch.tensor([clist for clist in node[1]], device=current_lbs.device, dtype=torch.long)
                    node = (node[0], tens_node1)
                fixed_len = node[1].shape[1]
                for clist in node[1]:
                    assert len(clist) == fixed_len, "Number of neurons for each batch must be equal"
                neuron_batch_size = fixed_len * 2
            else:
                # Optimise over single neuron.
                neuron_batch_size = 1
        c_batch_size = int(math.floor(self.max_batch / batch_size))
        n_batches = int(math.ceil(neuron_batch_size / float(c_batch_size)))
        print(f"----------------> {c_batch_size} * {n_batches}; total {neuron_batch_size}*{batch_size}")
        bound = None
        for sub_batch_idx in range(n_batches):
            # compute intermediate bounds on sub-batch
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, neuron_batch_size)

            slice_coeffs = create_final_coeffs_slice(
                start_batch_index, end_batch_index, batch_size, nb_out, current_lbs, node_layer_shape, node,
                upper_bound=upper_bound)
            additional_coeffs[lay_to_opt] = slice_coeffs

            c_bound = self.optimize(self.weights, additional_coeffs, self.lower_bounds, self.upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)
        end_opt_time = time.time()
        self.additional_coeffs = additional_coeffs

        self.opt_time_per_layer.append(end_opt_time - start_opt_time)
        if is_full_batch:
            # Return lbs, ubs for all neurons of the given layer.
            opted_ubs = -bound[:, :nb_out]
            opted_lbs = bound[:, nb_out:]
            ubs = opted_ubs.view(batch_size, *node_layer_shape)
            lbs = opted_lbs.view(batch_size, *node_layer_shape)

            # this is a bit of a hack for use in the context of standard counter-example verification problems
            if counterexample_verification:
                # if the bounds are not actual lower/upper bounds, then the subdomain for counter-example verification
                # is infeasible
                if lay_to_opt == len(self.weights):
                    # signal infeasible domains with infinity at the last layer bounds
                    lbs = torch.where(lbs > ubs, float('inf') * torch.ones_like(lbs), lbs)
                    ubs = torch.where(lbs > ubs, float('inf') * torch.ones_like(ubs), ubs)
                # otherwise, ignore the problem: it will be caught by the last layer
                return lbs, ubs

            assert (ubs - lbs).min() >= 0, "Incompatible bounds"

            return lbs, ubs
        elif is_part_batch:
            # Return lbs, ubs for all neurons of the given layer: for neurons outside the node[1] list,
            # previous intermediate bounds are returned
            opted_ubs = -bound[:, :fixed_len]
            opted_lbs = bound[:, fixed_len:]
            if current_lbs.dim() > 2:
                current_ubs = current_ubs.view((batch_size, -1))
                current_lbs = current_lbs.view((batch_size, -1))
            current_ubs.scatter_(1, node[1], opted_ubs)
            current_lbs.scatter_(1, node[1], opted_lbs)
            current_ubs = current_ubs.view(batch_size, *node_layer_shape)
            current_lbs = current_lbs.view(batch_size, *node_layer_shape)
            return current_lbs, current_ubs
        else:
            # Return lb or ub (depending on upper_bound=False or True) for the selected neuron.
            if upper_bound:
                bound = -bound
            return bound

    def define_linear_approximation(self, input_domain, no_conv=False, override_numerical_errors=False):
        '''
        no_conv is an option to operate only on linear layers, by transforming all
        the convolutional layers into equivalent linear layers.
        '''

        # store which relus are ambiguous. 1=passing, 0=blocking, -1=ambiguous. Shape: dom_batch_size x layer_width
        self.relu_mask = []
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
                    l_0 = l_0.view(l_0.shape[0], -1)
                    u_0 = u_0.view(u_0.shape[0], -1)
                    l_1 = l_1.view(l_1.shape[0], -1)
                    u_1 = u_1.view(u_1.shape[0], -1)
                self.lower_bounds = [-torch.ones_like(l_0), l_1]
                self.upper_bounds = [torch.ones_like(u_0), u_1]
                weights = [cond_first_linear]
                self.relu_mask.append(get_relu_mask(l_1, u_1))

            elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                assert next_is_linear
                next_is_linear = False

                orig_shape_prev_ub = self.original_shape_ubs[-1] if no_conv else None
                obj_layer, obj_layer_orig = self.build_obj_layer(self.upper_bounds[-1], layer, no_conv,
                                                                 orig_shape_prev_ub=orig_shape_prev_ub)
                weights.append(obj_layer)
                layer_opt_start_time = time.time()
                l_kp1, u_kp1 = self.solve_problem(weights, self.lower_bounds, self.upper_bounds,
                                                  override_numerical_errors=override_numerical_errors)
                layer_opt_end_time = time.time()
                time_used = layer_opt_end_time - layer_opt_start_time
                print(f"{type(self)} Time used for layer {lay_idx}: {time_used}")
                self.opt_time_per_layer.append(layer_opt_end_time - layer_opt_start_time)

                if no_conv:
                    if isinstance(layer, nn.Conv2d):
                        self.original_shape_lbs.append(
                            l_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_lbs[-1].unsqueeze(1).shape)).
                                squeeze(1)
                        )
                        self.original_shape_ubs.append(
                            u_kp1.view(obj_layer_orig.get_output_shape(self.original_shape_ubs[-1].unsqueeze(1).shape)).
                                squeeze(1)
                        )
                    else:
                        self.original_shape_lbs.append(l_kp1)
                        self.original_shape_ubs.append(u_kp1)
                self.lower_bounds.append(l_kp1)
                self.upper_bounds.append(u_kp1)
                if lay_idx < (len(self.layers) - 1):
                    # the relu mask doesn't make sense on the final layer
                    self.relu_mask.append(get_relu_mask(l_kp1, u_kp1))
            elif isinstance(layer, nn.ReLU):
                assert not next_is_linear
                next_is_linear = True
            else:
                pass
        self.weights = weights

    def build_model_using_bounds(self, domain, intermediate_bounds, build_limit=None, no_conv=False):
        """
        Build the model from the provided intermediate bounds.
        If no_conv is true, convolutional layers are treated as their equivalent linear layers. In that case,
        provided intermediate bounds should retain the convolutional structure.
        build_limit instructs to stop building the model at the k-th layer.
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
            self.lower_bounds = [clbs for clbs in ref_lbs]
            self.upper_bounds = [cubs for cubs in ref_ubs]
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
            if build_limit is not None and lay_idx >= build_limit:
                break
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

    def solve_problem(self, weights, lower_bounds, upper_bounds, override_numerical_errors=False):
        '''
        Compute bounds on the last layer of the problem.
        With batchification, we need to optimize over all layers in any case, as otherwise the tensors of different
         sizes should be kept as a list (slow)
        '''
        ini_lbs, ini_ubs = weights[-1].interval_forward(torch.clamp(lower_bounds[-1], 0, None),
                                                        torch.clamp(upper_bounds[-1], 0, None))

        out_shape = ini_lbs.shape[1:]
        nb_out = prod(out_shape)
        batch_size = ini_lbs.shape[0]

        # if the resulting batch size from parallelizing over the output neurons boundings is too large, we need
        # to divide into sub-batches
        neuron_batch_size = nb_out * 2
        c_batch_size = int(math.floor(self.max_batch / batch_size))
        n_batches = int(math.ceil(neuron_batch_size / float(c_batch_size)))
        bound = None
        for sub_batch_idx in range(n_batches):
            # compute intermediate bounds on sub-batch
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, neuron_batch_size)

            subbatch_coeffs = create_final_coeffs_slice(
                start_batch_index, end_batch_index, batch_size, nb_out, ini_lbs, out_shape)
            additional_coeffs = {len(lower_bounds): subbatch_coeffs}
            c_bound = self.optimize(weights, additional_coeffs, lower_bounds, upper_bounds)
            bound = c_bound if bound is None else torch.cat([bound, c_bound], 1)

        ubs = -bound[:, :nb_out]
        lbs = bound[:, nb_out:]
        lbs = lbs.view(batch_size, *out_shape)
        ubs = ubs.view(batch_size, *out_shape)

        if not override_numerical_errors:
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"
        else:
            ubs = torch.where((ubs - lbs <= 0) & (ubs - lbs >= -1e-5), lbs + 1e-5, ubs)
            assert (ubs - lbs).min() >= 0, "Incompatible bounds"

        return lbs, ubs

    def unbuild(self):
        # Release memory by discarding the stored model information.
        self.lower_bounds = []
        self.upper_bounds = []
        self.weights = []
        if "bounds_primal" in self.__dict__ and self.bounds_primal is not None:
            self.bounds_primal = []
        if "children_init" in self.__dict__ and self.children_init is not None:
            self.children_init = []
        if "additional_coeffs" in self.__dict__ and self.additional_coeffs is not None:
            self.additional_coeffs = None

    def update_relu_mask(self):
        # update all the relu masks of the given network
        for x_idx in range(1, len(self.lower_bounds)-1):
            self.relu_mask[x_idx-1] = get_relu_mask(
                self.lower_bounds[x_idx], self.upper_bounds[x_idx])

    # BaB-related method to implement automatic no. of iters.
    def increase_iters(self, to_max=False):
        # Increase the number of iterations of the algorithm this class represents.
        if not to_max:
            self.set_iters(min(self.steps + self.step_increase, self.max_steps))
        else:
            self.set_iters(self.max_steps)
        return self.steps != self.max_steps

    # BaB-related method to implement automatic no. of iters.
    def decrease_iters(self):
        # Decrease the number of iterations of the algorithm this class represents.
        self.set_iters(max(self.steps - self.step_increase, self.min_steps))