from torch.nn import functional as F
from collections import OrderedDict

from .base import *

class BoundedActivation(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.nonlinear = True
        # self.relaxed = False



class BoundedOptimizableActivation(BoundedActivation):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.opt_stage = None

    # Initialize bound optimization. Optimized bounds are not used in  this stage. 
    def opt_init(self):
        self.opt_stage = 'init'

    # Start optimizing bounds 
    def opt_start(self):
        self.opt_stage = 'opt'


class BoundedRelu(BoundedOptimizableActivation):
    
    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)
        self.flattened_nodes = None
        self.relu_options = options.get('relu', 'adaptive')
        self.patch_size = {}

        self.beta = self.beta_mask = self.masked_beta = self.sparse_beta = None
       
    @Bound.save_io_shape
    def forward(self, x):
        self.shape = x.shape[1:]
        if self.flattened_nodes is None:
            self.flattened_nodes = x[0].reshape(-1).shape[0]
        return F.relu(x)

    def interval_propagate(self, *v):
        if Interval.use_relative_bounds(*v):
            raise NotImplementedError

        h_L, h_U = v[0][0], v[0][1]
        return F.relu(h_L), F.relu(h_U)


    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, beta_for_intermediate_layers=False, unstable_idx=None):
        # print('[+] Bound backward from ======================================>', self)
        # print('\t- Input:', x)
        if x is not None:
            lb_r = x.lower.clamp(max=0)
            ub_r = x.upper.clamp(min=0)
        else:
            lb_r = self.lower.clamp(max=0)
            ub_r = self.upper.clamp(min=0)

        self.I = ((lb_r != 0) * (ub_r != 0)).detach()  # unstable neurons

        if hasattr(x, 'interval') and Interval.use_relative_bounds(x.interval):
            raise
        else:
            ub_r = torch.max(ub_r, lb_r + 1e-8)
            upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d

        flag_expand = False
        ub_lower_d = lb_lower_d = None

        # print('\t- relu_options:', self.relu_options)
        # print('\t- opt_stage:', self.opt_stage)
        if self.relu_options == "same-slope":
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.relu_options == "zero-lb":
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.relu_options == "one-lb":
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        elif self.relu_options == "reversed-adaptive":
            lower_d = (upper_d < 0.5).float()
        elif self.opt_stage == 'opt':
            # Alpha-CROWN.
            lower_d = None
            # Each alpha has shape (2, output_shape, batch_size, *relu_node_shape]. If slope is shared, output_shape will be 1.
            if unstable_idx is not None and self.alpha[start_node.name].size(1) != 1:
                raise
            else:
                selected_alpha = self.alpha[start_node.name]

            if x is not None:
                lower = x.lower
                upper = x.upper
            else:
                lower = self.lower
                upper = self.upper
            lower_mask = lower > 0
            upper_mask = upper < 0
            if last_lA is not None:
                lb_lower_d = selected_alpha[0].clamp(min=0.0, max=1.0)
                lb_lower_d[:, lower_mask] = 1.0
                lb_lower_d[:, upper_mask] = 0.0
            if last_uA is not None:
                ub_lower_d = selected_alpha[1].clamp(min=0.0, max=1.0)
                ub_lower_d[:, lower_mask] = 1.0
                ub_lower_d[:, upper_mask] = 0.0
            self.zero_backward_coeffs_l = self.zero_backward_coeffs_u = upper_mask.all().item()
            flag_expand = True

        else:
            # adaptive
            lower_d = (upper_d > 0.5).float()


        # save for calculate babsr score
        self.d = upper_d
        self.lA = last_lA
        # Save for initialization bounds.
        self.lower_d = lower_d

        # Upper bound always needs an extra specification dimension, since they only depend on lb and ub.
        upper_d = upper_d.unsqueeze(0)
        upper_b = upper_b.unsqueeze(0)

        if not flag_expand:
            if self.opt_stage == 'opt':
                # We have different slopes for lower and upper bounds propagation.
                lb_lower_d = lb_lower_d.unsqueeze(0) if last_lA is not None else None
                ub_lower_d = ub_lower_d.unsqueeze(0) if last_uA is not None else None
            else:
                lower_d = lower_d.unsqueeze(0)


        mode = "patches" if isinstance(last_lA, Patches) or isinstance(last_uA, Patches) else "matrix"

        # In patches mode, we need to unfold lower and upper slopes. In matrix mode we simply return.
        def _maybe_unfold(d_tensor, last_A):
            if mode == "matrix" or d_tensor is None or last_A is None:
                return d_tensor
            # Input are slopes with shape (spec, batch, input_c, input_h, input_w)
            # Here spec is the same as out_c.
            assert d_tensor.ndim == 5
            d_shape = d_tensor.size()
            # Reshape to 4-D tensor to unfold.
            d_tensor = d_tensor.view(-1, *d_shape[-3:])
            # unfold the slope matrix as patches. Patch shape is [spec * batch, out_h, out_w, in_c, H, W).
            d_unfolded = inplace_unfold(d_tensor, kernel_size=last_A.patches.shape[-2:], stride=last_A.stride, padding=last_A.padding)
            # Reshape to (spec, batch, out_h, out_w, in_c, H, W); here spec_size is out_c.
            d_unfolded_r = d_unfolded.view(*d_shape[:-3], *d_unfolded.shape[1:])
            if last_A.unstable_idx is not None:
                if d_unfolded_r.size(0) == 1:
                    # Broadcast the spec shape, so only need to select the reset dimensions.
                    # Change shape to (out_h, out_w, batch, in_c, H, W) or (out_h, out_w, in_c, H, W).
                    d_unfolded_r = d_unfolded_r.squeeze(0).permute(1, 2, 0, 3, 4, 5)
                    d_unfolded_r = d_unfolded_r[last_A.unstable_idx[1], last_A.unstable_idx[2]]
                    # output shape: (unstable_size, batch, in_c, H, W).
                else:
                    d_unfolded_r = d_unfolded_r[last_A.unstable_idx[0], :, last_A.unstable_idx[1], last_A.unstable_idx[2]]
                # For sparse patches, the shape after unfold is (unstable_size, batch_size, in_c, H, W).
            # For regular patches, the shape after unfold is (spec, batch, out_h, out_w, in_c, H, W).
            return d_unfolded_r

        # Choose upper or lower bounds based on the sign of last_A
        def _bound_oneside(last_A, d_pos, d_neg, b_pos, b_neg):
            if last_A is None:
                return None, 0

            if type(last_A) == torch.Tensor:
                # multiply according to sign of A (we use fused operation to save memory)
                # neg_A = last_A.clamp(max=0)
                # pos_A = last_A.clamp(min=0)
                # A = d_pos * pos_A + d_neg * neg_A
                A, pos_A, neg_A = self.clamp_mutiply(last_A, d_pos, d_neg) # affected by torchscript
                bias = 0
                if b_pos is not None:
                    bias = bias + torch.einsum('sb...,sb...->sb', pos_A, b_pos)
                if b_neg is not None:
                    bias = bias + torch.einsum('sb...,sb...->sb', neg_A, b_neg)
                return A, bias
            elif type(last_A) == Patches:
                # if last_A is not an identity matrix
                assert last_A.identity == 0
                if last_A.identity == 0:
                    # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W]. Here out_c is the spec dimension.
                    # or (unstable_size, batch_size, in_c, H, W) when it is sparse.
                    patches = last_A.patches
                    prod, pos_A_patches, neg_A_patches = self.clamp_mutiply_non_contiguous(patches, d_pos, d_neg)
                    # prod has shape [out_c, batch_size, out_h, out_w, in_c, H, W] or (unstable_size, batch_size, in_c, H, W) when it is sparse.

                    # Save the patch size, which will be used in init_slope() to determine the number of optimizable parameters.
                    if start_node is not None:
                        if last_A.unstable_idx is not None:
                            # Sparse patches, we need to construct the full patch size: (out_c, batch, out_h, out_w, c, h, w).
                            self.patch_size[start_node.name] = [last_A.output_shape[1], prod.size(1), last_A.output_shape[2], last_A.output_shape[3], prod.size(-3), prod.size(-2), prod.size(-1)]
                        else:
                            # Regular patches.
                            self.patch_size[start_node.name] = prod.size()

                    bias = 0
                    if b_pos is not None:
                        # For sparse patches the return bias size is (unstable_size, batch).
                        # For regular patches the return bias size is (spec, batch, out_h, out_w).
                        bias = bias + torch.einsum('sb...chw,sb...chw->sb...', b_pos, pos_A_patches)
                    if b_neg is not None:
                        bias = bias + torch.einsum('sb...chw,sb...chw->sb...', b_neg, neg_A_patches)
                    return Patches(prod, last_A.stride, last_A.padding, prod.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape), bias


        # In patches mode we might need an unfold.
        upper_d = _maybe_unfold(upper_d, last_lA if last_lA is not None else last_uA)
        lower_d = _maybe_unfold(lower_d, last_lA if last_lA is not None else last_uA)
        upper_b = _maybe_unfold(upper_b, last_lA if last_lA is not None else last_uA)
        ub_lower_d = _maybe_unfold(ub_lower_d, last_uA)
        lb_lower_d = _maybe_unfold(lb_lower_d, last_lA)


        uA, ubias = _bound_oneside(last_uA, upper_d, ub_lower_d if lower_d is None else lower_d, upper_b, None)
        lA, lbias = _bound_oneside(last_lA, lb_lower_d if lower_d is None else lower_d, upper_d, None, upper_b)

        self.masked_beta_lower = self.masked_beta_upper = None
        if self.options.get('optimize_bound_args', {}).get('ob_beta', False):
            raise
        return [(lA, uA)], lbias, ubias


    def init_opt_parameters(self, start_nodes):
        self.alpha = OrderedDict()
        ref = self.inputs[0].lower # a reference variable for getting the shape
        for ns, size_s in start_nodes:
            self.alpha[ns] = torch.empty([2, size_s, ref.size(0), *self.shape], dtype=torch.float, device=ref.device, requires_grad=True)
        for k, v in self.alpha.items():
            v.data.copy_(self.lower_d.data)  # Initial from adaptive lower bounds.    
