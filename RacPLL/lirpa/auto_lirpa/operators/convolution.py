from torch.nn import functional as F

from auto_lirpa.utils import *
from .base import *

class BoundedConv(Bound):

    def __init__(self, input_name, name, ori_name, attr, inputs, output_index, options, device):
        super().__init__(input_name, name, ori_name, attr, inputs, output_index, options, device)

        self.stride = attr['strides']
        self.padding = [attr['pads'][0], attr['pads'][1]]
        self.dilation = attr['dilations']
        self.groups = attr['group']
        self.mode = options.get("conv_mode", "matrix")

        if len(inputs) == 3:
            self.has_bias = True
        else:
            self.has_bias = False

        self.to(device)


    @Bound.save_io_shape
    def forward(self, *x):
        # x[0]: input, x[1]: weight, x[2]: bias if self.has_bias
        bias = x[2] if self.has_bias else None
        output = F.conv2d(x[0], x[1], bias, self.stride, self.padding, self.dilation, self.groups)
        return output


    def interval_propagate(self, *v, C=None):
        # print('BoundedConv interval_propagate')
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        norm = Interval.get_perturbation(v[0])
        norm = norm[0]
        if Interval.use_relative_bounds(*v):
            raise NotImplementedError

        h_L, h_U = v[0]
        weight = v[1][0]
        bias = v[2][0] if self.has_bias else None

        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        elif norm > 0:
            norm, eps = Interval.get_perturbation(v[0])
            # L2 norm, h_U and h_L are the same.
            mid = h_U
            # TODO: padding
            deviation = torch.mul(weight, weight).sum((1, 2, 3)).sqrt() * eps
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else: # Here we calculate the L0 norm IBP bound using the bound proposed in [Certified Defenses for Adversarial Patches, ICLR 2020]
            norm, eps, ratio = Interval.get_perturbation(v[0])
            mid = h_U
            k = int(eps)
            weight_sum = torch.sum(weight.abs(), 1)
            deviation = torch.sum(torch.topk(weight_sum.view(weight_sum.shape[0], -1), k)[0], dim=1) * ratio

            if self.has_bias:
                center = F.conv2d(mid, weight, v[2][0], self.stride, self.padding, self.dilation, self.groups)
            else:
                center = F.conv2d(mid, weight, None, self.stride, self.padding, self.dilation, self.groups)

            ss = center.shape
            deviation = deviation.repeat(ss[2] * ss[3]).view(-1, ss[1]).t().view(ss[1], ss[2], ss[3])
        
        center = F.conv2d(mid, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        upper = center + deviation
        lower = center - deviation
        return lower, upper


    def bound_backward(self, last_lA, last_uA, *x):
        if self.is_input_perturbed(1):
            raise NotImplementedError("Weight perturbation for convolution layers has not been implmented.")

        print('[+] Bound backward from', self)

        lA_y = uA_y = lA_bias = uA_bias = None
        weight = x[1].lower
        print('\t', type(x[0]))
        print('\t', type(x[1]))

        print('\t last_lA', last_lA)

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            if type(last_A) is OneHotC:
                raise

            if type(last_A) == torch.Tensor:
                raise

            elif type(last_A) == Patches:
                print('\t - identity', last_lA.identity)
                if last_A.identity == 0:
                    if not self.relu_followed:
                        # The last_A.patches was not padded, so we need to pad them here.
                        # If this Conv layer is followed by a ReLU layer, then the padding was already handled there and there is no need to pad again.
                        raise
                    else:
                        patches = last_A.patches

                    if self.has_bias:
                        # bias is x[2] (lower and upper are the same), and has shape (c,).
                        # Patches either has [out_c, batch, out_h, out_w, c, h, w] or [unstable_size, batch, c, h, w].
                        sum_bias = torch.einsum('sb...chw,c->sb...', patches, x[2].lower)
                        # sum_bias has shape (out_c, batch, out_h, out_w) or (unstable_size, batch).
                    else:
                        sum_bias = 0

                    flattened_patches = patches.reshape(-1, patches.size(-3), patches.size(-2), patches.size(-1))
                    pieces = F.conv_transpose2d(flattened_patches, weight, stride=self.stride)
                    # New patch size: (out_c, batch, out_h, out_w, c, h, w) or (unstable_size, batch, c, h, w).
                    pieces = pieces.view(*patches.shape[:-3], pieces.size(-3), pieces.size(-2), pieces.size(-1))

                elif last_A.identity == 1:

                    # New patches have size [out_c, batch, out_h, out_w, c, h, w] if it is not sparse.
                    # New patches have size [unstable_size, batch, c, h, w] if it is sparse.
                    if last_A.unstable_idx is not None:
                        pieces = weight.view(weight.size(0), 1, weight.size(1), weight.size(2), weight.size(3))
                        # Select based on the output channel (out_h and out_w are irrelevant here).
                        pieces = pieces[last_A.unstable_idx[0]]
                        # Expand the batch dimnension.
                        pieces = pieces.expand(-1, last_A.shape[1], -1, -1, -1)
                        # Do the same for the bias.
                        sum_bias = x[2].lower[last_A.unstable_idx[0]].unsqueeze(-1)
                        # bias has shape (unstable_size, batch).
                        sum_bias = sum_bias.expand(-1, last_A.shape[1])
                    else:
                        pieces = weight.view(weight.size(0), 1, 1, 1, weight.size(1), weight.size(2), weight.size(3)).expand(-1, *last_A.shape[1:4], -1, -1, -1)
                        sum_bias = x[2].lower.view(-1, 1, 1, 1).expand(-1, *last_A.shape[1:4])
                else:
                    raise NotImplementedError()

                padding = last_A.padding if last_A is not None else (0, 0, 0, 0)  # (left, right, top, bottom)
                stride = last_A.stride if last_A is not None else 1
                print('\t - padding', padding)
                print('\t - stride', stride)

                if type(padding) == int:
                    padding = padding * self.stride[0] + self.padding[0]
                else:
                    padding = tuple(p * self.stride[0] + self.padding[0] for p in padding)
                stride *= self.stride[0]
                if pieces.shape[-1] > self.input_shape[-1]:  
                    # The patches is too large and from now on, we will use matrix mode instead of patches mode.
                    # This is our desired matrix: the input will be flattend to (batch_size, input_channel*input_x * input_y) and multiplies on this matrix.
                    # After multiplication, the desired output is (batch_size, out_channel*output_x*output_y).
                    # A_matrix has size (batch, out_c*out_h*out_w, in_c*in_h*in_w)
                    A_matrix = patches_to_matrix(pieces, self.input_shape[1:], stride, padding, last_A.output_shape, last_A.unstable_idx)
                    if isinstance(sum_bias, torch.Tensor) and last_A.unstable_idx is None:
                        sum_bias = sum_bias.transpose(0, 1)
                        sum_bias = sum_bias.reshape(sum_bias.size(0), -1).transpose(0,1)
                    A_matrix = A_matrix.transpose(0,1)  # Spec dimension at the front.
                    return A_matrix, sum_bias

                print('\t - unstable_idx', last_lA.unstable_idx)
                return Patches(pieces, stride, padding, pieces.shape, unstable_idx=last_A.unstable_idx, output_shape=last_A.output_shape), sum_bias

            else:
                raise NotImplementedError()




        lA_x, lbias = _bound_oneside(last_lA)
        uA_x, ubias = _bound_oneside(last_uA)
        return [(lA_x, uA_x), (lA_y, uA_y), (lA_bias, uA_bias)], lbias, ubias






