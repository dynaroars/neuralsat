import numpy as np
import torch

from .utils import *

class Perturbation:
    r"""
    Base class for a perturbation specification. 

    Examples: 

    * `PerturbationLpNorm`: Lp-norm (p>=1) perturbation.

    * `PerturbationL0Norm`: L0-norm perturbation.

    * `PerturbationSynonym`: Synonym substitution perturbation for NLP.
    """

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps
    
    def concretize(self, x, A, sign=-1, aux=None):
        r"""
        Concretize bounds according to the perturbation specification.

        Args:
            x (Tensor): Input before perturbation.

            A (Tensor) : A matrix from LiRPA computation.

            sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.

            aux (object, optional): Auxilary information for concretization.

        Returns:
            bound (Tensor): concretized bound with the shape equal to the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        r"""
        Initialize bounds before LiRPA computation.

        Args:
            x (Tensor): Input before perturbation.

            aux (object, optional): Auxilary information.

            forward (bool): It indicates whether forward mode LiRPA is involved. 

        Returns:
            bound (LinearBound): Initialized bounds.

            center (Tensor): Center of perturbation. It can simply be `x`, or some other value.

            aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary information.
        """

        raise NotImplementedError


class PerturbationLpNorm(Perturbation):
    
    "Perturbation constrained by the L_p norm."

    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, relative=False):
        self.eps = eps
        self.norm = norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.relative = relative

    def init(self, x, aux=None, forward=False):
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            # For other norms, we pass in the BoundedTensor objects directly.
            x_L = x
            x_U = x

        if self.relative:
            nominal = x
            lower_offset = torch.max(x_L - x - 1e-8, torch.ones_like(x_L) * (-self.eps))
            upper_offset = torch.min(x_U - x + 1e-8, torch.ones_like(x_U) * (self.eps))
        else:
            nominal = lower_offset = upper_offset = None   

        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U, nominal=nominal, lower_offset=lower_offset, upper_offset=upper_offset), x, None
        
        raise NotImplementedError


    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return 'PerturbationLpNorm(norm=inf, eps={})'.format(self.eps)
            else:
                return 'PerturbationLpNorm(norm=inf, eps={}, x_L={}, x_U={})'.format(self.eps, self.x_L, self.x_U)
        else:
            return 'PerturbationLpNorm(norm={}, eps={})'.format(self.norm, self.eps)


    """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
    def concretize(self, x, A, sign=-1, aux=None, extra_constr=None):
        if A is None:
            return None
        # If A is an identity matrix, we will handle specially.
        def concretize_matrix(A):
            nonlocal x
            if not isinstance(A, eyeC):
                # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                A = A.reshape(A.shape[0], A.shape[1], -1)

                if extra_constr is not None:
                    # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
                    # For intermediate layer neurons, A has *neuron_size specifications.
                    beta = extra_constr['beta']
                    beta = beta.view(beta.size(0), -1, beta.size(-1))
                    # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
                    beta_coeffs = extra_constr['coeffs']
                    beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
                    # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
                    beta_bias = extra_constr['bias']
                    # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
                    extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
                    # Merge beta into the bias term. Output has size (batch, spec).
                    extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)

            if self.norm == np.inf:
                # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
                x_ub = x_U.reshape(x_U.shape[0], -1, 1)
                x_lb = x_L.reshape(x_L.shape[0], -1, 1)
                # Find the uppwer and lower bound similarly to IBP.
                center = (x_ub + x_lb) / 2.0
                diff = (x_ub - x_lb) / 2.0
                if not isinstance(A, eyeC):
                    if extra_constr is not None:
                        # Extra linear and bias terms from constraints.
                        print(
                            f'A extra: {(sign * extra_A).abs().sum().item()}, b extra: {(sign * extra_bias).abs().sum().item()}')
                        A = A - sign * extra_A
                        bound = A.matmul(center) - sign * extra_bias.unsqueeze(-1) + sign * A.abs().matmul(diff)
                    else:
                        bound = A.matmul(center) + sign * A.abs().matmul(diff)
                else:
                    assert extra_constr is None
                    # A is an identity matrix. No need to do this matmul.
                    bound = center + sign * diff
            else:
                assert extra_constr is None
                x = x.reshape(x.shape[0], -1, 1)
                if not isinstance(A, eyeC):
                    # Find the upper and lower bounds via dual norm.
                    deviation = A.norm(self.dual_norm, -1) * self.eps
                    bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
                else:
                    # A is an identity matrix. Its norm is all 1.
                    bound = x + sign * self.eps
            bound = bound.squeeze(-1)
            return bound

        def concretize_patches(A):
            nonlocal x
            if self.norm == np.inf:
                # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U

                # Here we should not reshape
                # Find the uppwer and lower bound similarly to IBP.
                center = (x_U + x_L) / 2.0
                diff = (x_U - x_L) / 2.0
                if not A.identity == 1:
                    # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W] or [unstable_size, batch_size, in_c, H, W]. Here out_c is the spec dimension.
                    patches = A.patches

                    # unfold the input as [batch_size, out_h, out_w, in_c, H, W]
                    unfold_input = inplace_unfold(center, kernel_size=A.patches.shape[-2:], padding = A.padding, stride = A.stride)
                    if A.unstable_idx is not None:
                        # We need to add a out_c dimension and select from it.
                        unfold_input = unfold_input.unsqueeze(0).expand(A.output_shape[1], -1, -1, -1, -1, -1, -1)
                        # When A is sparse, the shape is [unstable_size, batch_size, in_c, H, W]. Here unfold_input will match this shape.
                        unfold_input = unfold_input[A.unstable_idx[0], :, A.unstable_idx[1], A.unstable_idx[2]]
                        # size of bound: [batch_size, unstable_size].
                        bound = torch.einsum('sbchw,sbchw->bs', unfold_input, patches)
                    else:
                        # size of bound: [batch_size, out_c, out_h, out_w].
                        bound = torch.einsum('bijchw,sbijchw->bsij', unfold_input, patches)

                    # unfold the diff as [batch_size, out_h, out_w, in_c, H, W]
                    unfold_diff = inplace_unfold(diff, kernel_size=A.patches.shape[-2:], padding = A.padding, stride = A.stride)
                    if A.unstable_idx is not None:
                        # We need to add a out_c dimension and select from it.
                        unfold_diff = unfold_diff.unsqueeze(0).expand(A.output_shape[1], -1, -1, -1, -1, -1, -1)
                        # When A is sparse, the shape is [unstable_size, batch_size, in_c, H, W]
                        unfold_diff = unfold_diff[A.unstable_idx[0], :, A.unstable_idx[1], A.unstable_idx[2]]
                        # size of diff: [batch_size, unstable_size].
                        bound_diff = torch.einsum('sbchw,sbchw->bs', unfold_diff, patches.abs())
                    else:
                        # size of diff: [batch_size, out_c, out_h, out_w].
                        bound_diff = torch.einsum('bijchw,sbijchw->bsij', unfold_diff, patches.abs())

                    if sign == 1:
                        bound += bound_diff
                    elif sign == -1:
                        bound -= bound_diff
                    else:
                        raise ValueError("Unsupported Sign")

                    # The extra bias term from beta term.
                    if extra_constr is not None:
                        bound += extra_constr
                else:
                    assert extra_constr is None
                    # A is an identity matrix. No need to do this matmul.
                    bound = center + sign * diff
                return bound
            else:  # Lp norm
                # x_L = x - self.eps if self.x_L is None else self.x_L
                # x_U = x + self.eps if self.x_U is None else self.x_U

                input_shape = x.shape
                if not A.identity:
                    # Find the upper and lower bounds via dual norm.
                    # matrix has shape (batch_size, out_c * out_h * out_w, input_c, input_h, input_w) or (batch_size, unstable_size, input_c, input_h, input_w)
                    matrix = patches_to_matrix(A.patches, input_shape, A.stride, A.padding, A.output_shape, A.unstable_idx)
                    # Note that we should avoid reshape the matrix. Due to padding, matrix cannot be reshaped without copying.
                    deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                    # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                    bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                    if A.unstable_idx is None:
                        # Reshape to (batch, out_c, out_h, out_w).
                        bound = bound.view(matrix.size(0), A.patches.size(0), A.patches.size(2), A.patches.size(3))
                else:
                    # A is an identity matrix. Its norm is all 1.
                    bound = x + sign * self.eps
                return bound

        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            return concretize_matrix(A)
        elif isinstance(A, Patches):
            return concretize_patches(A)
        elif isinstance(A, BoundList):
            for b in A.bound_list:
                if isinstance(b, eyeC) or isinstance(b, torch.Tensor):
                    pass
        else:
            raise NotImplementedError()