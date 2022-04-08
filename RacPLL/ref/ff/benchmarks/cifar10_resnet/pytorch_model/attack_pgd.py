############################################################
#            Basic PGD attack (for VNN Comp 2021)          #
#                                                          #
# Copyright (C) 2021  Huan Zhang (huan@huan-zhang.com)     #
# Copyright (C) 2021  PyTorch Developers (pytorch.org)     #
#                                                          #
# This program is licenced under the BSD 2-Clause License  #
############################################################


import math
import torch
from torch.optim import Optimizer

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, num_restarts,
        multi_targeted=True, num_classes=10, use_adam=True, lr_decay=0.98,
        lower_limit=0.0, upper_limit=1.0, normalize=lambda x: x):

    best_loss = torch.empty(y.shape[0], device=X.device).fill_(float("-inf"))
    best_delta = torch.zeros_like(X, device=X.device)

    if multi_targeted:
        # Add an extra dimension for targets. Shape is (batch, target, ...).
        input_shape = X.size()
        X = X.unsqueeze(1).expand(-1, num_classes - 1, *(-1,) * (X.ndim - 1))
        # Generate target label list for each example.
        E = torch.eye(num_classes, dtype=X.dtype, device=X.device)
        c = E.unsqueeze(0) - E[y].unsqueeze(1)
        # remove specifications to self.
        I = ~(y.unsqueeze(1) == torch.arange(num_classes, device=y.device).unsqueeze(0))
        # c has shape (batch, num_classes - 1, num_classes).
        c = c[I].view(input_shape[0], num_classes - 1, num_classes)
        full_y = y.unsqueeze(1).repeat(1, 9).view(-1)

    # This is the maximal/minimal delta values for each sample, each element.
    sample_lower_limit = torch.clamp(lower_limit - X, min=-epsilon)
    sample_upper_limit = torch.clamp(upper_limit - X, max= epsilon)

    for n in range(num_restarts):

        delta = torch.empty_like(X).uniform_(-epsilon, epsilon)
        delta = torch.max(torch.min(delta, sample_upper_limit), sample_lower_limit).requires_grad_()

        if use_adam:
            opt = AdamClipping(params=[delta], lr=alpha)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        for _ in range(attack_iters):
            inputs = normalize(X + delta)
            if not multi_targeted:
                output = model(inputs)
            else:
                output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], num_classes-1, num_classes)
            if not multi_targeted:
                # Untargeted attack on logit space.
                runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1)[0] + 0.01
                groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
                # Use the margin as the loss function.
                loss = (runnerup - groundtruth)
            else:
                loss = torch.einsum('ijk,ijk->ij', c, output)

            loss.sum().backward()

            with torch.no_grad():
                # Save the best loss so far.
                if not multi_targeted:
                    best_delta[loss >= best_loss] = delta[loss >= best_loss]
                else:
                    # Need to find which label causes the worst case margin.
                    # Keep the one with largest margin.
                    all_loss, indices = loss.max(1)
                    delta_targeted = delta.gather(dim=1, index=indices.view(-1,1,1,1,1).expand(-1,-1,*input_shape[1:])).squeeze(1)
                    best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
                best_loss = torch.max(best_loss, all_loss)

                # Optimizer step.
                if use_adam:
                    opt.step(clipping=True, lower_limit=sample_lower_limit, upper_limit=sample_upper_limit, sign=1)
                    opt.zero_grad(set_to_none=True)
                    scheduler.step()
                else:
                    d = delta + alpha * torch.sign(delta.grad)
                    d = torch.max(torch.min(d, sample_upper_limit), sample_lower_limit)
                    delta.copy_(d)
                    delta.grad = None

    return best_delta


class AdamClipping(Optimizer):
    r"""Implements Adam algorithm, with per-parameter gradient clipping.
    The function is from PyTorch source code.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def _clip_update(exp_avg : torch.Tensor, denom : torch.Tensor, step_size : float, clipping_step_eps : float, lower_limit : torch.Tensor, upper_limit : torch.Tensor, p : torch.Tensor):
        # Compute the Adam update.
        update = exp_avg / denom * step_size
        # update = p.grad
        # Linf norm, scale according to sign.
        scaled_update = torch.sign(update) * clipping_step_eps
        # Apply the update.
        d = p.data + scaled_update
        # Avoid out-of-boundary updates.
        d = torch.max(torch.min(d, upper_limit), lower_limit)
        p.copy_(d)

    @torch.no_grad()
    def step(self, clipping=None, lower_limit=None, upper_limit=None, sign=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Currently we only deal with 1 parameter group.
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                if clipping:
                    assert sign == 1  # gradient ascent for adversarial attacks.
                    self._clip_update(exp_avg, denom, step_size, step_size, lower_limit, upper_limit, p)
                else:
                    # No clipping. Original Adam update.
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
