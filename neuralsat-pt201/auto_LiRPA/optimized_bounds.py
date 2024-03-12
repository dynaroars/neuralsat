import time
import os
from collections import OrderedDict
from contextlib import ExitStack
from auto_LiRPA.operators.leaf import BoundInput

import torch
from torch import optim
from .beta_crown import print_optimized_beta
from .cuda_utils import double2float
from .utils import logger, reduction_sum, multi_spec_keep_func_all

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule

default_optimize_bound_args = {
    'enable_alpha_crown': True,  # Enable optimization of alpha.
    'enable_beta_crown': False,  # Enable beta split constraint.
    'iteration': 20,  # Number of alpha/beta optimization iterations.
    # Share some alpha variables to save memory at the cost of slightly
    # looser bounds.
    'use_shared_alpha': False,
    # Optimizer used for alpha and beta optimization.
    'optimizer': 'adam',
    # Save best results of alpha/beta/bounds during optimization.
    'keep_best': True,
    # Only optimize bounds of last layer during alpha/beta CROWN.
    'fix_interm_bounds': True,
    # Learning rate for the optimizable parameter alpha in alpha-CROWN.
    'lr_alpha': 0.5,
    # Learning rate for the optimizable parameter beta in beta-CROWN.
    'lr_beta': 0.05,
    # Initial alpha variables by calling CROWN once.
    'init_alpha': True,
    'lr_coeffs': 0.01,  # Learning rate for coeffs for refinement
    # Layers to be refined, separated by commas.
    # -1 means preactivation before last activation.
    'intermediate_refinement_layers': [-1],
    # When batch size is not 1, this reduction function is applied to
    # reduce the bounds into a scalar.
    'loss_reduction_func': reduction_sum,
    # Criteria function of early stop.
    'stop_criterion_func': lambda x: False,
    # Learning rate decay factor during bounds optimization.
    'lr_decay': 0.98,
    # Number of iterations that we will start considering early stop
    # if tracking no improvement.
    'early_stop_patience': 10,
    # Start to save optimized best bounds
    # when current_iteration > int(iteration*start_save_best)
    'start_save_best': 0.5,
    # Use double fp (float64) at the last iteration in alpha/beta CROWN.
    'use_float64_in_last_iteration': False,
    # Use the newly fixed loss function. By default, it is set to False
    # for compatibility with existing use cases.
    # Try to ensure that the parameters always match with the optimized bounds.
    'deterministic': False,
}


def opt_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_reuse()


def opt_no_reuse(self: 'BoundedModule'):
    for node in self.get_enabled_opt_act():
        node.opt_no_reuse()


def _set_alpha(optimizable_activations, parameters, alphas, lr):
    """Set best_alphas, alphas and parameters list."""
    for node in optimizable_activations:
        alphas.extend(list(node.alpha.values()))
        node.opt_start()
    # Alpha has shape (2, output_shape, batch_dim, node_shape)
    parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
    # best_alpha is a dictionary of dictionary. Each key is the alpha variable
    # for one activation layer, and each value is a dictionary contains all
    # activation layers after that layer as keys.
    best_alphas = OrderedDict()
    for m in optimizable_activations:
        best_alphas[m.name] = {}
        for alpha_m in m.alpha:
            best_alphas[m.name][alpha_m] = m.alpha[alpha_m].detach().clone()
            # We will directly replace the dictionary for each activation layer after
            # optimization, so the saved alpha might not have require_grad=True.
            m.alpha[alpha_m].requires_grad_()

    return best_alphas


def _set_gammas(nodes, parameters):
    """
    Adds gammas to parameters list
    """
    gammas = []
    for node in nodes:
        if hasattr(node, 'gammas'):
            gammas.append(node.gammas)
    gamma_lr = 0.1
    parameters.append({'params': gammas, 'lr': gamma_lr})

def _save_ret_first_time(bounds, fill_value, x, best_ret):
    """Save results at the first iteration to best_ret."""
    if bounds is not None:
        best_bounds = torch.full_like(
            bounds, fill_value=fill_value, device=x[0].device, dtype=x[0].dtype)
    else:
        best_bounds = None

    if bounds is not None:
        best_ret.append(bounds.detach().clone())
    else:
        best_ret.append(None)

    return best_bounds


def _to_float64(self: 'BoundedModule', C, x, aux_reference_bounds, interm_bounds):
    """
    Transfer variables to float64 only in the last iteration to help alleviate
    floating point error.
    """
    self.to(torch.float64)
    C = C.to(torch.float64)
    x = self._to(x, torch.float64)
    # best_intermediate_bounds is linked to aux_reference_bounds!
    # we only need call .to() for one of them
    self._to(aux_reference_bounds, torch.float64, inplace=True)
    interm_bounds = self._to(
        interm_bounds, torch.float64)

    return C, x, interm_bounds


def _to_default_dtype(self: 'BoundedModule', x, total_loss, full_ret, ret,
                      best_intermediate_bounds, return_A):
    """
    Switch back to default precision from float64 typically to adapt to
    afterwards operations.
    """
    total_loss = total_loss.to(torch.get_default_dtype())
    self.to(torch.get_default_dtype())
    x[0].to(torch.get_default_dtype())
    full_ret = list(full_ret)
    if isinstance(ret[0], torch.Tensor):
        # round down lower bound
        full_ret[0] = double2float(full_ret[0], 'down')
    if isinstance(ret[1], torch.Tensor):
        # round up upper bound
        full_ret[1] = double2float(full_ret[1], 'up')
    for _k, _v in best_intermediate_bounds.items():
        _v[0] = double2float(_v[0], 'down')
        _v[1] = double2float(_v[1], 'up')
        best_intermediate_bounds[_k] = _v
    if return_A:
        full_ret[2] = self._to(full_ret[2], torch.get_default_dtype())

    return total_loss, x, full_ret


def _get_idx_mask(idx, full_ret_bound, best_ret_bound, loss_reduction_func):
    """Get index for improved elements."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    if idx == 0:
        idx_mask = (loss_reduction_func(full_ret_bound)
                    > loss_reduction_func(best_ret_bound)).view(-1)
    else:
        idx_mask = (loss_reduction_func(full_ret_bound)
                    < loss_reduction_func(best_ret_bound)).view(-1)
    improved_idx = None
    if idx_mask.any():
        # we only pick up the results improved in a batch
        improved_idx = idx_mask.nonzero(as_tuple=True)[0]
    return idx_mask, improved_idx


def _update_best_ret(full_ret_bound, best_ret_bound, full_ret, best_ret,
                     need_update, loss_reduction_func, idx, deterministic=False):
    """Update best_ret_bound and best_ret by comparing with new results."""
    assert idx in [0, 1], (
        '0 means updating lower bound, 1 means updating upper bound')
    idx_mask, improved_idx = _get_idx_mask(
        idx, full_ret_bound, best_ret_bound, loss_reduction_func)

    if improved_idx is not None:
        need_update = True
        compare = torch.max if idx == 0 else torch.min
        if not deterministic:
            best_ret_bound[improved_idx] = compare(
                full_ret_bound[improved_idx], best_ret_bound[improved_idx])
        else:
            best_ret_bound[improved_idx] = full_ret_bound[improved_idx]
        if full_ret[idx] is not None:
            if not deterministic:
                best_ret[idx][improved_idx] = compare(
                    full_ret[idx][improved_idx],
                    best_ret[idx][improved_idx])
            else:
                best_ret[idx][improved_idx] = full_ret[idx][improved_idx]

    return best_ret_bound, best_ret, need_update, idx_mask, improved_idx


def _update_optimizable_activations(
        optimizable_activations, interm_bounds,
        fix_interm_bounds, best_intermediate_bounds,
        reference_idx, idx, alpha, best_alphas, deterministic):
    """
    Update bounds and alpha of optimizable_activations.
    """
    for node in optimizable_activations:
        # Update best intermediate layer bounds only when they are optimized.
        # If they are already fixed in interm_bounds, then do
        # nothing.
        if (interm_bounds is None
                or node.inputs[0].name not in interm_bounds
                or not fix_interm_bounds):
            if deterministic:
                best_intermediate_bounds[node.name][0][idx] = node.inputs[0].lower[reference_idx]
                best_intermediate_bounds[node.name][1][idx] = node.inputs[0].upper[reference_idx]
            else:
                best_intermediate_bounds[node.name][0][idx] = torch.max(
                    best_intermediate_bounds[node.name][0][idx],
                    node.inputs[0].lower[reference_idx])
                best_intermediate_bounds[node.name][1][idx] = torch.min(
                    best_intermediate_bounds[node.name][1][idx],
                    node.inputs[0].upper[reference_idx])
        if alpha:
            # Each alpha has shape (2, output_shape, batch, *shape) for act.
            # For other activation function this can be different.
            for alpha_m in node.alpha:
                best_alphas[node.name][alpha_m][:, :,
                    idx] = node.alpha[alpha_m][:, :, idx]


def update_best_beta(self: 'BoundedModule', enable_opt_interm_bounds, betas,
                     best_betas, idx):
    """
    Update best beta by given idx.
    """
    if enable_opt_interm_bounds and betas:
        for node in self.splittable_activations:
            for node_input in node.inputs:
                for key in node_input.sparse_betas.keys():
                    best_betas[node_input.name][key] = (
                        node_input.sparse_betas[key].val.detach().clone())
    else:
        for node in self.nodes_with_beta:
            best_betas[node.name][idx] = node.sparse_betas[0].val[idx]


def _get_optimized_bounds(
        self: 'BoundedModule', x=None, aux=None, C=None, IBP=False,
        forward=False, method='backward', bound_side='lower',
        reuse_ibp=False, return_A=False, average_A=False, final_node_name=None,
        interm_bounds=None, reference_bounds=None,
        aux_reference_bounds=None, needed_A_dict=None,
        decision_thresh=None):
    """
    Optimize CROWN lower/upper bounds by alpha and/or beta.
    """

    opts = self.bound_opts['optimize_bound_args']
    iteration = opts['iteration']
    beta = opts['enable_beta_crown']
    alpha = opts['enable_alpha_crown']
    opt_choice = opts['optimizer']
    keep_best = opts['keep_best']
    fix_interm_bounds = opts['fix_interm_bounds']
    loss_reduction_func = opts['loss_reduction_func']
    stop_criterion_func = opts['stop_criterion_func']
    use_float64_in_last_iteration = opts['use_float64_in_last_iteration']
    early_stop_patience = opts['early_stop_patience']
    start_save_best = opts['start_save_best']
    deterministic = opts['deterministic']
    enable_opt_interm_bounds = self.bound_opts.get(
        'enable_opt_interm_bounds', False)
    sparse_intermediate_bounds = self.bound_opts.get(
        'sparse_intermediate_bounds', False)
    verbosity = self.bound_opts['verbosity']

    if bound_side not in ['lower', 'upper']:
        raise ValueError(bound_side)
    bound_lower = bound_side == 'lower'
    bound_upper = bound_side == 'upper'

    assert alpha or beta, (
        'nothing to optimize, use compute bound instead!')

    if C is not None:
        self.final_shape = C.size()[:2]
        self.bound_opts.update({'final_shape': self.final_shape})
    if opts['init_alpha']:
        # TODO: this should set up aux_reference_bounds.
        self.init_alpha(x, share_alphas=opts['use_shared_alpha'],
                        method=method, c=C, final_node_name=final_node_name)

    optimizable_activations = self.get_enabled_opt_act()

    alphas, parameters = [], []
    dense_coeffs_mask = []
    if alpha:
        best_alphas = _set_alpha(
            optimizable_activations, parameters, alphas, opts['lr_alpha'])
    if beta:
        ret_set_beta = self.set_beta(
            enable_opt_interm_bounds=enable_opt_interm_bounds, parameters=parameters,
            lr_beta=opts['lr_beta'], dense_coeffs_mask=dense_coeffs_mask)
        betas, best_betas, coeffs, dense_coeffs_mask = ret_set_beta[:4]

    start = time.time()

    if isinstance(decision_thresh, torch.Tensor):
        if decision_thresh.dim() == 1:
            # add the spec dim to be aligned with compute_bounds return
            decision_thresh = decision_thresh.unsqueeze(-1)

    if opt_choice == 'adam':
        opt = optim.Adam(parameters)
    elif opt_choice == 'sgd':
        opt = optim.SGD(parameters, momentum=0.9)
    else:
        raise NotImplementedError(opt_choice)

    # Create a weight vector to scale learning rate.
    loss_weight = torch.ones(size=(x[0].size(0),), device=x[0].device)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, opts['lr_decay'])

    # best_intermediate_bounds is linked to aux_reference_bounds!
    best_intermediate_bounds = {}
    if (sparse_intermediate_bounds and aux_reference_bounds is None
            and reference_bounds is not None):
        aux_reference_bounds = {}
        for name, (lb, ub) in reference_bounds.items():
            aux_reference_bounds[name] = [
                lb.detach().clone(), ub.detach().clone()]
    if aux_reference_bounds is None:
        aux_reference_bounds = {}

    need_grad = True
    patience = 0
    for i in range(iteration):

        intermediate_constr = None

        if not fix_interm_bounds:
            # If we still optimize all intermediate neurons, we can use
            # interm_bounds as reference bounds.
            if reference_bounds is None:
                reference_bounds = {}
            if interm_bounds is not None:
                reference_bounds.update(interm_bounds)
            interm_bounds = {}

        if i == iteration - 1:
            # No grad update needed for the last iteration
            need_grad = False
            if (self.device == 'cuda'
                    and torch.get_default_dtype() == torch.float32
                    and use_float64_in_last_iteration):
                C, x, interm_bounds = self._to_float64(
                    C, x, aux_reference_bounds, interm_bounds)

        with torch.no_grad() if not need_grad else ExitStack():
            # ret is lb, ub or lb, ub, A_dict (if return_A is set to true)
            ret = self.compute_bounds(
                x, aux, C, method=method, IBP=IBP, forward=forward,
                bound_lower=bound_lower, bound_upper=bound_upper,
                reuse_ibp=reuse_ibp, return_A=return_A,
                final_node_name=final_node_name, average_A=average_A,
                # When intermediate bounds are recomputed, we must set it
                # to None
                interm_bounds=interm_bounds if fix_interm_bounds else None,
                # This is the currently tightest interval, which will be used to
                # pass split constraints when intermediate betas are used.
                reference_bounds=reference_bounds,
                # This is the interval used for checking for unstable neurons.
                aux_reference_bounds=aux_reference_bounds if sparse_intermediate_bounds else None,
                # These are intermediate layer beta variables and their
                # corresponding A matrices and biases.
                intermediate_constr=intermediate_constr,
                needed_A_dict=needed_A_dict,
                update_mask=None)
        ret_l, ret_u = ret[0], ret[1]

        if i == 0:
            # save results at the first iteration
            best_ret = []
            best_ret_l = _save_ret_first_time(
                ret[0], float('-inf'), x, best_ret)
            best_ret_u = _save_ret_first_time(
                ret[1], float('inf'), x, best_ret)
            ret_0 = ret[0].detach().clone() if bound_lower else ret[1].detach().clone()

            for node in optimizable_activations:
                new_intermediate = [node.inputs[0].lower.detach().clone(),
                                    node.inputs[0].upper.detach().clone()]
                best_intermediate_bounds[node.name] = new_intermediate
                if sparse_intermediate_bounds:
                    # Always using the best bounds so far as the reference
                    # bounds.
                    aux_reference_bounds[node.inputs[0].name] = new_intermediate

        l = ret_l
        # Reduction over the spec dimension.
        if ret_l is not None and ret_l.shape[1] != 1:
            l = loss_reduction_func(ret_l)
        u = ret_u
        if ret_u is not None and ret_u.shape[1] != 1:
            u = loss_reduction_func(ret_u)

        # full_l, full_ret_l and full_u, full_ret_u is used for update the best
        full_ret_l, full_ret_u = ret_l, ret_u
        full_l = l
        full_ret = ret

        stop_criterion = (stop_criterion_func(full_ret_l) if bound_lower else stop_criterion_func(-full_ret_u))

        loss_ = l if bound_lower else -u
        total_loss = -1 * loss_

        if type(stop_criterion) == bool:
            loss = total_loss.sum() * (not stop_criterion)
        else:
            assert total_loss.shape == stop_criterion.shape
            loss = (total_loss * stop_criterion.logical_not()).sum()
        # For logging, print the total sum. Otherwise the loss may appear
        # to be increasing as more examples are stopped.
        loss_sum = total_loss.sum()

        stop_criterion_final = isinstance(
            stop_criterion, torch.Tensor) and stop_criterion.all()

        if i == iteration - 1:
            best_ret = list(best_ret)
            if best_ret[0] is not None:
                best_ret[0] = best_ret[0].to(torch.get_default_dtype())
            if best_ret[1] is not None:
                best_ret[1] = best_ret[1].to(torch.get_default_dtype())

        if (i == iteration - 1 and self.device == 'cuda'
                and torch.get_default_dtype() == torch.float32
                and use_float64_in_last_iteration):
            total_loss, x, full_ret = self._to_default_dtype(
                x, total_loss, full_ret, ret, best_intermediate_bounds, return_A)

        with torch.no_grad():
            # for lb and ub, we update them in every iteration since updating
            # them is cheap
            need_update = False
            if keep_best:
                if best_ret_u is not None:
                    best_ret_u, best_ret, need_update, idx_mask, improved_idx = _update_best_ret(
                        full_ret_u, best_ret_u, full_ret, best_ret, need_update,
                        loss_reduction_func, idx=1, deterministic=deterministic)
                if best_ret_l is not None:
                    best_ret_l, best_ret, need_update, idx_mask, improved_idx = _update_best_ret(
                        full_ret_l, best_ret_l, full_ret, best_ret, need_update,
                        loss_reduction_func, idx=0, deterministic=deterministic)
            else:
                # Not saving the best, just keep the last iteration.
                if full_ret[0] is not None:
                    best_ret[0] = full_ret[0]
                if full_ret[1] is not None:
                    best_ret[1] = full_ret[1]
            if return_A:
                # FIXME: A should also be updated by idx.
                best_ret = [best_ret[0], best_ret[1], full_ret[2]]

            if need_update:
                patience = 0  # bounds improved, reset patience
            else:
                patience += 1

            # Save variables if this is the best iteration.
            # To save computational cost, we only check keep_best at the first
            # (in case divergence) and second half iterations
            # or before early stop by either stop_criterion or
            # early_stop_patience reached
            if (i < 1 or i > int(iteration * start_save_best) or deterministic
                    or stop_criterion_final or patience == early_stop_patience):

                # compare with the first iteration results and get improved indexes
                if bound_lower:
                    if deterministic:
                        idx = improved_idx
                    else:
                        idx_mask, idx = _get_idx_mask(
                            0, full_ret_l, ret_0, loss_reduction_func)
                    ret_0[idx] = full_ret_l[idx]
                else:
                    if deterministic:
                        idx = improved_idx
                    else:
                        idx_mask, idx = _get_idx_mask(
                            1, full_ret_u, ret_0, loss_reduction_func)
                    ret_0[idx] = full_ret_u[idx]

                if idx is not None:
                    # for update propose, we condition the idx to update only
                    # on domains preserved
                    reference_idx = idx

                    _update_optimizable_activations(
                        optimizable_activations, interm_bounds,
                        fix_interm_bounds, best_intermediate_bounds,
                        reference_idx, idx, alpha, best_alphas, deterministic)

                    if beta:
                        self.update_best_beta(enable_opt_interm_bounds, betas,
                                              best_betas, idx)


        if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
            print(f'****** iter [{i}]',
                  f'loss: {loss_sum.item()}, lr: {opt.param_groups[0]["lr"]}',
            )

        if stop_criterion_final:
            # print(f'\nall verified at {i}th iter')
            break

        if patience > early_stop_patience:
            logger.debug(
                f'Early stop at {i}th iter due to {early_stop_patience}'
                ' iterations no improvement!')
            break

        if i != iteration - 1 and not loss.requires_grad:
            assert i == 0, (i, iteration)
            print('[WARNING] No optimizable parameters found. Will skip optimiziation. '
                  'This happens e.g. if all optimizable layers are freezed or the '
                  'network has no optimizable layers.')
            break

        opt.zero_grad(set_to_none=True)

        if verbosity > 2:
            current_lr = [param_group['lr'] for param_group in opt.param_groups]
            print(f'*** iter [{i}]\n', f'loss: {loss.item()}',
                  total_loss.squeeze().detach().cpu().numpy(), 'lr: ',
                  current_lr)
            if beta:
                print_optimized_beta(optimizable_activations)
            if beta and i == 0 and verbosity > 2:
                breakpoint()

        if i != iteration - 1:
            # we do not need to update parameters in the last step since the
            # best result already obtained
            loss.backward()

            # All intermediate variables are not needed at this point.
            self._clear_and_set_new(None)
            if opt_choice == 'adam-autolr':
                opt.step(lr_scale=[loss_weight, loss_weight])
            else:
                opt.step()
                
            scheduler.step()

        if beta:
            for b in betas:
                b.data = (b >= 0) * b.data
            for dmi in range(len(dense_coeffs_mask)):
                # apply dense mask to the dense split coeffs matrix
                coeffs[dmi].data = (
                    dense_coeffs_mask[dmi].float() * coeffs[dmi].data)

        if alpha:
            for m in optimizable_activations:
                m.clip_alpha()

    if verbosity > 3:
        breakpoint()

    if keep_best:
        # Set all variables to their saved best values.
        with torch.no_grad():
            for idx, node in enumerate(optimizable_activations):
                if alpha:
                    # Assigns a new dictionary.
                    node.alpha = best_alphas[node.name]
                # Update best intermediate layer bounds only when they are
                # optimized. If they are already fixed in
                # interm_bounds, then do nothing.
                best_intermediate = best_intermediate_bounds[node.name]
                node.inputs[0].lower.data = best_intermediate[0].data
                node.inputs[0].upper.data = best_intermediate[1].data
            if beta:
                for node in self.nodes_with_beta:
                    assert getattr(node, 'sparse_betas', None) is not None
                    if enable_opt_interm_bounds:
                        for key in node.sparse_betas.keys():
                            node.sparse_betas[key].val.copy_(
                                best_betas[node.name][key])
                    else:
                        node.sparse_betas[0].val.copy_(best_betas[node.name])

    if interm_bounds is not None and not fix_interm_bounds:
        for l in self._modules.values():
            if (l.name in interm_bounds.keys()
                    and hasattr(l, 'lower')):
                l.lower = torch.max(l.lower, interm_bounds[l.name][0])
                l.upper = torch.min(l.upper, interm_bounds[l.name][1])
                infeasible_neurons = l.lower > l.upper
                if infeasible_neurons.any():
                    print(f'Infeasibility detected in layer {l.name}.',
                          infeasible_neurons.sum().item(),
                          infeasible_neurons.nonzero()[:, 0])

    if verbosity > 0:
        if best_ret_l is not None:
            # FIXME: unify the handling of l and u.
            print('best_l after optimization:', best_ret_l.sum().item())
            if beta:
                print('beta sum per layer:', [p.sum().item() for p in betas])
        print('alpha/beta optimization time:', time.time() - start)

    for node in optimizable_activations:
        node.opt_end()

    if os.environ.get('AUTOLIRPA_DEBUG_OPT', False):
        print()

    return best_ret


def init_alpha(self: 'BoundedModule', x, share_alphas=False, method='backward',
               c=None, bound_lower=True, bound_upper=True, final_node_name=None,
               interm_bounds=None, activation_opt_params=None,
               skip_bound_compute=False):
    self(*x) # Do a forward pass to set perturbed nodes
    final = (self.final_node() if final_node_name is None
             else self[final_node_name])
    self._set_used_nodes(final)

    optimizable_activations = self.get_enabled_opt_act()
    for node in optimizable_activations:
        # TODO(7/6/2023) In the future, we may need to enable alpha sharing
        # automatically by consider the size of all the optimizable nodes in the
        # graph. For now, only an adhoc check in MatMul is added.
        node._all_optimizable_activations = optimizable_activations

        # initialize the parameters
        node.opt_init()

    if (not skip_bound_compute or interm_bounds is None or
            activation_opt_params is None or not all(
                [act.name in activation_opt_params
                 for act in self.optimizable_activations])):
        skipped = False
        # if new interval is None, then CROWN interval is not present
        # in this case, we still need to redo a CROWN pass to initialize
        # lower/upper
        with torch.no_grad():
            # We temporarilly deactivate output constraints
            l, u = self.compute_bounds(
                x=x, C=c, method=method, bound_lower=bound_lower,
                bound_upper=bound_upper, final_node_name=final_node_name,
                interm_bounds=interm_bounds)
    else:
        # we skip, but we still would like to figure out the "used",
        # "perturbed", "backward_from" of each note in the graph
        skipped = True
        # this set the "perturbed" property
        self.set_input(*x, interm_bounds=interm_bounds)
        self.backward_from = {node: [final] for node in self._modules}

    final_node_name = final_node_name or self.final_name

    init_intermediate_bounds = {}
    for node in optimizable_activations:
        start_nodes = []
        if method in ['forward', 'forward+backward']:
            start_nodes.append(('_forward', 1, None, False))
        if method in ['backward', 'forward+backward']:
            backward_from_node = node
            start_nodes += self.get_alpha_crown_start_nodes(
                node,
                c=c,
                share_alphas=share_alphas,
                final_node_name=final_node_name,
                backward_from_node=backward_from_node
            )
        if skipped:
            node.restore_optimized_params(activation_opt_params[node.name])
        else:
            node.init_opt_parameters(start_nodes)
        if node in self.splittable_activations:
            for i in node.requires_input_bounds:
                input_node = node.inputs[i]
                if not input_node.perturbed:
                    continue
                init_intermediate_bounds[node.inputs[i].name] = (
                    [node.inputs[i].lower.detach(),
                    node.inputs[i].upper.detach()])

    if self.bound_opts['verbosity'] >= 1:
        print('Optimizable variables initialized.')
    if skip_bound_compute:
        return init_intermediate_bounds
    else:
        return l, u, init_intermediate_bounds



def get_refined_interm_bounds(self: 'BoundedModule'):
    interm_bounds = {
        node.name: [
            node.lower.to(self.device) if hasattr(node, 'lower') else None, 
            node.upper.to(self.device) if hasattr(node, 'upper') else None, 
        ] for node in self.split_nodes
    }

    if any([(None in _ ) for _ in interm_bounds.values()]):
        return None
    
    return interm_bounds
    
    