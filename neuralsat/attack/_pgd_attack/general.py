import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time

from .util import *

def attack(model, x, data_min, data_max, list_target_label_arrays, initialization="uniform", GAMA_loss=False, attack_iters=100, num_restarts=10):
    r""" Interface to PGD attack.

    Args:
        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).
        [batch_size, *x_shape]

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)
        shape: [batch_size, spec_num, *input_shape]

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)
        shape: [batch_size, spec_num, *input_shape]

        list_target_label_arrays: a list of list of tuples:
                We have N examples, and list_target_label_arrays is a list containing N lists.
                Each inner list contains the target_label_array for an example:
                    [(prop_mat_1, prop_rhs_1), (prop_mat_2, prop_rhs_2), ..., (prop_mat_n, prop_rhs_n)]
                    prop_mat is a numpy array with shape [num_and, num_output], prop_rhs is a numpy array with shape [num_and]

        initialization (string): initialization of PGD attack, chosen from 'uniform' and 'osi'

        GAMA_loss (boolean): whether to use GAMA (Guided adversarial attack) loss in PGD attack
    """
    # attack_start_time = time.time()
    # assert arguments.Config["specification"]["norm"] == np.inf, print('We only support Linf-norm attack.')
    max_eps = torch.max(data_max - data_min).item() / 2

    device = x.device

    # if arguments.Config["attack"]["pgd_alpha"] == 'auto':
        # alpha = max_eps/4
    # else:
    #     alpha = float(arguments.Config["attack"]["pgd_alpha"])
    alpha = max_eps / 4

    # print(f'Attack parameters: initialization={initialization}, steps={arguments.Config["attack"]["pgd_steps"]}, restarts={arguments.Config["attack"]["pgd_restarts"]}, alpha={alpha}, initialization={initialization}, GAMA={GAMA_loss}')

    # Set all parameters without gradient, this can speedup things significantly.
    grad_status = {}
    for p in model.parameters():
        grad_status[p] = p.requires_grad
        p.requires_grad_(False)

    output = model(x).detach()
    # print('Model output of first 5 examples:\n', output[0, :5], output.shape)
    
    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)


    output = output.unsqueeze(1).unsqueeze(1).repeat(1, 1, len(cond_mat[0]), 1)
    # print('Model output of first 5 examples:\n', output[0, :5], output.shape)

    if test_conditions(x, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max.unsqueeze(1), data_min.unsqueeze(1)).all():
        # print("Clean prediction incorrect, attack skipped.")
        # Obtain attack margin.
        attack_image, _, attack_margin = gen_adv_example(model, x, torch.zeros_like(x), data_max, data_min, C_mat, rhs_mat, cond_mat)
        return True, attack_image.detach(), attack_margin.detach(), None


    data_min = data_min.to(device)
    data_max = data_max.to(device)
    rhs_mat = rhs_mat.to(device)
    C_mat = C_mat.to(device)
    best_deltas, last_deltas = attack_with_general_specs(model, 
                                                         x, 
                                                         data_min, 
                                                         data_max, 
                                                         C_mat, 
                                                         rhs_mat, 
                                                         cond_mat, 
                                                         same_number_const, 
                                                         alpha, 
                                                         attack_iters=attack_iters, 
                                                         num_restarts=num_restarts, 
                                                         initialization=initialization, 
                                                         GAMA_loss=GAMA_loss)

    attack_image, attack_output, attack_margin = gen_adv_example(model, 
                                                                 x, 
                                                                 best_deltas, 
                                                                 data_max, 
                                                                 data_min, 
                                                                 C_mat, 
                                                                 rhs_mat, 
                                                                 cond_mat)

    # Adversarial images/candidates in all restarts and targets. Useful for BaB-attack.
    # last_deltas has shape [batch, num_restarts, specs, c, h, w]. Need the extra num_restarts and specs dim.
    # x has shape [batch, c, h, w] and data_min/data_max has shape [batch, num_specs, c, h, w].
    all_adv_candidates = torch.max(
            torch.min(x.unsqueeze(1).unsqueeze(1) + last_deltas, 
                data_max.unsqueeze(1)), data_min.unsqueeze(1))

    # Go back to original requires_grad status.
    for p in model.parameters():
        p.requires_grad_(grad_status[p])

    # attack_time = time.time() - attack_start_time
    # print(f'Attack finished in {attack_time:.4f} seconds.')
    if test_conditions(attack_image.unsqueeze(1), attack_output.unsqueeze(1), C_mat, rhs_mat, cond_mat, same_number_const, data_max, data_min).all():
        # print("PGD attack succeeded!")
        return True, attack_image.detach(), attack_margin.detach(), all_adv_candidates
    else:
        # print("PGD attack failed")
        return False, attack_image.detach(), attack_margin.detach(), all_adv_candidates



def attack_with_general_specs(model, X, data_min, data_max, C_mat, rhs_mat, cond_mat, same_number_const, alpha, 
                              use_adam=True, normalize=lambda x: x, initialization='uniform', GAMA_loss=False, 
                              num_restarts=10, attack_iters=100, only_replicate_restarts=False):

    r''' the functional function for pgd attack

    Args:
        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)

        C_mat (torch.tensor): [num_example, num_spec, num_output]

        rhs_mat (torch.tensor): [num_example, num_spec]

        cond_mat (list): [[] * num_example] mark the group of conditions

        same_number_const (bool): if same_number_const is True, it means that there are same number of and specifications in every or specification group.

        alpha (float): alpha for pgd attack
    '''
    device = X.device
    
    lr_decay = 0.99
    early_stop = True
    gama_lambda = 10

    if only_replicate_restarts:
        input_shape = (X.shape[0], *X.shape[2:])
    else:
        input_shape = X.size()
    num_classes = C_mat.shape[-1]

    
    num_or_spec = len(cond_mat[0])

    extra_dim = (num_restarts, num_or_spec) if only_replicate_restarts == False else (num_restarts,)
    # shape of x: [num_example, *shape_of_x]

    best_loss = torch.empty(X.size(0), device=device).fill_(float("-inf"))
    best_delta = torch.zeros(input_shape, device=device)

    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)
    # [1, 1, num_spec, *input_shape]

    X_ndim = X.ndim

    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X

    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim - 1))
    extra_dim = (X.shape[1], X.shape[2])

    if initialization == 'osi':
        # X_init = OSI_init(model, X, y, epsilon, alpha, num_classes, iter_steps=attack_iters, extra_dim=extra_dim, upper_limit=upper_limit, lower_limit=lower_limit)
        # osi_start_time = time.time()
        X_init = OSI_init_C(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
        # osi_time = time.time() - osi_start_time
        # print(f'diversed PGD initialization time: {osi_time:.4f}')


    if initialization == 'osi':
        delta = (X_init - X).detach().requires_grad_()
    elif initialization == 'uniform':
        delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
    elif initialization == 'none':
        delta = torch.zeros_like(X).requires_grad_()
    else:
        raise ValueError(f"Unknown initialization method {initialization}")

    if use_adam:
        opt = AdamClipping(params=[delta], lr=alpha)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    for _ in range(attack_iters):
        inputs = normalize(X + delta)
        output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)

        if GAMA_loss:
            # Output on original model is needed if gama loss is used.
            origin_out = torch.softmax(model(normalize(X.reshape(-1, *input_shape[1:]))), 1)
            origin_out = origin_out.view(output.shape)
        else:
            origin_out = None

        loss, loss_gama = build_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const, gama_lambda if GAMA_loss else 0.0, mode='hinge')
        gama_lambda *= 0.9
        # shape of loss: [num_example, num_restarts, num_or_spec]
        # or float when gama_lambda > 0

        loss_gama.sum().backward()

        with torch.no_grad():
            # Save the best loss so far.
            if same_number_const:
                loss = loss.amin(-1)
                # loss has shape [num_example, num_restarts, num_or_spec].
                # margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
            else:
                group_C = torch.zeros(len(cond_mat[0]), C_mat.shape[1]).to(loss.device) # [num_or_spec, num_total_spec]
                x_index = []
                y_index = []
                index = 0
                for i, cond in enumerate(cond_mat[0]):
                    for _ in range(cond):
                        x_index.append(i)
                        y_index.append(index)
                        index += 1
                group_C[x_index, y_index] = 1.0

                # loss shape: [batch_size, num_restarts, num_total_spec]
                loss = group_C.matmul(loss.unsqueeze(-1)).squeeze(-1)
                # loss shape: [batch_size, num_restarts, num_or_spec]
            
            loss = loss.view(loss.shape[0], -1)
            # all_loss and indices have shape (batch, ), and this is the best loss over all restarts and number of classes.
            all_loss, indices = loss.max(1)
            # delta has shape (batch, restarts, num_class-1, c, h, w). For each batch element, we want to select from the best over (restarts, num_classes-1) dimension.
            # delta_targeted has shape (batch, c, h, w).
            delta_targeted = delta.view(delta.size(0), -1, *input_shape[1:]).gather(dim=1, index=indices.view(-1,1,*(1,) * (len(input_shape) - 1)).expand(-1,-1,*input_shape[1:])).squeeze(1)

            best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
            best_loss = torch.max(best_loss, all_loss)
        
        if early_stop:
            if test_conditions(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max, data_min).all():
                # print("pgd early stop")
                break

        if use_adam:
            opt.step(clipping=True, lower_limit=delta_lower_limit, upper_limit=delta_upper_limit, sign=1)
            opt.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            d = delta + alpha * torch.sign(delta.grad)
            d = torch.max(torch.min(d, delta_upper_limit), delta_lower_limit)
            delta.copy_(d)
            delta.grad = None

    return best_delta, delta
    



def _pgd_whitebox(model, X, constraints, specLB, specUB, device, 
                  num_steps=50, step_size=0.2, ODI_num_steps=10, ODI_step_size=1.0, 
                  batch_size=50, lossFunc="margin", restarts=1, stop_early=True):
    out_X = model(X).detach()
    adex = None
    worst_x = None
    best_loss = torch.tensor(-np.inf)

    y = translate_constraints_to_label([constraints])[0]

    for _ in range(restarts):
        if adex is not None:
            break
        X_pgd = torch.autograd.Variable(X.data.repeat((batch_size,) + (1,) * (X.dim() - 1)), requires_grad=True).to(device)
        randVector_ = torch.ones_like(model(X_pgd)).uniform_(-1, 1)
        random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5) * (specUB - specLB)
        X_pgd = torch.autograd.Variable(torch.minimum(torch.maximum(X_pgd.data + random_noise, specLB), specUB), requires_grad=True,)

        lr_scale = (specUB - specLB) / 2
        lr_scheduler = step_lr_scheduler(
            step_size, 
            gamma=0.1,
            interval=[
                np.ceil(0.5 * num_steps),
                np.ceil(0.8 * num_steps),
                np.ceil(0.9 * num_steps),
            ],
        )
        gama_lambda = 10
        y_target = constraints[0][0][-1]
        regression = (
            len(constraints) == 2
            and ([(-1, 0, y_target)] in constraints)
            and ([(0, -1, y_target)] in constraints)
        )

        for i in range(ODI_num_steps + num_steps + 1):
            # print('restart', _, 'iter', i)
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                out = model(X_pgd)

                cstrs_hold = evaluate_cstr(constraints, out.detach(), torch_input=True)
                if (not regression) and (not cstrs_hold.all()) and (stop_early):
                    adv_idx = (~cstrs_hold.cpu()).nonzero(as_tuple=False)[0].item()
                    adex = X_pgd[adv_idx : adv_idx + 1]
                    assert not evaluate_cstr(constraints, model(adex), torch_input=True)[0], f"{model(adex)},{constraints}"
                    return [adex.detach().cpu().numpy()], None
                if i == ODI_num_steps + num_steps:
                    adex = None
                    break

                if i < ODI_num_steps:
                    loss = (out * randVector_).sum()
                elif lossFunc == "xent":
                    loss = nn.CrossEntropyLoss()(out, torch.tensor([y] * out.shape[0], dtype=torch.long))
                elif lossFunc == "margin":
                    and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size / len(constraints)))
                    and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size - len(and_idx))], axis=0)).to(device)
                    loss = constraint_loss(out, constraints, and_idx=and_idx).sum()
                elif lossFunc == "GAMA":
                    and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size / len(constraints)))
                    and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size - len(and_idx))], axis=0)).to(device)
                    out = torch.softmax(out, 1)
                    loss = (constraint_loss(out, constraints, and_idx=and_idx) + (gama_lambda * (out_X - out) ** 2).sum(dim=1)).sum()
                    gama_lambda *= 0.9

            max_loss = torch.max(loss).item()
            if max_loss > best_loss:
                best_loss = max_loss
                worst_x = X_pgd[torch.argmax(loss)].detach().cpu().numpy()

            loss.backward()
            if i < ODI_num_steps:
                eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
            else:
                eta = lr_scheduler.get_lr() * lr_scale * X_pgd.grad.data.sign()
                lr_scheduler.step()
            X_pgd = torch.autograd.Variable(torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB), requires_grad=True)
    return adex, worst_x
