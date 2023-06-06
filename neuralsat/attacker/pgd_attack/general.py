import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time

from .util import *

def attack(model, x, data_min, data_max, list_target_label_arrays, initialization="uniform", GAMA_loss=False, attack_iters=100, num_restarts=10):
    device = x.device
    max_eps = torch.max(data_max - data_min).item() / 2
    alpha = max_eps / 4

    # set all parameters without gradient, this can speedup things significantly.
    grad_status = {}
    for p in model.parameters():
        grad_status[p] = p.requires_grad
        p.requires_grad_(False)

    output = model(x).detach()
    
    C_mat, rhs_mat, cond_mat, same_number_const = build_conditions(x, list_target_label_arrays)

    output = output.unsqueeze(1).unsqueeze(1).repeat(1, 1, len(cond_mat[0]), 1)
    if test_conditions(x, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max.unsqueeze(1), data_min.unsqueeze(1)).all():
        return True, x[:, None, None].detach()


    data_min = data_min.to(device)
    data_max = data_max.to(device)
    rhs_mat = rhs_mat.to(device)
    C_mat = C_mat.to(device)
    attack_images = general_attack(
        model=model, 
        X=x, 
        data_min=data_min, 
        data_max=data_max, 
        C_mat=C_mat, 
        rhs_mat=rhs_mat, 
        cond_mat=cond_mat, 
        same_number_const=same_number_const, 
        alpha=alpha, 
        attack_iters=attack_iters, 
        num_restarts=num_restarts, 
        initialization=initialization, 
        GAMA_loss=GAMA_loss,
    )

    # set back to original requires_grad status.
    for p in model.parameters():
        p.requires_grad_(grad_status[p])

    if attack_images is not None:
        return True, attack_images.detach()
    return False, None



def general_attack(model, X, data_min, data_max, C_mat, rhs_mat, cond_mat, same_number_const, alpha, 
                   use_adam=True, normalize=lambda x: x, initialization='uniform', GAMA_loss=False, 
                   num_restarts=10, attack_iters=100, only_replicate_restarts=False):
    lr_decay = 0.99
    gama_lambda = 10

    input_shape = (X.shape[0], *X.shape[2:]) if only_replicate_restarts else X.size()
    device = X.device

    num_classes = C_mat.shape[-1]
    num_or_spec = len(cond_mat[0])
    extra_dim = (num_restarts, num_or_spec) if only_replicate_restarts == False else (num_restarts,)
    # shape of x: [num_example, *shape_of_x]

    # [1, 1, num_spec, *input_shape]
    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)

    X_ndim = X.ndim
    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X

    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim - 1))
    extra_dim = (X.shape[1], X.shape[2])

    if initialization == 'osi':
        X_init = osi_init(model, X, alpha, C_mat.shape[-1], attack_iters, data_min, data_max)
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

        _, loss_gama = build_loss(origin_out, output, C_mat, rhs_mat, cond_mat, same_number_const, gama_lambda if GAMA_loss else 0.0, mode='hinge')
        gama_lambda *= 0.9
        loss_gama.sum().backward()

        if test_conditions(inputs, output, C_mat, rhs_mat, cond_mat, same_number_const, data_max, data_min).all():
            # print("early stop")
            return inputs

        if use_adam:
            opt.step(clipping=True, lower_limit=delta_lower_limit, upper_limit=delta_upper_limit, sign=1)
            opt.zero_grad(set_to_none=True)
            scheduler.step()
        else:
            d = delta + alpha * torch.sign(delta.grad)
            d = torch.max(torch.min(d, delta_upper_limit), delta_lower_limit)
            delta.copy_(d)
            delta.grad = None

    return None
    