# copy from alpha-beta-crown


from argparse import ArgumentTypeError
import math
import torch
import numpy as np
import arguments

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

def clamp(X, lower_limit=None, upper_limit=None):
    if lower_limit is None and upper_limit is None:
        return X
    if lower_limit is not None:
        return torch.max(X, lower_limit)
    if upper_limit is not None:
        return torch.min(X, upper_limit)
    return torch.max(torch.min(X, upper_limit), lower_limit)


def OSI_init(model, X, y, eps, alpha, num_classes, iter_steps=50, lower_limit=0.0, upper_limit=1.0, extra_dim=None):
    input_shape = X.size()
    if extra_dim is not None:
        X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
    expand_shape = X.size()

    X_init = X.clone().detach()

    # sample_lower_limit = torch.clamp(lower_limit - X, min=-eps)
    # sample_upper_limit = torch.clamp(upper_limit - X, max= eps)
        
    delta = (torch.empty_like(X).uniform_() * (upper_limit - lower_limit) + lower_limit)
    X_init = X_init + delta

    X = X.reshape(-1, *input_shape[1:])
    X_init = X_init.reshape(-1, *input_shape[1:])
    # Random vector from [-1, 1].
    w_d = (torch.rand([X.shape[0], num_classes], device=X.device) - 0.5) * 2

    if eps != float('inf'):
        lower_limit = torch.clamp(X-eps, min=lower_limit)
        upper_limit = torch.clamp(X+eps, max=upper_limit)

    for i in range(iter_steps):
        X_init = X_init.detach().requires_grad_()
        output = model(X_init)
        dot = (w_d * output).sum()
        grad = torch.autograd.grad(dot, X_init)[0]

        X_init = X_init + alpha * torch.sign(grad)
        X_init = torch.max(torch.min(X_init, upper_limit), lower_limit)

    assert (X_init <= upper_limit).all()
    assert (X_init >= lower_limit).all()

    assert (X_init <= X+eps).all()
    assert (X_init >= X-eps).all()

    return X_init.view(expand_shape)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, num_restarts,
        multi_targeted=True, num_classes=10, use_adam=True, lr_decay=0.98,
        lower_limit=0.0, upper_limit=1.0, normalize=lambda x: x, early_stop=True, target=None,
        initialization='uniform'):

    if initialization == 'osi':
        if multi_targeted:
            extra_dim = (num_restarts, num_classes - 1,)
        else:
            extra_dim = (num_restarts)
        X_init = OSI_init(model, X, y, epsilon, alpha, num_classes, iter_steps=attack_iters, extra_dim=extra_dim, upper_limit=upper_limit, lower_limit=lower_limit)

    best_loss = torch.empty(X.size(0), device=X.device).fill_(float("-inf"))
    best_delta = torch.zeros_like(X, device=X.device)

    input_shape = X.size()
    if multi_targeted:
        assert target is None  # Multi-targeted attack is for non-targed attack only.
        extra_dim = (num_restarts, num_classes - 1,)
        # Add two extra dimensions for targets. Shape is (batch, restarts, target, ...).
        X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
        # Generate target label list for each example.
        E = torch.eye(num_classes, dtype=X.dtype, device=X.device)
        c = E.unsqueeze(0) - E[y].unsqueeze(1)
        # remove specifications to self.
        I = ~(y.unsqueeze(1) == torch.arange(num_classes, device=y.device).unsqueeze(0))
        # c has shape (batch, num_classes - 1, num_classes).
        c = c[I].view(input_shape[0], num_classes - 1, num_classes)
        # c has shape (batch, restarts, num_classes - 1, num_classes).
        c = c.unsqueeze(1).expand(-1, num_restarts, -1, -1)
        target_y = y.view(-1,*(1,) * len(extra_dim),1).expand(-1, *extra_dim, 1)
        # Restart is processed in a batch and no need to do individual restarts.
        num_restarts = 1
        # If element-wise lower and upper limits are given, we should reshape them to the same as X.
        if lower_limit.ndim == len(input_shape):
            lower_limit = lower_limit.unsqueeze(1).unsqueeze(1)
        if upper_limit.ndim == len(input_shape):
            upper_limit = upper_limit.unsqueeze(1).unsqueeze(1)
    else:
        if target is not None:
            # An attack target for targeted attack, in dimension (batch, ).
            target = torch.tensor(target, device='cpu').view(-1,1)
            target_index = target.view(-1,1,1).expand(-1, num_restarts, 1)
            # Add an extra dimension for num_restarts. Shape is (batch, num_restarts, ...).
            X = X.unsqueeze(1).expand(-1, num_restarts, *(-1,) * (X.ndim - 1))
            # Only run 1 restart, since we run all restarts together.
            extra_dim = (num_restarts, )
            num_restarts = 1
            # If element-wise lower and upper limits are given, we should reshape them to the same as X.
            if lower_limit.ndim == len(input_shape):
                lower_limit = lower_limit.unsqueeze(1)
            if upper_limit.ndim == len(input_shape):
                upper_limit = upper_limit.unsqueeze(1)

    # This is the maximal/minimal delta values for each sample, each element.
    sample_lower_limit = torch.clamp(lower_limit - X, min=-epsilon)
    sample_upper_limit = torch.clamp(upper_limit - X, max=epsilon)

    success = False

    for n in range(num_restarts):
        # one_label_loss = True if n % 2 == 0 else False  # random select target loss or marginal loss
        one_label_loss = False  # Temporarily disabled. Will do more tests.
        if early_stop and success:
            break
        
        if initialization == 'osi':
            delta = (X_init - X).detach().requires_grad_()
        elif initialization == 'uniform':
            delta = (torch.empty_like(X).uniform_() * (sample_upper_limit - sample_lower_limit) + sample_lower_limit).requires_grad_()
        elif initialization == 'none':
            delta = torch.zeros_like(X).requires_grad_()
        else:
            raise ValueError(f"Unknown initialization method {initialization}")

        if use_adam:
            opt = AdamClipping(params=[delta], lr=alpha)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

        for _ in range(attack_iters):
            inputs = normalize(X + delta)
            if multi_targeted or target is not None:
                output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)
            else:
                output = model(inputs)
            if not multi_targeted:
                # Not using the multi-targeted loss. Can be target or non-target attack.
                if target is not None:
                    # Targeted attack. In this case we have an extra (num_starts, ) dimension.
                    runnerup = output.scatter(dim=2, index=target_index, value=-float("inf")).max(2).values
                    # t = output.gather(dim=2, index=target_index).squeeze(-1)
                    t = output[:, :, target].squeeze(-1).squeeze(-1)
                    loss = (t - runnerup)
                else:
                    # Non-targeted attack.
                    if one_label_loss:
                        # Loss 1: simply reduce the loss groundtruth target.
                        loss = -output.gather(dim=1, index=y.view(-1,1)).squeeze(1)  # -groundtruth
                    else:
                        # Loss 2: reduce the margin between groundtruth and runner up label.
                        runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
                        groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
                        # Use the margin as the loss function.
                        loss = (runnerup - groundtruth)
                loss.sum().backward()
            else:
                # Non-targeted attack, using margins between groundtruth class and all target classes together.
                # loss = torch.einsum('ijkl,ijkl->ijk', c, output)
                loss = torch.einsum('ijkl,ijkl->', c, output)
                loss.backward()

            # print(loss.sum().item(), output.detach().cpu().numpy())
            # print(loss[:, :, 5])

            with torch.no_grad():
                # Save the best loss so far.
                if not multi_targeted:
                    # Not using multi-targeted loss.
                    if target is not None:
                        # Targeted attack, need to check if the top-1 label is target label.
                        # Since we merged the random restart dimension, we need to find the best one among all random restarts.
                        all_loss, indices = loss.max(1)
                        # Gather the delta for the best loss in all random restarts.
                        delta_best = delta.gather(dim=1, index=indices.view(-1,1,1,1,1).expand(-1,-1,*input_shape[1:])).squeeze(1)
                        best_delta[all_loss >= best_loss] = delta_best[all_loss >= best_loss]
                        best_loss = torch.max(best_loss, all_loss)
                    else:
                        # Non-targeted attack. Success when the groundtruth is not top-1.
                        if one_label_loss:
                            runnerup = output.scatter(dim=1, index=y.view(-1,1), value=-100.0).max(1).values
                            groundtruth = output.gather(dim=1, index=y.view(-1,1)).squeeze(1)
                            # Use the margin as the loss function.
                            criterion = (runnerup - groundtruth)  # larger is better.
                        else:
                            criterion = loss
                        # Larger is better.
                        best_delta[criterion >= best_loss] = delta[criterion >= best_loss]
                        best_loss = torch.max(best_loss, criterion)
                else:
                    # Using multi-targeted loss. Need to find which label causes the worst case margin.
                    # Keep the one with largest margin.
                    # Note that we recompute the runnerup label here - the runnerup label might not be the target label.
                    # output has shape (batch, restarts, num_classes-1, num_classes).
                    # runnerup has shape (batch, restarts, num_classes-1).
                    runnerup = output.scatter(dim=3, index=target_y, value=-float("inf")).max(3).values
                    # groundtruth has shape (batch, restarts, num_classes-1).
                    groundtruth = output.gather(dim=3, index=target_y).squeeze(-1)
                    # margins has shape (batch, restarts * num_classes), ).
                    margins = (runnerup - groundtruth).view(groundtruth.size(0), -1)
                    # all_loss and indices have shape (batch, ), and this is the best loss over all restarts and number of classes.
                    all_loss, indices = margins.max(1)
                    # delta has shape (batch, restarts, num_class-1, c, h, w). For each batch element, we want to select from the best over (restarts, num_classes-1) dimension.
                    # delta_targeted has shape (batch, c, h, w).
                    delta_targeted = delta.view(delta.size(0), -1, *input_shape[1:]).gather(dim=1, index=indices.view(-1,1,*(1,) * (len(input_shape) - 1)).expand(-1,-1,*input_shape[1:])).squeeze(1)
                    best_delta[all_loss >= best_loss] = delta_targeted[all_loss >= best_loss]
                    best_loss = torch.max(best_loss, all_loss)

                if early_stop:
                    if multi_targeted:
                        # Must be a untargeted attack. If any of the target succeed, that element in batch is successfully attacked.
                        # output has shape (batch, num_restarts, num_classes-1, num_classes,).
                        if (output.view(output.size(0), -1, num_classes).max(2).indices != y.unsqueeze(1)).any(1).all():
                            print('pgd early stop.')
                            success = True
                            break
                    elif target is not None:
                        # Targeted attack, the top-1 label of every element in batch must match the target.
                        # If any attack in some random restarts succeeds, the attack is successful.
                        if (output.max(2).indices == target).any(1).all():
                            print('pgd early stop.')
                            success = True
                            break
                    else:
                        # Non-targeted attack, the top-1 label of every element in batch must not be the groundtruth label.
                        if (output.max(1).indices != y).all():
                            print('pgd early stop.')
                            success = True
                            break

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

    return best_delta, delta


def pgd_attack(dataset, model, x, max_eps, data_min, data_max, vnnlib=None, y=None,
               target=None, only_target_attack=False, initialization="uniform"):
    # FIXME (01/11/2022): any parameter that can be read from config should not be passed in.
    r"""Interface to PGD attack.

    Args:
        dataset (str): The name of dataset. Each dataset might have different attack configurations.

        model (torch.nn.Module): PyTorch module under attack.

        x (torch.tensor): Input image (x_0).

        max_eps (float): Perturbation Epsilon. Assuming Linf perturbation for now. (e.g., 0.3 for MNIST)

        data_min (torch.tensor): Lower bounds of data input. (e.g., 0 for mnist)

        data_max (torch.tensor): Lower bounds of data input. (e.g., 1 for mnist)

        vnnlib (list, optional): VNNLIB specifications. It will be used to extract attack target.

        y (int, optional): Groundtruth label. If specified, vnnlib will be ignored.

    Returns:
        success (bool): True if attack is successful. Otherwise False.
        attack_images (torch.tensor): last adversarial examples so far, may not be a real adversarial example if attack failed
    """
    assert arguments.Config["specification"]["norm"] == np.inf, print('We only support Linf-norm attack.')
    if dataset in ["MNIST", "CIFAR", "UNKNOWN"]:  # FIXME (01/11/2022): Make the attack function generic, not for the two datasets only!
        # FIXME (01/11/2022): Generic specification PGD.
        if y is not None and target is None:
            # Use y as the groundtruth label.
            pidx_list = ["all"]
            if arguments.Config["specification"]["type"] == "lp":
                if data_max is None:
                    data_max = x + max_eps
                    data_min = x - max_eps
                else:
                    data_max = torch.min(x + max_eps, data_max)
                    data_min = torch.max(x - max_eps, data_min)
            # If arguments.Config["specification"]["type"] == "bound", then we keep data_min and data_max.
        else:
            pidx_list = []
            if vnnlib is not None:
                # Extract attack target from vnnlib.
                for prop_mat, prop_rhs in vnnlib[0][1]:
                    if len(prop_rhs) > 1:
                        output = model(x).detach().cpu().numpy().flatten()
                        print('model output:', output)
                        vec = prop_mat.dot(output)
                        selected_prop = prop_mat[vec.argmax()]
                        y = int(np.where(selected_prop == 1)[0])  # true label, whatever in target attack
                        pidx = int(np.where(selected_prop == -1)[0])  # target label
                        only_target_attack = True
                    else:
                        assert len(prop_mat) == 1
                        y = np.where(prop_mat[0] == 1)[0]
                        if len(y) != 0:
                            y = int(y)
                        else:
                            y = None
                        pidx = int(np.where(prop_mat[0] == -1)[0])  # target label
                    if pidx == y:
                        raise NotImplementedError
                    pidx_list.append(pidx)
            elif target is not None:
                pidx_list.append(target)
            else:
                raise NotImplementedError

        print('##### PGD attack: True label: {}, Tested against: {} ######'.format(y, pidx_list))
        if not isinstance(max_eps, float):
            max_eps = torch.max(max_eps).item()

        if only_target_attack:
            # Targeted attack PGD.
            if arguments.Config["attack"]["pgd_alpha"] == 'auto':
                alpha = max_eps
            else:
                alpha = float(arguments.Config["attack"]["pgd_alpha"])
            best_deltas, last_deltas = attack_pgd(model, X=x, y=None, epsilon=float("inf"), alpha=alpha,
                    attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
                    multi_targeted=False, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=pidx_list[0], initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"])
        else:
            # Untargeted attack PGD.
            if arguments.Config["attack"]["pgd_alpha"] == 'auto':
                alpha = max_eps/4.0
            else:
                alpha = float(arguments.Config["attack"]["pgd_alpha"])
            best_deltas, last_deltas = attack_pgd(model, X=x, y=torch.tensor([y], device=x.device), epsilon=float("inf"), alpha=alpha,
                    attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"], upper_limit=data_max, lower_limit=data_min,
                    multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"], target=None, initialization=initialization, early_stop=arguments.Config["attack"]["pgd_early_stop"])

        if x.shape[0] == 1:
            attack_image = torch.max(torch.min(x + best_deltas, data_max), data_min)
            assert (attack_image >= data_min).all()
            assert (attack_image <= data_max).all()
            # assert (attack_image-x).abs().max() <= eps_temp.max(), f"{(attack_image-x).abs().max()} <= {eps_temp.max()}"
            attack_output = model(attack_image).squeeze(0)

            # FIXME (10/02): This is not the best image for each attack target. We should save best attack delta for each target.
            all_targets_attack_image = torch.max(torch.min(x + last_deltas, data_max), data_min)

            attack_label = attack_output.argmax()
            print("pgd prediction:", attack_output)

            # FIXME (10/05): Cleanup.
            if only_target_attack:
                # Targeted attack, must have one label.
                attack_logit = attack_output.data[pidx_list[0]].item()
                attack_output.data[pidx_list[0]] = -float("inf")
                attack_margin = attack_output.max().item() - attack_logit
                print("attack margin", attack_margin)
                if attack_label == pidx_list[0]:
                    assert len(pidx_list) == 1
                    print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    # FIXME (10/05): Please check! attack_image is for one target only.
                    return True, attack_image.detach(), [attack_margin]
                else:
                    print(f"targeted pgd failed, margin {attack_margin}")
                    return False, attack_image.detach(), [attack_margin]
            else:
                # Untargeted attack, any non-groundtruth label is ok.
                groundtruth_logit = attack_output.data[y].item()
                attack_output.data[y] = -float("inf")
                attack_margin = groundtruth_logit - attack_output
                print("attack margin", attack_margin)
                # Untargeted attack, any non-groundtruth label is ok.
                if attack_label != y:
                    print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    return True, all_targets_attack_image.detach().detach(), attack_margin.detach().cpu().numpy()
                else:
                    print("untargeted pgd failed")
                    return False, all_targets_attack_image.detach(), attack_margin.detach().cpu().numpy()

        else:
            # FIXME (10/02): please remove duplicated code!
            attack_images = torch.max(torch.min(x + best_deltas, data_max), data_min)
            attack_output = model(attack_images).squeeze(0)
            # do in batch attack
            attack_label = attack_output.argmax(1)

            if only_target_attack:
                # Targeted attack, must have one label.
                if (attack_label == pidx_list[0]).any():
                    # FIXME (10/02): remove duplicated code.
                    assert len(pidx_list) == 1
                    # print("targeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    attack_logit = attack_output.data[:, pidx_list[0]].clone()
                    attack_output.data[:, pidx_list[0]] = -float("inf")
                    attack_margin = attack_output.max(1).values - attack_logit
                    return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
                else:
                    attack_logit = attack_output.data[:, pidx_list[0]].clone()
                    attack_output.data[:, pidx_list[0]] = -float("inf")
                    attack_margin = attack_output.max(1).values - attack_logit
                    # print(f"targeted pgd failed, margin {attack_margin}")
                    return False, attack_images.detach(), attack_margin.detach().cpu().numpy()
            else:
                raise NotImplementedError
                # TODO support batch
                # Untargeted attack, any non-groundtruth label is ok.
                groundtruth_logit = attack_output.data[y].item()
                attack_output.data[y] = -float("inf")
                attack_margin = groundtruth_logit - attack_output
                # Untargeted attack, any non-groundtruth label is ok.
                if attack_label != y:
                    print("untargeted pgd succeed, label {}, against label {}".format(y, attack_label))
                    return True, attack_images.detach(), attack_margin.detach().cpu().numpy()
                else:
                    print("untargeted pgd failed")
                    return False, attack_images.detach(), attack_margin.detach().cpu().numpy()



    else:
        print("pgd attack not supported for dataset", dataset)
        raise NotImplementedError
