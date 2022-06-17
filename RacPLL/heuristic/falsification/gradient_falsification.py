import numpy as np
import torch

from utils.adam import AdamClipping
import settings

class GradientFalsification:

    def __init__(self, net, spec):

        self.net = net
        self.spec = spec

        self.upper = torch.tensor([b[1] for b in self.spec.bounds], dtype=settings.DTYPE).view(self.net.input_shape)
        self.lower = torch.tensor([b[0] for b in self.spec.bounds], dtype=settings.DTYPE).view(self.net.input_shape)

        self.x = (self.upper + self.lower) / 2

        self.get_target()

        self.max_eps = 0.3
        self.attack_iters = 50
        self.num_restarts = 50

    def get_target(self):
        pidx_list = []
        self.only_target_attack = False
        for lhs, rhs in self.spec.mat:
            if len(rhs) > 1:
                output = self.net(self.x).detach().cpu().numpy().flatten()
                vec = lhs.dot(output)
                selected_prop = lhs[vec.argmax()]
                y = int(np.where(selected_prop == 1)[0])  # true label, whatever in target attack
                pidx = int(np.where(selected_prop == -1)[0])  # target label
                self.only_target_attack = True
            else:
                assert len(lhs) == 1
                y = np.where(lhs[0] == 1)[0]
                if len(y) != 0:
                    y = int(y)
                else:
                    y = None
                pidx = int(np.where(lhs[0] == -1)[0])  # target label
            pidx_list.append(pidx)

        self.target_label = y
        self.attack_label = pidx_list


    def evaluate(self):
        x = self.x.clone()
        if self.only_target_attack:
            # targeted attack.
            best_deltas, last_deltas = self.attack_pgd(
                X=x, 
                y=None, 
                alpha=self.max_eps,
                attack_iters=self.attack_iters, 
                num_restarts=self.num_restarts, 
                upper_limit=self.upper, 
                lower_limit=self.lower,
                multi_targeted=False, 
                target=self.attack_label[0])
        else:
            # untargeted attack PGD.
            alpha = self.max_eps/4.0
            best_deltas, last_deltas = self.attack_pgd(
                X=x, 
                y=torch.tensor([self.target_label]), 
                alpha=alpha,
                attack_iters=self.attack_iters, 
                num_restarts=self.num_restarts, 
                upper_limit=self.upper, 
                lower_limit=self.lower,
                multi_targeted=True, 
                target=None)

        attack_image = torch.max(torch.min(x + best_deltas, self.upper), self.lower)
        assert (attack_image >= self.lower).all()
        assert (attack_image <= self.upper).all()

        attack_output = self.net(attack_image).squeeze(0)
        attack_label = attack_output.argmax()

        # This is not the best image for each attack target. We should save best attack delta for each target.
        # all_targets_attack_image = torch.max(torch.min(x + last_deltas, self.upper), self.lower)


        if self.only_target_attack:
            # Targeted attack, must have one label.
            if attack_label == self.attack_label[0]:
                assert len(self.attack_label) == 1
                return True, attack_image.detach()
            return False, attack_image.detach()
        else:
            # Untargeted attack, any non-groundtruth label is ok.
            if attack_label != self.target_label:
                return True, attack_image.detach()
            return False, attack_image.detach()


    def attack_pgd(self, X, y, alpha, attack_iters, num_restarts, 
        epsilon=float("inf"), multi_targeted=True, num_classes=10, 
        use_adam=True, lr_decay=0.99, lower_limit=0.0, upper_limit=1.0, 
        normalize=lambda x: x, early_stop=True, target=None, 
        initialization='uniform'):


        best_loss = torch.empty(X.size(0), dtype=settings.DTYPE).fill_(float("-inf"))
        best_delta = torch.zeros_like(X, dtype=settings.DTYPE)

        input_shape = X.size()
        if multi_targeted:
            assert target is None  # Multi-targeted attack is for non-targed attack only.
            extra_dim = (num_restarts, num_classes - 1,)
            # Add two extra dimensions for targets. Shape is (batch, restarts, target, ...).
            X = X.unsqueeze(1).unsqueeze(1).expand(-1, *extra_dim, *(-1,) * (X.ndim - 1))
            # Generate target label list for each example.
            E = torch.eye(num_classes, dtype=settings.DTYPE)
            c = E.unsqueeze(0) - E[y].unsqueeze(1)
            # remove specifications to self.
            I = ~(y.unsqueeze(1) == torch.arange(num_classes).unsqueeze(0))
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
            # An attack target for targeted attack, in dimension (batch, ).
            target = torch.tensor(target, dtype=settings.DTYPE).view(-1,1)
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
            
            if initialization == 'uniform':
                delta = (torch.empty_like(X).uniform_() * (sample_upper_limit - sample_lower_limit) + sample_lower_limit).requires_grad_()
            else:
                raise ValueError(f"Unknown initialization method {initialization}")

            if use_adam:
                opt = AdamClipping(params=[delta], lr=alpha)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

            for _ in range(attack_iters):
                inputs = normalize(X + delta)
                if multi_targeted or target is not None:
                    output = self.net.forward_grad(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)
                else:
                    output = self.net.forward_grad(inputs)
                if not multi_targeted:
                    # Not using the multi-targeted loss. Can be target or non-target attack.
                    if target is not None:
                        # Targeted attack. In this case we have an extra (num_starts, ) dimension.
                        runnerup = output.scatter(dim=2, index=target_index, value=-float("inf")).max(2).values
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
                                success = True
                                break
                        elif target is not None:
                            # Targeted attack, the top-1 label of every element in batch must match the target.
                            # If any attack in some random restarts succeeds, the attack is successful.
                            if (output.max(2).indices == target).any(1).all():
                                success = True
                                break
                        else:
                            # Non-targeted attack, the top-1 label of every element in batch must not be the groundtruth label.
                            if (output.max(1).indices != y).all():
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