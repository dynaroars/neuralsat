import torch
import random
import math
import torch.nn as nn
import torch.distributions as dist
from adv_exp.attack_class import Attack_Class

default_params = {
        'iters': 40,
        'optimizer': None,
        'num_adv_ex': 5,
        'lr': 1e-4,
        'check_adv': 100,
        'mu': 0.1,
        'decay_alpha': True,
        'original_alpha': False,
        'lr_tensor': True,
        'mu_tensor': True,
        'decay_tensor': True,
        'early_stopping': True,
    }


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(dist.Uniform(lb.log(), ub.log()),
                                         dist.ExpTransform())


class MI_FGSM_Attack_CAN(Attack_Class):

    def __init__(self, params=None, cpu=False, store_loss_progress=False, model=None, data=None, model_can=None):
        self.__name__ = 'MI_FGSM__attack'
        self.params = dict(default_params, **params) if params is not None else default_params
        self.cpu = cpu
        self.store_loss_progress = store_loss_progress
        self.data = data
        self.model_can = model_can

    def create_adv_examples(self, data=None, model=None, return_criterion="one", init_tensor=None,
                            target=None, gpu=False, return_iters=False, multi_targets=False):
        with torch.enable_grad():
            if data is None:
                data = self.data

            x_lbs, x_ubs = data

            iters = self.params['iters']
            num_adv = self.params['num_adv_ex']

            # Calculate the mean of the normal distribution in logit space
            # As x_lbs == x_ubs raises an exception on dist.Uniform, handle the case separately
            adapted_x_ubs = torch.where(x_lbs == x_ubs, x_lbs + 1e-3, x_ubs)
            prior = dist.Uniform(low=x_lbs, high=adapted_x_ubs)
            images = torch.where((x_lbs == x_ubs).unsqueeze(0), x_lbs.unsqueeze(0), prior.sample(torch.Size([num_adv])))

            if not isinstance(init_tensor, type(None)):
                if images[0].size() == init_tensor.size():
                    images[0] = init_tensor
                elif images[0].size() == init_tensor[0].size():
                    images[:init_tensor.size()[0]] = init_tensor
                else:
                    raise RuntimeError("init tensor doesn't match image dimension")

            images.requires_grad = True
            self.loss_progress = []

            g_vec = torch.zeros_like(images)
            if self.params['original_alpha']:
                alpha = ((x_ubs[-1] - x_lbs[-1])/2) / iters
                eps = float(((x_ubs[-1] - x_lbs[-1]).view(-1)[0])/2)
                alpha = eps/iters
            elif self.params['lr_tensor']:
                # sample the learning rate
                alpha_min = self.params['lr'] * 0.1 * torch.ones(num_adv, device=images.device)
                alpha_max = self.params['lr'] * 10 * torch.ones(num_adv, device=images.device)
                dist_ = LogUniform(alpha_min, alpha_max)
                alpha = dist_.sample()
                alpha = alpha.view([-1] + [1] * (len(images.size()) - 1))
            else:
                alpha = self.params['lr']

            if self.params['mu_tensor']:
                # sample the momentum term
                mu_low = torch.zeros(num_adv, device=images.device)
                mu_high = torch.ones(num_adv, device=images.device)
                mu_dist = dist.Uniform(low=mu_low, high=mu_high)
                mu = mu_dist.sample()
                mu = mu.view([-1] + [1] * (len(images.size()) - 1))
            else:
                mu = self.params['mu']

            if self.params['decay_tensor']:
                # sample the decay factor
                dec_low = torch.ones(num_adv, device=images.device) * 0.85
                dec_high = torch.ones(num_adv, device=images.device) * 0.99
                dec_dist = dist.Uniform(low=dec_low, high=dec_high)
                dec_vec = dec_dist.sample()
                dec_vec = dec_vec.view([-1] + [1] * (len(images.size()) - 1))

            def forward_pass(images):
                im = images
                for layer in self.model_can:
                    im = layer(im)
                return im.mean(), im

            for i in range(iters):
                images.requires_grad = True

                cost, _ = forward_pass(images)
                cost.backward()

                g_vec = mu * g_vec - images.grad/torch.norm(images.grad, p=1)

                adv_images = images + alpha*g_vec.sign()
                images = torch.max(torch.min(adv_images, x_ubs), x_lbs).detach_()

                if self.params['decay_tensor']:
                    alpha = alpha * dec_vec
                elif self.params['decay_alpha']:
                    alpha = alpha * (float(i+1)/float(i+2))

                if self.store_loss_progress:
                    self.loss_progress.append(cost.detach())

                if i % self.params['check_adv'] == 0:
                    _, cost = forward_pass(images)
                    mean_ = (cost < torch.zeros_like(cost)).sum()
                    if return_criterion == "one" and mean_ > 0:
                        print("return early, iter ", i)
                        break

                if self.params['early_stopping'] and i == 0:
                    score_last_iter = cost.min()

                if self.params['early_stopping'] and i % 10 == 0 and i>0:
                    improvement = (score_last_iter - cost.min())
                    if not improvement > cost.min()/((iters - i)/10):
                        print(f"attack success unlikely, returning at {i} iters")
                        break
                    score_last_iter = cost.min()

            scores = forward_pass(images)[1]
            succ = scores < torch.zeros_like(cost)

            return images, succ, scores
