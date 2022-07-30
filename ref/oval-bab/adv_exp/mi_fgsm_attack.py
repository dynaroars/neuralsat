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
        'decay_alpha': False,
        'original_alpha': True,
    }


class MI_FGSM_Attack(Attack_Class):

    def __init__(self, params=None, cpu=False, store_loss_progress=False, model=None, data=None):
        self.__name__ = 'MI_FGSM__attack'
        self.params = dict(default_params, **params) if params is not None else default_params
        self.cpu = cpu
        self.store_loss_progress = store_loss_progress
        self.model = model
        self.data = data

    def create_adv_examples(self, data=None, model=None, return_criterion="all", init_tensor=None,
                            target=None, gpu=False, return_iters=False, multi_targets=False):
        with torch.enable_grad():
            assert return_criterion in ["one", "half", "all", "not_early"]
            # self.targeted_attack = type(target) != type(None)
            self.targeted_attack = not isinstance(target, type(None))

            if model is None:
                model = self.model
            if data is None:
                data = self.data

            x, y, x_lbs, x_ubs = data
            if gpu and torch.cuda.is_available():
                x = x.cuda()
                x_lbs = x_lbs.cuda()
                x_ubs = x_ubs.cuda()
                model.cuda()
            device = x.device

            iters = self.params['iters']
            num_adv = self.params['num_adv_ex']

            if device.type == 'cpu':
                labels = torch.LongTensor([y]*num_adv, device=device)
            else:
                labels = torch.cuda.LongTensor([y]*num_adv, device=device)

            if multi_targets:
                num_classes = model[-1].out_features
                num_per_class = math.ceil(self.params['num_adv_ex'] / (num_classes - 1))
                target_list = [[k]*num_per_class for k in range(num_classes) if k != y]
                target_list = [item for sublist in target_list for item in sublist]
                random.shuffle(target_list)
                target_list = target_list[:self.params['num_adv_ex']]
                target = torch.LongTensor(target_list).unsqueeze(1)
                if gpu:
                    target = target.cuda()

            # Calculate the mean of the normal distribution in logit space
            prior = dist.Uniform(low=x_lbs, high=x_ubs)
            images = prior.sample(torch.Size([num_adv]))   # Alg1 line 2

            if not isinstance(init_tensor, type(None)):
                if images[0].size() == init_tensor.size():
                    images[0] = init_tensor
                elif images[0].size() == init_tensor[0].size():
                    images[:init_tensor.size()[0]] = init_tensor
                else:
                    print("image size", images.size(), images[0].size())
                    print("init tensor size", init_tensor.size(), init_tensor[0].size())
                    input("images and init tensor not compatible")

            if self.params['optimizer']:
                if self.params['optimizer'] == 'default':
                    alpha = self.params['lr']
                    images.requires_grad = True
                else:
                    print("optimizer", self.params['optimizer'])
                    raise NotImplementedError

            if not isinstance(target, type(None)):
                self.loss_type = 'targeted_loss'
            else:
                self.loss_type = 'CE_loss'
                self.CE_loss = nn.CrossEntropyLoss()
            loss = nn.CrossEntropyLoss()

            self.loss_progress = []

            g_vec = torch.zeros_like(images)
            mu = self.params['mu']
            if self.params['original_alpha']:
                alpha = ((x_ubs[-1] - x_lbs[-1])/2) / iters
                eps = float(((x_ubs[-1] - x_lbs[-1]).view(-1)[0])/2)
                alpha = eps/iters
            else:
                alpha = self.params['lr']

            for i in range(iters):

                images.requires_grad = True
                outputs = model(images)

                model.zero_grad()
                cost = self._loss(outputs, labels, target).to(device)
                cost.backward()

                g_vec = mu * g_vec + images.grad/torch.norm(images.grad, p=1)

                adv_images = images + alpha*g_vec.sign()
                images = torch.max(torch.min(adv_images, x_ubs), x_lbs).detach_()

                if self.params['decay_alpha']:
                    alpha = alpha * (float(i+1)/float(i+2))

                if self.store_loss_progress:
                    self.loss_progress.append(cost.detach())

                if i % self.params['check_adv'] == 0:
                    outputs = model(images)
                    succ, sum_, mean_ = self.success_tensor(outputs, y, target)
                    if return_criterion == "all" and mean_ == 1:
                        break
                    elif return_criterion == "one" and mean_ > 0:
                        print("return early, iter ", i)
                        break
                    elif return_criterion == "half" and mean_ >= 0.5:
                        break

            succ, sum_, mean_ = self.success_tensor(outputs, y, target)

            output = model(images)
            _, loss = self._loss(outputs, labels, target, return_vector=True)
            loss = -loss.to(device)

            if return_iters:
                return images, succ, i
            else:
                return images, succ, loss
