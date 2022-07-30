import torch
from torch import nn


class Attack_Class():
    ##############
    #   a super class for different algorithms that return adversarial examples
    #   subclasses include: mi_fgsm_attack.py and mi_fgsm_attack_canonical_form.py
    #   main function that needs to be implemented in subclasses is create_adv_examples(args**)
    #############
    def __init__(self, params=None, cpu=False):
        self.params = dict(default_params, **params) if params is not None else default_params
        self.cpu = cpu

    def update_params(self, params=None):
        self.params = dict(self.params, **params) if params is not None else self.params

    def adjust_lr(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def adjust_noise(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['noise_scale'] = new_lr

    def _loss(self, outputs, labels, target, return_vector=False):
        # returns the optimization loss
        if self.loss_type == 'CE_loss':
            return -self.CE_loss(outputs, labels)
        elif self.loss_type == 'targeted_loss':
            if len(target.size()) > 0:
                target_score = outputs.gather(1, target).squeeze()
            else:
                target_score = outputs[:, target]
            loss_vec = -outputs[:, labels[0]] + target_score
            assert(len(loss_vec.size()) == 1)
            if return_vector:
                return loss_vec.mean(), loss_vec
            else:
                return loss_vec.mean()

    def success_tensor(self, outputs, y, target):
        # returns a tensor of booleans indicating which examples are adversarial
        if self.targeted_attack:
            if len(target.size()) > 0:
                target_score = outputs.gather(1, target).squeeze()
            else:
                target_score = outputs[:, target]

            succ_tensor = outputs[:, y] < target_score

            assert(len(succ_tensor.size()) == 1)
        else:
            succ_tensor = outputs.max(dim=1).values > outputs[:, y]

        return succ_tensor, succ_tensor.sum(), succ_tensor.sum()/float(len(succ_tensor))

    def set_layers(self, layers):
        self.layers = layers
        self.net = nn.Sequential(*layers)

        for param in self.net.parameters():
            param.requires_grad = False
