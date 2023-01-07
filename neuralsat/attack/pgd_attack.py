import torch.nn as nn
import numpy as np
import random
import torch
import copy

from util.spec.spec_vnnlib import SpecVNNLIB
from attack.error import AttackTimeoutError
import arguments

from ._pgd_attack.util import preprocess_spec
from ._pgd_attack.general import attack

class PGDAttack:

    def __init__(self, net, spec, timeout=0.5, mode='PGD'):
        self.net = net
        self.spec = spec
        self.timeout = timeout

        assert mode in ['diversed_PGD', 'diversed_GAMA_PGD', 'PGD']
        self.mode = mode

        self.initialization = "uniform"
        self.gama_loss = False

        if "diversed" in mode:
            self.initialization = "osi"

        if "GAMA" in mode:
            self.gama_loss = True

        self.dtype = arguments.Config['dtype']
        self.device = arguments.Config['device']
        self.seed = None

        self.list_target_labels, self.orig_data_min, self.orig_data_max = preprocess_spec(self.spec, self.net.input_shape, self.net.device)
        self.num_dnf = len(self.list_target_labels[0])

    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    def run(self, mutate=False):
        if not mutate:
            data_min, data_max = self.orig_data_min, self.orig_data_max
            # [batch, self.num_dnf, *input_shape]
        else:
            data_min, data_max = self.mutate()

        assert torch.all(data_min <= data_max)
       
        x = (data_min[:, 0] + data_max[:, 0]) / 2

        is_attacked, attack_images, attack_margins, all_adv_candidates = attack(model=self.net,
                                                                                x=x, 
                                                                                data_min=data_min[:,:self.num_dnf,...],
                                                                                data_max=data_max[:,:self.num_dnf,...], 
                                                                                list_target_label_arrays=self.list_target_labels, 
                                                                                initialization=self.initialization, 
                                                                                GAMA_loss=self.gama_loss,
                                                                                attack_iters=100, 
                                                                                num_restarts=20)

        if is_attacked:
            spec_vnnlib = SpecVNNLIB(self.spec)
            with torch.no_grad():
                for idx in range(self.num_dnf):
                    adv = attack_images[:, idx]
                    if spec_vnnlib.check_solution(self.net(adv)):
                        return True, adv
            assert False
        return False, None

    def mutate(self):
        # print('\t- seed:', self.seed)

        mean = (self.orig_data_max + self.orig_data_min) / 2
        eps = (self.orig_data_max - self.orig_data_min) / 2

        scaled = torch.randint(low=1, high=5, size=(2,))
        factor = torch.rand(2, *self.orig_data_max.shape, device=self.device)
        # factor = factor.to(self.device)

        data_max = self.orig_data_max - (factor[0] / scaled[0]) * eps
        data_min = self.orig_data_min + (factor[1] / scaled[1]) * eps

        # print(factor[0].flatten()[:5])

        assert torch.all(data_max <= self.orig_data_max)
        assert torch.all(data_min >= self.orig_data_min)

        # print(scaled, torch.sum(self.orig_data_max - self.orig_data_min).item(), torch.sum(data_max - data_min).item())
        # exit()

        return data_min, data_max

    
    def __str__(self):
        return f'PGDAttack(mode={self.mode}, seed={self.seed})'