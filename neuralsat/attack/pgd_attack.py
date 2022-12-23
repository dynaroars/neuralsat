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
        self.mode = mode

        # choices=['diversed_PGD', 'diversed_GAMA_PGD', 'PGD'],
        self.initialization = "uniform"
        self.gama_loss = False

        if "diversed" in mode:
            self.initialization = "osi"

        if "GAMA" in mode:
            self.gama_loss = True

        self.dtype = arguments.Config['dtype']
        self.seed = None

    def manual_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed


    def run(self):
        list_target_labels, data_min, data_max = preprocess_spec(self.spec, self.net.input_shape, self.net.device)
        x_range = torch.tensor(self.spec[0], dtype=self.dtype, device=self.net.device)
        x = x_range.mean(-1).reshape(self.net.input_shape)

        is_attacked, attack_images, attack_margins, all_adv_candidates = attack(self.net,
                                                                                x, 
                                                                                data_min[:,:len(list_target_labels[0]),...],
                                                                                data_max[:,:len(list_target_labels[0]),...], 
                                                                                list_target_labels, 
                                                                                initialization=self.initialization, 
                                                                                GAMA_loss=self.gama_loss)

        if is_attacked:
            spec_vnnlib = SpecVNNLIB(self.spec)
            with torch.no_grad():
                for idx in range(len(list_target_labels[0])):
                    adv = attack_images[:, idx]
                    if spec_vnnlib.check_solution(self.net(adv)):
                        return True, adv
            assert False
        return False, None

    
    def __str__(self):
        return f'PGDAttack(mode={self.mode}, seed={self.seed})'