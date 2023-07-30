from beartype import beartype
import torch
import random

from util.misc.check import check_solution
from .random_attack import RandomAttacker
from .pgd_attack.general import attack
from util.misc.logger import logger

import pdb
DBG = pdb.set_trace

class Attacker:
    
    @beartype
    def __init__(self, net, objective, input_shape, device) -> None:
        self.attackers = [
            RandomAttacker(net, objective, input_shape, device=device),
            PGDAttacker(net, objective, input_shape, device=device),
        ]
 
    @beartype
    def run(self, timeout=0.5):
        return self._attack(timeout=timeout)

    @beartype
    def _attack(self, timeout: float) -> tuple[bool, None | torch.Tensor]:
        for atk in self.attackers:
            seed = random.randint(0, 1000)
            atk.manual_seed(seed)
            logger.info(atk)
            is_attacked, adv = atk.run(timeout=timeout)
            if is_attacked:
                return is_attacked, adv
        return False, None


class PGDAttacker:

    def __init__(self, net, objective, input_shape, device='cpu'):
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.device = device
        self.seed = None


    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    def run(self, iterations=50, restarts=20, timeout=1.0):
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        # assert torch.all(data_min <= data_max)
        # x = (data_min[:, 0] + data_max[:, 0]) / 2
        x = (data_max[:, 0] - data_min[:, 0]) * torch.rand(data_min[:, 0].shape, device=self.device) + data_min[:, 0]
        assert torch.all(x <= data_max[:, 0])
        assert torch.all(x >= data_min[:, 0])
        
        cs = self.objective.cs.to(self.device)
        rhs = self.objective.rhs.to(self.device)
        
        # TODO: add timeout
        is_attacked, attack_images = attack(
            model=self.net,
            x=x, 
            data_min=data_min,
            data_max=data_max,
            cs=cs,
            rhs=rhs,
            attack_iters=iterations, 
            num_restarts=restarts,
        )
        
        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if check_solution(self.net, adv, cs=cs[j], rhs=rhs[j], data_min=data_min[:, j], data_max=data_max[:, j]):
                            return True, adv
            logger.debug("[!] Invalid counter-example")
            
        return False, None

    
    def __str__(self):
        return f'PGDAttack(seed={self.seed}, device={self.device})'

