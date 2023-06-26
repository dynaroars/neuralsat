from beartype import beartype
import torch
import random

from .random_attack import RandomAttacker
from .pgd_attack.general import attack
from util.misc.logger import logger

import pdb
DBG = pdb.set_trace

class Attacker:
    
    @beartype
    def __init__(self, net, objective, input_shape, device) -> None:
        self.attackers = [
            PGDAttacker(net, objective, input_shape, device=device),
            RandomAttacker(net, objective, input_shape, device=device),
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
        self.dtype = torch.get_default_dtype()
        self.device = device
        self.seed = None


    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    def run(self, iterations=100, restarts=20, timeout=1.0):
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        assert torch.all(data_min <= data_max)
        x = (data_min[:, 0] + data_max[:, 0]) / 2
        
        # TODO: add timeout
        is_attacked, attack_images = attack(
            model=self.net,
            x=x, 
            data_min=data_min,
            data_max=data_max,
            cs=self.objective.cs,
            rhs=self.objective.rhs,
            attack_iters=iterations, 
            num_restarts=restarts,
        )

        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if self._check_adv(adv, data_min=data_min[:, j], data_max=data_max[:, j]):
                            return True, adv
            logger.debug("[!] Invalid counter-example")
            
        return False, None


    @torch.no_grad()
    def _check_adv(self, adv, data_min, data_max):
        if torch.all(data_min <= adv) and torch.all(adv <= data_max):
            output = self.net(adv).detach().cpu()
            cond = torch.matmul(self.objective.cs, output.unsqueeze(-1)).squeeze(-1) - self.objective.rhs
            return (cond.amax(dim=-1, keepdim=True) < 0.0).any(dim=-1).any(dim=-1)
        return False
    
    
    def __str__(self):
        return f'PGDAttack(seed={self.seed}, device={self.device})'

