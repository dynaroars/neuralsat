from beartype import beartype
import torch
import random

from .pgd_attack.general import attack
from util.misc.logger import logger

import pdb
DBG = pdb.set_trace

class Attacker:
    
    @beartype
    def __init__(self, net, objective, input_shape, device) -> None:
        self.attackers = [PGDAttacker(net, objective, input_shape, mode=mode, device=device) 
                          for mode in ['PGD', 'diversed_PGD', 'diversed_GAMA_PGD']]
 
    @beartype
    def run(self):
        return self._attack()

    @beartype
    def _attack(self) -> tuple[bool, None | torch.Tensor]:
        for atk in self.attackers:
            seed = random.randint(0, 1000)
            atk.manual_seed(seed)
            logger.info(atk)
            is_attacked, adv = atk.run()
            if is_attacked:
                return is_attacked, adv
        return False, None


class PGDAttacker:

    def __init__(self, net, objective, input_shape, mode='PGD', device='cpu'):
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.dtype = torch.get_default_dtype()
        self.device = device

        assert mode in ['diversed_PGD', 'diversed_GAMA_PGD', 'PGD']
        self.mode = mode

        self.initialization = "uniform"
        self.gama_loss = False
        if "diversed" in mode:
            self.initialization = "osi"
        if "GAMA" in mode:
            self.gama_loss = True

        self.seed = None


    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    def run(self, iterations=100, restarts=20):
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        assert torch.all(data_min <= data_max)
        x = (data_min[:, 0] + data_max[:, 0]) / 2
        
        list_target_labels = [[(cs_.numpy(), rhs_.numpy()) for cs_, rhs_ in zip(self.objective.cs, self.objective.rhs)]]
        is_attacked, attack_images = attack(
            model=self.net,
            x=x, 
            data_min=data_min,
            data_max=data_max,
            list_target_label_arrays=list_target_labels, 
            initialization=self.initialization, 
            GAMA_loss=self.gama_loss,
            attack_iters=iterations, 
            num_restarts=restarts,
        )

        if is_attacked:
            with torch.no_grad():
                for idx in range(len(list_target_labels[0])):
                    adv = attack_images[:, 0, idx]
                    if self._check_adv(adv, data_min=data_min[:, idx], data_max=data_max[:, idx]):
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
        return f'PGDAttack(mode={self.mode}, seed={self.seed}, device={self.device})'

