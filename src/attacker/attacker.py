from beartype import beartype
import random
import torch
import time
import os

from helper.network.onnx2pytorch import ConvertModel
from helper.spec.objective import DnfObjectives

from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.misc.check import check_solution
from helper.misc.logger import logger

from attacker.random_attack import RandomAttacker
from attacker.pgd_attack.general import attack

    
import pdb
DBG = pdb.set_trace

class Attacker:
    
    @beartype
    def __init__(self: 'Attacker', net: ConvertModel | torch.nn.Module, objective: DnfObjectives, input_shape: tuple, device: str) -> None:
        self.attackers = [
            RandomAttacker(net, objective, input_shape, device=device),
            PGDAttacker(net, objective, input_shape, device=device),
        ]
 
    @beartype
    def run(self: 'Attacker', timeout: float = 0.5) -> tuple[bool, torch.Tensor | None]:
        return self._attack(timeout=timeout)

    @beartype
    def _attack(self: 'Attacker', timeout: float) -> tuple[bool, torch.Tensor | None]:
        tic = time.time()
        for atk in self.attackers:
            if time.time() - tic > timeout:
                return False, None
            remaining_timeout = timeout - (time.time() - tic)
            seed = random.randint(0, 1000)
            atk.manual_seed(seed)
            try:
                # attacker using float64 might get OOM
                is_attacked, adv = atk.run(timeout=remaining_timeout)
            except RuntimeError as exception:
                if is_cuda_out_of_memory(exception):
                    # restore to default data type
                    atk.net.to(torch.get_default_dtype())
                    logger.info(f"[Failed] {atk} got OOM")
                    continue
                else:
                    raise NotImplementedError
            except:
                raise NotImplementedError
            else:
                gc_cuda()
            logger.info(f"{'[Success]' if is_attacked else '[Failed]'} {atk}")
            if is_attacked:
                return is_attacked, adv
        return False, None


class PGDAttacker:

    @beartype
    def __init__(self: 'PGDAttacker', net: ConvertModel | torch.nn.Module, objective: DnfObjectives, input_shape: tuple, device: str = 'cpu') -> None:
        self.net = net
        self.objective = objective
        self.input_shape = input_shape
        self.device = device
        self.seed = None

    @beartype
    def manual_seed(self: 'PGDAttacker', seed: int) -> None:
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)


    @beartype
    def run(self: 'PGDAttacker', iterations: int = 100000, restarts: int = 20, timeout: float = 2.0) -> tuple[bool, torch.Tensor | None]:
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        
        # assert torch.all(data_min <= data_max)
        # x = (data_min[:, 0] + data_max[:, 0]) / 2
        x = (data_max[:, 0] - data_min[:, 0]) * torch.rand(data_min[:, 0].shape, device=self.device) + data_min[:, 0]
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(x <= data_max[:, 0])
            assert torch.all(x >= data_min[:, 0])
        
        cs = self.objective.cs.to(self.device)
        rhs = self.objective.rhs.to(self.device)
        
        print(f'Attacking PGD F32 {iterations=} {restarts=} {timeout=} {cs.shape=} {rhs.shape=}')
        is_attacked, attack_images = attack(
            model=self.net,
            x=x.to(cs.dtype), 
            data_min=data_min,
            data_max=data_max,
            cs=cs,
            rhs=rhs,
            attack_iters=iterations, 
            num_restarts=restarts,
            timeout=timeout / 2.0,
        )
    
        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if check_solution(self.net, adv, cs=cs[j], rhs=rhs[j], data_min=data_min[:, j], data_max=data_max[:, j]):
                            return True, adv
            logger.debug("[!] Invalid counter-example")
        
        self.net.to(cs.dtype)
        return False, None

    
    def __str__(self):
        return f'PGDAttack(seed={self.seed}, device={self.device})'

