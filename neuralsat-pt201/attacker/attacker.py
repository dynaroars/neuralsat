from beartype import beartype
import random
import torch

from onnx2pytorch.convert.model import ConvertModel
from verifier.objective import DnfObjectives

from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from util.misc.check import check_solution
from util.misc.logger import logger

from attacker.random_attack import RandomAttacker
from attacker.pgd_attack.general import attack

    
import pdb
DBG = pdb.set_trace

class Attacker:
    
    @beartype
    def __init__(self: 'Attacker', net: ConvertModel, objective: DnfObjectives, input_shape: tuple, device: str) -> None:
        self.attackers = [
            RandomAttacker(net, objective, input_shape, device=device),
            PGDAttacker(net, objective, input_shape, device=device),
        ]
 
    @beartype
    def run(self: 'Attacker', timeout: float = 0.5) -> tuple[bool, torch.Tensor | None]:
        return self._attack(timeout=timeout)

    @beartype
    def _attack(self: 'Attacker', timeout: float) -> tuple[bool, torch.Tensor | None]:
        for atk in self.attackers:
            seed = random.randint(0, 1000)
            atk.manual_seed(seed)
            try:
                # attacker using float64 might get OOM
                is_attacked, adv = atk.run(timeout=timeout)
            except RuntimeError as exception:
                if is_cuda_out_of_memory(exception):
                    # restore to default data type
                    atk.net.to(torch.get_default_dtype())
                    logger.info(f"[Failed] {atk} got OOM")
                    return False, None
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
    def __init__(self: 'PGDAttacker', net: ConvertModel, objective: DnfObjectives, input_shape: tuple, device: str = 'cpu') -> None:
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
    def run(self: 'PGDAttacker', iterations: int = 50, restarts: int = 20, timeout: float = 2.0) -> tuple[bool, torch.Tensor | None]:
        data_min = self.objective.lower_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max = self.objective.upper_bounds.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        
        data_min_f64 = self.objective.lower_bounds_f64.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        data_max_f64 = self.objective.upper_bounds_f64.view(-1, *self.input_shape[1:]).unsqueeze(0).to(self.device)
        
        # assert torch.all(data_min <= data_max)
        # x = (data_min[:, 0] + data_max[:, 0]) / 2
        x = (data_max[:, 0] - data_min[:, 0]) * torch.rand(data_min[:, 0].shape, device=self.device) + data_min[:, 0]
        assert torch.all(x <= data_max[:, 0])
        assert torch.all(x >= data_min[:, 0])
        
        cs = self.objective.cs.to(self.device)
        rhs = self.objective.rhs.to(self.device)
        
        cs_f64 = self.objective.cs_f64.to(self.device)
        rhs_f64 = self.objective.rhs_f64.to(self.device)
        
        
        # TODO: add timeout
        self.net.to(cs_f64.dtype)
        is_attacked, attack_images = attack(
            model=self.net,
            x=x.to(cs_f64.dtype), 
            data_min=data_min_f64,
            data_max=data_max_f64,
            cs=cs_f64,
            rhs=rhs_f64,
            attack_iters=iterations, 
            num_restarts=restarts,
            timeout=timeout,
        )
        
        if is_attacked:
            with torch.no_grad():
                for i in range(attack_images.shape[1]): # restarts
                    for j in range(attack_images.shape[2]): # props
                        adv = attack_images[:, i, j]
                        if check_solution(self.net, adv, cs=cs_f64[j], rhs=rhs_f64[j], data_min=data_min_f64[:, j], data_max=data_max_f64[:, j]):
                            return True, adv
            logger.debug("[!] Invalid counter-example")
        
        self.net.to(cs.dtype)
        return False, None

    
    def __str__(self):
        return f'PGDAttack(seed={self.seed}, device={self.device})'

