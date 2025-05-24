from collections import namedtuple
from beartype import beartype
import torch.nn as nn
import typing
import torch
import time
import copy
import os

from abstractor.auto_LiRPA.perturbations import PerturbationLpNorm
from heuristic.decompose_heuristics import DecomposeHeuristic
from attacker.pgd_attack.general import attack as pgd_attack
from helper.spec.write_vnnlib import write_vnnlib_classify
from helper.misc.torch_cuda_memory import gc_cuda
from abstractor.auto_LiRPA import BoundedTensor
from .verifier import Verifier


SubNetworks = namedtuple('SubNetworks', ['network', 'input_shape', 'output_shape'])

class PytorchWrapper(nn.Module):

    def __init__(self, module_lists):
        super(PytorchWrapper, self).__init__()
        self.layers = module_lists
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

@beartype
def new_input(self, x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(x_L <= x_U + 1e-8) #, f'{x_L=}\n\n{x_U=}'
    new_x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(self.device)
    if hasattr(self, 'extras'):
        new_x.ptb.extras = self.extras
    return new_x        

   
    

@beartype
def _setup_subnet_verifier(self, subnet_idx: int, objective: typing.Any | None = None, batch: int=500) -> Verifier:
    subnet_params = self.sub_networks[subnet_idx]
    # network
    if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
        tmp_network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
    else:
        tmp_network = subnet_params.network
    tmp_network.eval()
    network = copy.deepcopy(tmp_network)
    network.eval()
    
    verifier = Verifier(
        net=network,
        input_shape=subnet_params.input_shape,
        batch=batch,
        device=self.device,
    )
    verifier._setup_restart_naive(0, objective)
    verifier.abstractor.extras = self.extras.get(subnet_idx, None)
    
    return verifier


def attack_subnet(self, model, objective, timeout=5.0):
    print(model)
    print(f'{objective.lower_bounds.shape=}')
    input_lower = objective.lower_bounds.to(self.device)
    input_upper = objective.upper_bounds.to(self.device)
    assert torch.all(input_lower < input_upper)
    
    for i in range(1):
        x_attack = (input_upper - input_lower) * torch.rand(input_lower.shape, device=self.device) + input_lower
        print(f'Trial {i}: {x_attack.sum().item()=}')
        pred = model(x_attack).cpu().detach()
        
        write_vnnlib_classify(
            spec_path='example/vnnlib/spec_dec.vnnlib',
            data_lb=input_lower,
            data_ub=input_upper,
            prediction=pred
        )
        
        torch.onnx.export(
            model,
            x_attack,
            'example/onnx/net_dec.onnx',
            verbose=False,
            opset_version=12,
        )
        
        # exit()
        cs = objective.cs.to(self.device)
        rhs = objective.rhs.to(self.device)
        data_min_attack = input_lower.unsqueeze(1).expand(-1, len(cs), *input_lower.shape[1:])
        data_max_attack = input_upper.unsqueeze(1).expand(-1, len(cs), *input_upper.shape[1:])
        assert torch.all(data_min_attack < data_max_attack)
        # print(rhs.shape, cs.shape, data_max_attack.shape, input_upper.shape)
        is_attacked, attack_images = pgd_attack(
            model=model,
            x=x_attack, 
            data_min=data_min_attack,
            data_max=data_max_attack,
            cs=cs,
            rhs=rhs,
            attack_iters=500, 
            num_restarts=20,
            timeout=timeout,
            use_gama=False,
        )
        print(f'Trial {i}: {is_attacked=}\n')
        if is_attacked:
            raise
    exit()
    
    


@beartype
def _verify_subnet(self, subnet_idx: int, objective: typing.Any, verify_batch: int, timeout: int | float = 20.0) -> str:
    # release memory
    print(f'[+] _verify_subnet {subnet_idx=}')
    gc_cuda()
        
    subnet_input_outputs = self.input_output_bounds[subnet_idx]
    class TMP:
        pass
    
    assert torch.all(subnet_input_outputs.over_input[0] <= subnet_input_outputs.over_input[1])
    tmp_objective = TMP()
    tmp_objective.lower_bounds = subnet_input_outputs.over_input[0].clone()
    tmp_objective.upper_bounds = subnet_input_outputs.over_input[1].clone()
    tmp_objective.lower_bounds_f64 = tmp_objective.lower_bounds.to(torch.float64)
    tmp_objective.upper_bounds_f64 = tmp_objective.upper_bounds.to(torch.float64)
    tmp_objective.ids = objective.ids
    tmp_objective.cs = objective.cs
    tmp_objective.rhs = objective.rhs
    tmp_objective.cs_f64 = objective.cs.to(torch.float64)
    tmp_objective.rhs_f64 = objective.rhs.to(torch.float64)

    assert torch.all(tmp_objective.lower_bounds <= tmp_objective.upper_bounds)
    verifier = self._setup_subnet_verifier(
        subnet_idx=subnet_idx, 
        objective=copy.deepcopy(tmp_objective), 
        batch=verify_batch,
    )
    # print(verifier.net)

    verifier.start_time = time.time()

    # verifier.abstractor.extras = None
    status = verifier._verify_one(
        objective=copy.deepcopy(tmp_objective), 
        preconditions={}, 
        reference_bounds={}, 
        timeout=timeout,
    )
    
    del verifier.abstractor.net
    del verifier.abstractor
    del verifier
    
    print(f'{status=}')

    return status
    
    
@beartype
def decompose_network(self, net: nn.Module, objectives, input_shape: tuple) -> None:
    self.sub_networks = {}
    
    # decompose
    heuristic = DecomposeHeuristic(input_shape=input_shape)
    objective = objectives.pop(1)
    subnets = heuristic.decompose(net, objective)
    
    in_shape = input_shape
    for sidx, subnet in enumerate(subnets):
        out_shape = subnet(torch.randn(in_shape)).size()
        self.sub_networks[sidx] = SubNetworks(subnet, in_shape, out_shape)
        in_shape = out_shape
        
    # checking correctness
    net = net.cpu()
    dummy = torch.randn(2, *input_shape[1:])
    y1 = net(dummy)
    
    y2 = dummy
    for i in range(len(self.sub_networks)):
        y2 = self.sub_networks[i].network(y2)
        
    assert torch.allclose(y1, y2, 1e-4, 1e-4), f'norm={torch.norm(y1 - y2).item()}'
    print(f'[+] Passed decomposing network: {len(self.sub_networks)=}', f'norm={torch.norm(y1 - y2).item()}')

    