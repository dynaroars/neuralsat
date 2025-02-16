from collections import namedtuple
from beartype import beartype
import torch.nn as nn
import typing
import torch
import time
import copy
import os
import io

from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.network.read_onnx import parse_onnx, decompose_pytorch
from attacker.pgd_attack.general import attack as pgd_attack
from helper.spec.write_vnnlib import write_vnnlib_classify
from verifier.objective import DnfObjectives
from verifier.verifier import Verifier

from auto_LiRPA.abstractor.perturbations import PerturbationLpNorm
from auto_LiRPA.abstractor import BoundedModule, BoundedTensor

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
        # network = torch.nn.Sequential(torch.nn.Identity(), subnet_params.network)
        tmp_network = subnet_params.network
    tmp_network.eval()
    network = copy.deepcopy(tmp_network)
    network.eval()
    # print(network)
    
    # TODO: generalize for other verifiers
    verifier = Verifier(
        net=network,
        input_shape=subnet_params.input_shape,
        batch=batch,
        device=self.device,
    )
    verifier._setup_restart_naive(0, objective)
    verifier.abstractor.extras = self.extras.get(subnet_idx, None)
    
    return verifier


@beartype
def extract_tokenizer(self, net: nn.Module, objectives: DnfObjectives, input_shape: tuple) -> tuple[nn.Module, typing.Any, tuple]:
    return net, objectives, input_shape
    if not hasattr(net, 'tokenizer'):
        return net, objectives, input_shape
    
    # extract tokenizer
    # step 1: verify correctness
    dummy = torch.randn(2, *input_shape[1:], device=self.device)
    y1 = net(dummy)
    
    assert len(net.classifier) == 2 # TODO: allow more than 2 subnets
    new_net = PytorchWrapper(net.classifier)
    y2 = new_net(net.tokenizer(dummy))
    
    if not torch.allclose(y1, y2):
        return net, objectives, input_shape
        
    print('[Extract Tokenizer] Correctness checking: Pass')
    
    # step 2: initialize abstract domain
    tokenizer_abstractor = BoundedModule(
        model=net.tokenizer, 
        global_input=torch.zeros(input_shape, device=self.device),
        bound_opts={'conv_mode': 'patches', 'verbosity': 0},
        device=self.device,
        verbose=False,
    )
    tokenizer_abstractor.eval()
    
    # step 3: initialize intervals
    input_lowers = objectives.lower_bounds.view(-1, *input_shape[1:])
    input_uppers = objectives.upper_bounds.view(-1, *input_shape[1:])
    assert torch.all(input_lowers < input_uppers)
    x = self.new_input(x_L=input_lowers, x_U=input_uppers)
    assert not hasattr(x.ptb, 'extras')
    
    # step 3: forward intervals
    token_lowers, token_uppers = tokenizer_abstractor.compute_bounds(
        x=(x,), 
        method='backward', 
        bound_upper=True,
    )
    assert torch.all(token_lowers < token_uppers)
    # print(tokenizer_abstractor)
    # print(f'{input_lowers.shape=}')
    # print(f'{token_lowers.shape=}')
    
    # step 5: convert objectives
    objectives.lower_bounds = token_lowers.flatten(1).to(objectives.lower_bounds)
    objectives.upper_bounds = token_uppers.flatten(1).to(objectives.upper_bounds)
    assert torch.all(objectives.lower_bounds < objectives.upper_bounds)
    
    objectives.lower_bounds_f64 = token_lowers.flatten(1).to(objectives.lower_bounds_f64)
    objectives.upper_bounds_f64 = token_uppers.flatten(1).to(objectives.upper_bounds_f64)
    assert torch.all(objectives.lower_bounds_f64 < objectives.upper_bounds_f64)
    
    new_input_shape = (1,) + token_lowers.shape[1:]
    return new_net, objectives, new_input_shape



def attack_subnet(self, model, objective, timeout=5.0):
    #     tmp_objective = copy.deepcopy(objective)
    #     # FIXME: generalize to more than 2 subnetworks
    #     assert subnet_idx == 1, f'Invalid {subnet_idx=}'
    #     # update the input bounds
    #     tmp_objective.lower_bounds = self.input_output_bounds[subnet_idx].over_input[0].clone()
    #     tmp_objective.upper_bounds = self.input_output_bounds[subnet_idx].over_input[1].clone()
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
def decompose_network_auto(self, net: nn.Module, input_shape: tuple, min_layer: int) -> None:
    self.sub_networks = {}
    
    tmp_net = net
    tmp_shape = input_shape
    count = 0
    while True:
        prefix, suffix = self._split_network(tmp_net, tmp_shape, min_layer=min_layer)
        self.sub_networks[count] = SubNetworks(*prefix)
        if suffix is None:
            break 
        tmp_net, tmp_shape, _ = suffix
        count += 1
        # FIXME: support more than 2
        min_layer = 10000
        print(prefix)
        exit()
    
    # FIXME: support more than 2
    assert len(self.sub_networks) <= 2
    
    # check correctness
    dummy = torch.randn(2, *input_shape[1:], device=self.device)
    output_1 = net(dummy)
    output_2 = dummy.clone()
    for idx in range(len(self.sub_networks)):
        output_2 = self.sub_networks[idx].network(output_2)
    assert torch.allclose(output_1, output_2)
    print(f'[+] Passed {len(self.sub_networks)=}')


@beartype
def _split_network(self, net: nn.Module, input_shape: tuple, min_layer: int) -> tuple[tuple, tuple | None]:
    split_idx = min_layer
    while True:
        prefix_onnx_byte, suffix_onnx_byte = decompose_pytorch(net, input_shape, split_idx + 1)
        net = net.to(self.device)
        if (prefix_onnx_byte is None) or (suffix_onnx_byte is None):
            return (net, input_shape, None), None
        assert prefix_onnx_byte is not None

        # parse subnets
        try:
            prefix, prefix_input_shape, prefix_output_shape, _ = parse_onnx(prefix_onnx_byte)
            suffix, suffix_input_shape, suffix_output_shape, _ = parse_onnx(suffix_onnx_byte)
        except:
            print(f'Failed to split at {split_idx=}')
            # traceback.print_exc()
            split_idx += 1
            continue

        # move to device
        print(f'Succeeded to split at {split_idx=}')
        prefix = prefix.to(self.device)
        suffix = suffix.to(self.device)
        print(f'{prefix=}')
        print(f'{suffix=}')
            
        exit()
        # check correctness
        dummy = torch.randn(2, *prefix_input_shape[1:], device=self.device) # try batch=2
        if torch.allclose(net(dummy), suffix(prefix(dummy))):
            return (prefix.to(self.device), prefix_input_shape, prefix_output_shape), (suffix.to(self.device), suffix_input_shape, suffix_output_shape)
        

@beartype
def _verify_subnet(self, subnet_idx: int, objective: typing.Any, verify_batch: int, timeout: int | float = 20.0) -> str:
    # TODO: generalize for other verifier
    # release memory
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

    # tmp_objective2 = TMP()
    # tmp_objective2.lower_bounds = subnet_input_outputs.over_input[0].clone()
    # tmp_objective2.upper_bounds = subnet_input_outputs.over_input[1].clone()
    # tmp_objective2.lower_bounds_f64 = tmp_objective2.lower_bounds.to(torch.float64)
    # tmp_objective2.upper_bounds_f64 = tmp_objective2.upper_bounds.to(torch.float64)
    # tmp_objective2.ids = objective.ids
    # tmp_objective2.cs = objective.cs
    # tmp_objective2.rhs = objective.rhs
    # tmp_objective2.cs_f64 = objective.cs.to(torch.float64)
    # tmp_objective2.rhs_f64 = objective.rhs.to(torch.float64)
    
    assert torch.all(tmp_objective.lower_bounds <= tmp_objective.upper_bounds)
    verifier = self._setup_subnet_verifier(
        subnet_idx=subnet_idx, 
        objective=copy.deepcopy(tmp_objective), 
        batch=verify_batch,
    )
    # print(verifier.net)
    print(f'{objective.cs.shape=}')

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
    
def convert_tmp(net, input_shape):
    net.cpu().eval()
    print(net)
    
    # export
    # onnx_buffer = io.BytesIO()
    torch.onnx.export(
        net,
        torch.zeros(input_shape),
        'onnx_buffer.onnx',
        verbose=False,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    # onnx_buffer.seek(0)
    
    new_net = parse_onnx('onnx_buffer.onnx')[0]
    # print(new_net)
    exit()
    
@beartype
def decompose_network(self, net: nn.Module, input_shape: tuple, min_layer: int) -> None:
    self.sub_networks = {}
    if not (hasattr(net, 'layers') and isinstance(net.layers, nn.ModuleList)):
        return self.decompose_network_auto(net, input_shape, min_layer)
    
    assert isinstance(net.layers, nn.ModuleList)
    
    in_shape = input_shape
    for count, idx in enumerate(range(0, len(net.layers), min_layer)):
        subnet = PytorchWrapper(net.layers[idx:idx+min_layer])
        subnet.eval()
        # convert_tmp(subnet, in_shape)
        out_shape = subnet(torch.randn(in_shape, device=self.device)).size()
        self.sub_networks[count] = SubNetworks(subnet, in_shape, out_shape)
        in_shape = out_shape
        
        min_layer += 10000 # TODO: remove
        # print(count)
        if count == 1:
            break
        
    # checking correctness
    net = net.cpu()
    dummy = torch.randn(2, *input_shape[1:])
    y1 = net(dummy)
    
    y2 = dummy
    for i in range(len(self.sub_networks)):
        y2 = self.sub_networks[i].network(y2)
        
    assert torch.allclose(y1, y2, 1e-4, 1e-4), f'norm={torch.norm(y1 - y2).item()}'
    print(f'[+] Passed decomposing network: {len(self.sub_networks)=}', f'norm={torch.norm(y1 - y2).item()}')
    
    
    # for idx in range(len(self.sub_networks)):
    #     print(self.sub_networks[idx].network)
    #     subnet_input_shape = self.sub_networks[idx].input_shape
    #     subnet_output_shape = self.sub_networks[idx].output_shape
    #     print(f'{idx=} {subnet_input_shape=} {subnet_output_shape=}')
    #     print()
    