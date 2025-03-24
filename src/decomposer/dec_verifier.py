from collections import namedtuple
from beartype import beartype
import torch.nn as nn
import traceback
import logging
import random
import typing
import torch
import time
import copy
import tqdm
import math
import sys
import os


from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.misc.result import ReturnStatus, CoefficientMatrix
from helper.spec.objective import DnfObjectives
from helper.misc.logger import logger

from tightener.utils import optimize_dnn, verify_dnf_pairs

from verifier.utils import get_used_gpu_memory
from verifier.verifier import Verifier

from setting import Settings

from trainer.models.resnet.resnet import *
from trainer.models.vit.vit import *

from abstractor.auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule
from abstractor.auto_LiRPA.utils import stop_criterion_batch_any
from abstractor.params import get_initialize_opt_params

InOutBounds = namedtuple('InputOutputBounds', ['under_input', 'under_output', 'over_input', 'over_output'], defaults=(None,) * 4)

def redundant_compute_bounds(net, input_lowers, input_uppers, cs, method='backward'):
    assert method in ['backward', 'crown-optimized']
    new_x = BoundedTensor(input_lowers, PerturbationLpNorm(x_L=input_lowers, x_U=input_uppers)).to(input_uppers)
    
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros(input_lowers.shape, device=input_lowers.device),
        bound_opts={'conv_mode': 'patches', 'verbosity': 0},
        device=input_lowers.device,
        verbose=False,
    )
    print(net)
    l, u, aux_reference_bounds = abstract.init_alpha(
        x=(new_x,), 
        share_alphas=Settings.share_alphas, 
        c=cs, 
        bound_lower=True,
        bound_upper=True,
    )
    
    if method == 'backward':
        return (l, u), None
    
    abstract.set_bound_opts(get_initialize_opt_params(lambda x: False))
    # abstract.set_bound_opts({'optimize_bound_args': {'iteration': 50}})
    
    l, _ = abstract.compute_bounds(
        x=(new_x,), 
        C=cs,
        method='crown-optimized',
        aux_reference_bounds=aux_reference_bounds, 
        bound_lower=True, 
        bound_upper=False, 
    )
    return (l.clone(), l.clone()), None
    
    _, u = abstract.compute_bounds(
        x=(new_x,), 
        C=cs,
        method='crown-optimized',
        aux_reference_bounds=aux_reference_bounds, 
        bound_lower=False, 
        bound_upper=True, 
    )
    
    assert torch.all(l <= u)
    del abstract
    return (l, u), None
    
class DecompositionalVerifier:
    
    @beartype
    def __init__(self, net: nn.Module, input_shape: tuple, min_layer: int, device: str = 'cpu') -> None:
        self.net = net.to(device) # pytorch model
        self.device = device
        self.input_shape = input_shape
        self.min_layer = min_layer
        
    @beartype
    def reset(self) -> None:
        if not hasattr(self, 'input_output_bounds'):
            self.input_output_bounds = {k: None for k in range(len(self.sub_networks))}
        
        if not hasattr(self, 'extras'):
            self.extras = {0: None}
        
        self.tightening_candidates = {}

        
    def compute_bounds(self, abstractor, input_lowers, input_uppers, cs, method, output_sequential=False, output_batch=10):
        
        diff = (input_uppers - input_lowers).clone()
        eps = diff.max().item()
        
        if not output_sequential:
            print(f'{cs.device=} {cs.shape=} {eps=}')
            # return abstractor.compute_bounds(
            #     input_lowers=input_lowers,
            #     input_uppers=input_uppers,
            #     cs=cs,
            #     method=method,
            # )
            return redundant_compute_bounds(
                net = abstractor.pytorch_model,
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                cs=cs,
                method=method
            )
            
        assert len(input_lowers) == 1, f'Only support batch=1: {len(input_lowers)=}'
        n_outputs = abstractor.net(input_lowers).flatten(1).shape[1]
        
        # print(f'{n_outputs=}')
        output_lowers, output_uppers = [], []
        output_coeffs = []
        
        pbar = tqdm.tqdm(range(0, n_outputs, output_batch), desc=f'Compute bounds: {output_batch=} {method=} {eps=:.06f}')
        for i in pbar:
            indices = torch.arange(i, min(i+output_batch, n_outputs))
            ci = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lowers)
            # TODO: reuse optimized slopes
            (lb, ub), coeffs = abstractor.compute_bounds(
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                cs=ci,
                method=method,
                # reuse_alpha=i!=0,
            )
            # assert torch.all(lb <= ub + 1e-6), f'{(lb > ub).sum()} {lb[lb > ub]} {ub[lb > ub]}'
            # print(ci.shape, lb.shape)
            output_lowers.append(lb)
            output_uppers.append(ub)
            # print(f'{coeffs.lA.shape=}')
            if coeffs:
                output_coeffs.append(coeffs)
                # raise # TODO: disabling coeffs might save memory
            
        output_lowers = torch.cat(output_lowers, dim=-1)
        output_uppers = torch.cat(output_uppers, dim=-1)
        
        
        if output_coeffs:
            output_coeffs = CoefficientMatrix(
                lA=torch.cat([c.lA for c in output_coeffs], dim=0),
                uA=torch.cat([c.uA for c in output_coeffs], dim=0),
                lbias=torch.cat([c.lbias for c in output_coeffs], dim=-1),
                ubias=torch.cat([c.ubias for c in output_coeffs], dim=-1),
            )
            # raise # TODO: disabling coeffs might save memory
        
        return (output_lowers, output_uppers), output_coeffs
        
    @beartype
    def _init_interm_bounds(self, objective: typing.Any, use_extra: bool = True, method='crown-optimized', interm_batch: int = 200) -> tuple:
        # assert method == 'backward'
        print(f'Init interm bounds {method=}')
        # input 
        input_shape = self.sub_networks[0].input_shape
        input_lb_0 = objective.lower_bounds.view(input_shape).to(self.device)
        input_ub_0 = objective.upper_bounds.view(input_shape).to(self.device)
        if self.input_output_bounds[0] is None:
            self.input_output_bounds[0] = InOutBounds(over_input=(input_lb_0.clone(), input_ub_0.clone()))
        
        n_subnets = len(self.sub_networks)
        
        # TODO: try computing bounds with original network first
        
        # try computing bounds with sub-networks
        for idx in range(n_subnets):
            # if idx == n_subnets - 1:
            #     method = 'backward'
                
            # if idx <= 1:
            #     method = 'backward'
            # else:
            #     method = 'crown-optimized'
                
            if (idx != n_subnets-1) and (self.input_output_bounds[idx].over_output is not None):
                # no need to re-init for different objectives, only need to update last over_output due to cs
                # TODO: recheck input property
                print(f'Reuse computed bounds subnet {idx=}')
                continue
            verifier = self._setup_subnet_verifier(idx)
            # print(f'{verifier.abstractor.net=}')
            # print(verifier.net)
            # print(self.sub_networks[idx].input_shape)
            # print(self.sub_networks[idx].output_shape)
            
            print(f'Processing subnet {idx+1}/{n_subnets} {method=}')
            cs_to_use = objective.cs.to(input_lb_0)
            (output_lb, output_ub), output_coeffs = self.compute_bounds(
                abstractor=verifier.abstractor,
                input_lowers=self.input_output_bounds[idx].over_input[0],
                input_uppers=self.input_output_bounds[idx].over_input[1],
                cs=cs_to_use if idx==n_subnets-1 else None,
                method=method,
                output_sequential=idx!=n_subnets-1,
                output_batch=interm_batch,
            )
            
            print(f'[+] Init bounds subnet {idx+1}/{n_subnets} {method=}:', get_used_gpu_memory(), 'MB')
            # exit()
            
            # # DEBUG: check correctness
            # (output_lb2, output_ub2), output_coeffs2 = verifier.abstractor.compute_bounds(
            #     input_lowers=self.input_output_bounds[idx].over_input[0],
            #     input_uppers=self.input_output_bounds[idx].over_input[1],
            #     cs=objective.cs if idx==n_subnets-1 else None,
            #     method=method,
            # )
            # assert torch.allclose(output_ub, output_ub2), f'{torch.norm(output_ub - output_ub2)}'
            # print('Pass ub', torch.norm(output_ub - output_ub2))
            # assert torch.allclose(output_lb, output_lb2), f'{torch.norm(output_lb - output_lb2)}'
            # print('Pass lb', torch.norm(output_lb - output_lb2))
            
            # for field in ['lA', 'uA', 'lbias', 'ubias']:
            #     f1 = getattr(output_coeffs, field)
            #     f2 = getattr(output_coeffs2, field)
            #     assert torch.allclose(f1, f2, atol=1e-5), f'{torch.norm(f1 - f2)}'
            #     print(f'Pass {field}', torch.norm(f1 - f2))
                
            if idx == 0 and use_extra and output_coeffs is not None: 
                assert len(output_coeffs)
                # TODO: generalize for more than 2 subnets
                # additional backsub up to the original input 
                self.extras[idx + 1] = {
                    'input': (input_lb_0.clone(), input_ub_0.clone()),
                    'coeff': output_coeffs
                }
            
            # flatten output
            subnet_params = self.sub_networks[idx]
            print(f'{output_lb.shape=} {cs_to_use.shape=}')
            if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2 and (idx < n_subnets-1):
                assert len(output_lb) == len(output_ub) == 1
                output_lb = output_lb.view(subnet_params.output_shape)
                output_ub = output_ub.view(subnet_params.output_shape)

            # print(f'{output_lb=}')
            print(f'{output_lb.shape=}')
            
            # update bounds
            assert self.input_output_bounds[idx].over_output is None
            self.input_output_bounds[idx] = self.input_output_bounds[idx]._replace(over_output=(output_lb.clone(), output_ub.clone()))
            assert self.input_output_bounds[idx].over_output is not None
            
            if self.input_output_bounds.get(idx + 1, False) is None:
                self.input_output_bounds[idx + 1] = InOutBounds(over_input=(output_lb.clone(), output_ub.clone()))
        
        
        return self.input_output_bounds[n_subnets-1].over_output
    
    @beartype
    def _under_estimate(self, subnet_idx: int, verify_batch: int, verify_timeout: int | float, tighten_batch: int) -> None:
        # release memory
        gc_cuda()
            
        subnet_input_outputs = self.input_output_bounds[subnet_idx]
        subnet_params = self.sub_networks[subnet_idx]
        # network
        if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
            network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
        else:
            # network = torch.nn.Sequential(torch.nn.Identity(), subnet_params.network)
            network = subnet_params.network
        network = network.to(self.device)
        network.eval()
        
        if subnet_input_outputs.under_output is None:
            # TODO: sampling from input to split layer
            min_i = optimize_dnn(network, subnet_input_outputs.over_input[0], subnet_input_outputs.over_input[1], is_min=True).view(subnet_params.output_shape)
            max_i = optimize_dnn(network, subnet_input_outputs.over_input[0], subnet_input_outputs.over_input[1], is_min=False).view(subnet_params.output_shape)
            assert torch.all(min_i <= max_i), f'{min_i=} {max_i=}'
            self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(min_i.clone(), max_i.clone()))
            subnet_input_outputs = self.input_output_bounds[subnet_idx]
            if os.environ.get("NEURALSAT_LOG_SUBVERIFIER"):
                print(f'Setup under output {subnet_idx=}:')
                print(f'\t- Lower: {subnet_input_outputs.under_output[0].detach().cpu().numpy().tolist()}')
                print(f'\t- Upper: {subnet_input_outputs.under_output[1].detach().cpu().numpy().tolist()}')

        candidate_neurons = self.extract_candidate(
            subnet_idx=subnet_idx, 
            batch=tighten_batch,
        )
        
        verifier = self._setup_subnet_verifier(
            subnet_idx=subnet_idx,
            objective=None, 
            batch=verify_batch,
        )
        
        verified_candidates, attack_samples = verify_dnf_pairs(
            verifier=verifier,
            input_lower=subnet_input_outputs.over_input[0],
            input_upper=subnet_input_outputs.over_input[1],
            n_outputs=math.prod(subnet_input_outputs.under_output[0].shape[1:]),
            candidate_neurons=candidate_neurons,
            batch=Settings.verify_candidate_batch,
            timeout=verify_timeout,
        )
        
        print(f'[+] verify_dnf_pairs:', get_used_gpu_memory(), 'MB')
        
        # print(f'{len(verified_candidates)=}')
        # FIXME: handle attack
        # if len(attack_samples) and subnet_idx==0:
        #     print(f'Attacked {len(attack_samples)=} {subnet_input_outputs.over_input[0].sum().item()=} {subnet_input_outputs.over_input[1].sum().item()=}')
        #     assert torch.all(attack_samples <= subnet_input_outputs.over_input[1])
        #     assert torch.all(attack_samples >= subnet_input_outputs.over_input[0])
        #     attack_outputs = network(attack_samples)
        #     # print(f'{attack_outputs=}')
        #     attack_outputs_min = attack_outputs.amin(dim=0)
        #     attack_outputs_max = attack_outputs.amax(dim=0)
        #     # print(f'{attack_outputs_min=}')
        #     # print(f'{attack_outputs_max=}')
        #     old_under_output_lower, old_under_output_upper = subnet_input_outputs.under_output
        #     # print(f'{old_under_output_lower=}')
        #     # print(f'{old_under_output_upper=}')
        #     indices_min = attack_outputs_min < old_under_output_lower
        #     indices_max = attack_outputs_max > old_under_output_upper
        #     print(f'{indices_min=}')
        #     print(f'Update under_output lower {subnet_idx=}:')
        #     for idx, iv in enumerate(indices_min.flatten()):
        #         if iv:
        #             print(f'[{idx}] {old_under_output_lower.flatten()[idx]} => {attack_outputs_min.flatten()[idx]}')
        #     print()
        #     print(f'Update under_output upper {subnet_idx=}:')
        #     for idx, iv in enumerate(indices_max.flatten()):
        #         if iv:
        #             print(f'[{idx}] {old_under_output_upper.flatten()[idx]} => {attack_outputs_max.flatten()[idx]}')
        #     # exit()
        #     new_under_output_lower = torch.where(indices_min, attack_outputs_min, old_under_output_lower)
        #     new_under_output_upper = torch.where(indices_max, attack_outputs_max, old_under_output_upper)
            
        #     assert torch.all(new_under_output_lower <= new_under_output_upper)
        #     self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(new_under_output_lower.clone(), new_under_output_upper.clone()))
        #     subnet_input_outputs = self.input_output_bounds[subnet_idx]
        
        new_over_output_lower = subnet_input_outputs.over_output[0].clone()
        new_over_output_upper = subnet_input_outputs.over_output[1].clone()
        
        # print(f'{new_over_output_lower.shape=}')
        improved_neuron_indices = []
        # improved_lower, improved_upper = 0.0, 0.0
        for (neuron_idx, neuron_bound, neuron_direction) in verified_candidates:
            assert neuron_direction in ['lt', 'gt']
            if neuron_direction == 'lt': # lower bound
                # assert new_over_output_lower[0][neuron_idx] <= neuron_bound, f'{neuron_idx=} {new_over_output_lower[0][neuron_idx]=} <= {neuron_bound=}'
                # improved_lower += abs(new_over_output_lower[0][neuron_idx] - neuron_bound)
                new_over_output_lower[0].flatten()[neuron_idx] = max(new_over_output_lower[0].flatten()[neuron_idx], neuron_bound)
            else: # upper bound
                # assert new_over_output_upper[0][neuron_idx] >= neuron_bound, f'{neuron_idx=} {new_over_output_upper[0][neuron_idx]=} >= {neuron_bound=}'
                # improved_upper += abs(new_over_output_upper[0][neuron_idx] - neuron_bound)
                new_over_output_upper[0].flatten()[neuron_idx] = min(new_over_output_upper[0].flatten()[neuron_idx], neuron_bound)
            improved_neuron_indices.append(neuron_idx)
        
        for neuron_idx in list(sorted(set(improved_neuron_indices))):
            print(
                f'Tightened {neuron_idx=:3d}:\t'
                f'[{subnet_input_outputs.over_output[0][0].flatten()[neuron_idx]:.04f}, {subnet_input_outputs.over_output[1][0].flatten()[neuron_idx]:.04f}]\t'
                f'=>\t[{new_over_output_lower[0].flatten()[neuron_idx]:.04f}, {new_over_output_upper[0].flatten()[neuron_idx]:.04f}]'
            )
        
        # update bounds
        self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(over_output=(new_over_output_lower.clone(), new_over_output_upper.clone()))
        self.input_output_bounds[subnet_idx+1] = self.input_output_bounds[subnet_idx+1]._replace(over_input=(new_over_output_lower.clone(), new_over_output_upper.clone()))
            
    @beartype
    def extract_candidate(self, subnet_idx: int, batch: int, eps: float=0.0) -> list:
        
        # TODO: remove
        random.seed(36)
        if (subnet_idx not in self.tightening_candidates) or len(self.tightening_candidates[subnet_idx]) < batch // 4:
            subnet_input_outputs = self.input_output_bounds[subnet_idx]
            under_output = subnet_input_outputs.under_output
            over_output = subnet_input_outputs.over_output
            
            assert over_output is not None, f'Unsupported {over_output=}'
            assert under_output is not None, f'Unsupported {under_output=}'
            
            best_interm_min = under_output[0].flatten()
            best_interm_max = under_output[1].flatten()
            
            over_output_min = over_output[0].flatten()
            over_output_max = over_output[1].flatten()
            assert torch.all(best_interm_min <= best_interm_max)
            assert torch.all(over_output_min <= over_output_max)
            
            assert torch.all(over_output_min <= best_interm_min), f'{over_output_min=} {best_interm_min=} {over_output_min < best_interm_min}'
            assert torch.all(over_output_max >= best_interm_max), f'{over_output_max=} {best_interm_max=} {over_output_max < best_interm_max}'

            candidates = []
            for i in range(len(best_interm_min)):
                if (over_output_min[i] * over_output_max[i] < 0) or 1: # unstable neurons
                    candidates.append([(i, (2*best_interm_min[i] + over_output_min[i]) / 3 + eps, 'lt')])
                    candidates.append([(i, (2*best_interm_max[i] + over_output_max[i]) / 3 - eps, 'gt')])
                    # candidates.append([(i, (best_interm_min[i] + over_output_min[i]) / 2 + eps, 'lt')])
                    # candidates.append([(i, (best_interm_max[i] + over_output_max[i]) / 2 - eps, 'gt')])
                    if len(best_interm_min) < 500:
                        print(f'[{over_output_min[i]:.04f}, {over_output_max[i]:.04f}],\t[{best_interm_min[i]:.04f}, {best_interm_max[i]:.04f}]\t=>\t{candidates[-2:]}')
                    
            random.shuffle(candidates)
            self.tightening_candidates[subnet_idx] = candidates
        
        assert len(self.tightening_candidates[subnet_idx]) > 0
        candidates = self.tightening_candidates[subnet_idx][:batch]
        print(f'Extracted {len(candidates)=} from {len(self.tightening_candidates[subnet_idx])} candidates')
        self.tightening_candidates[subnet_idx] = self.tightening_candidates[subnet_idx][len(candidates):]
    
        return candidates
        
    # @beartype
    def verify_one(self, objective: typing.Any, verify_batch: int, tighten_batch: int, timeout: int | float = 3600, use_extra: bool = True, interm_batch: int = 200) -> str:
        self.iteration = 0
        self.start_time = time.time()
        self.reset()
        
        lb, _ = self._init_interm_bounds(objective, use_extra=use_extra, method=Settings.init_abstraction_method, interm_batch=interm_batch)
        print('[+] _init_interm_bounds:', lb.flatten())
        print('[+] _init_interm_bounds:', get_used_gpu_memory(), 'MB')
        
        # lb, _ = self._init_interm_bounds(objective, use_extra=use_extra, method='backward', interm_batch=interm_batch)
        # assert all([i == 0.0 for i in objective.rhs.flatten()]), f'{objective.rhs=}'
        stop_criterion_func = stop_criterion_batch_any(objective.rhs.to(lb))
        
        if stop_criterion_func(lb).all().item():
            return ReturnStatus.UNSAT
        
        if Settings.use_decompose_incomplete:
            return ReturnStatus.UNKNOWN
            
        # release memory
        gc_cuda()
        
        n_subnets = len(self.sub_networks)
        while True:
            print('[+] Iteration:', self.iteration)
            for subnet_idx in range(n_subnets-1):
                if tighten_batch and self.iteration > 0:
                    self._under_estimate(
                        subnet_idx=subnet_idx,
                        verify_batch=verify_batch,
                        verify_timeout=Settings.verify_interm_timeout,
                        tighten_batch=tighten_batch,
                    )
                    print(f'[+] _under_estimate {subnet_idx=}:', get_used_gpu_memory(), 'MB')
                    
                    if os.environ.get("NEURALSAT_LOG_SUBVERIFIER"):
                        print('###################')
                        print(f'Bounds {subnet_idx=}')
                        subnet_input_outputs = self.input_output_bounds[subnet_idx]
                        print(f'Over output: {subnet_input_outputs.over_output[0].shape}')
                        print(f'\t- Lower: {subnet_input_outputs.over_output[0].detach().cpu().numpy().tolist()}')
                        print(f'\t- Upper: {subnet_input_outputs.over_output[1].detach().cpu().numpy().tolist()}')
                        print(f'Under output:')
                        print(f'\t- Lower: {subnet_input_outputs.under_output[0].detach().cpu().numpy().tolist()}')
                        print(f'\t- Upper: {subnet_input_outputs.under_output[1].detach().cpu().numpy().tolist()}')
                    
                
            # last subnet
            
            # subnet_params = self.sub_networks[n_subnets-1]
            # network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
            # x = torch.ones(subnet_params.input_shape)
            # y = network(x)
            # model_scripted = torch.jit.script(network) # Export to TorchScript
            # model_scripted.save('test_last.pth')
            # print(y)
            # exit()
            
            status = self._verify_subnet(
                subnet_idx=n_subnets-1, 
                objective=copy.deepcopy(objective), 
                verify_batch=verify_batch,
                timeout=min(timeout, Settings.verify_last_timeout),
            )
            print(f'[+] verify last:', get_used_gpu_memory(), 'MB')
            # exit()
            
            if status not in [ReturnStatus.TIMEOUT, ReturnStatus.SAT]:
                return status
            
            self.iteration += 1
            if time.time() - self.start_time > timeout:
                return status
    
    # @beartype
    def decompositional_verify(self, objectives: DnfObjectives, timeout: int |float = 3600, batch: int = 500, interm_batch: int = 200) -> str:
        # decomposition
        # step 1: Extract Tokenizer ViT (if needed) + Convert objectives
        new_network, new_objectives, new_input_shape = self.extract_tokenizer(
            net=self.net, 
            objectives=copy.deepcopy(objectives), 
            input_shape=self.input_shape,
        )
        gc_cuda()
        
        # step 2: Extract Prefix + Suffix of Classifier     
        self.decompose_network(
            net=new_network, 
            input_shape=new_input_shape,
            min_layer=self.min_layer,
        )
        self.reset()
        
        # step 3: Verify each objecive
        while len(new_objectives):
            objective = new_objectives.pop(1)
            status = self.verify_one(
                objective=objective,
                verify_batch=batch, # batch size of sub-verifiers
                tighten_batch=Settings.verify_candidate_num, # number of tightening candidates
                timeout=timeout,
                use_extra=True, # FIXME: only work with 2 subnets
                interm_batch=interm_batch,
            )
            if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT, ReturnStatus.UNKNOWN]:
                return status 
            if status == ReturnStatus.UNSAT:
                continue
            raise ValueError(status)
        
            
        return status
    
    @beartype
    def original_verify(self, objectives: DnfObjectives, timeout: int | float = 3600, batch: int = 500) -> str:
        self.iteration = 0
        verifier = Verifier(
            net=self.net,
            input_shape=self.input_shape,
            batch=batch,
            device=self.device,
        )
        status = verifier.verify(
            dnf_objectives=copy.deepcopy(objectives), 
            timeout=timeout, 
            force_split=None,
        )
        
        return status
        
    @beartype
    def verify(self, objectives: DnfObjectives, force_decomposition: bool = False, timeout: int | float = 3600, batch: int = 500) -> str:
        if force_decomposition:
            return self.decompositional_verify(
                objectives=objectives,
                timeout=timeout,
                batch=batch,
            )
            
        try:
            # Try verify entire network
            return self.original_verify(
                objectives=objectives,
                timeout=timeout,
                batch=batch,
            )
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                print('[Debug] Switch to decompositional verification due to OOM')
                return self.decompositional_verify(objectives)
            else:
                traceback.print_exc()
                raise NotImplementedError()
        except SystemExit:
            exit()
        except:
            traceback.print_exc()
            raise NotImplementedError()


    from .utils import (
        extract_tokenizer, 
        decompose_network,
        _setup_subnet_verifier,
        new_input,
        _verify_subnet,
    )

def formatted_print(a, b, name):
    if a.numel() > 5:
        a = ', '.join([f'{_.item():.03f}' for _ in a.flatten()])
        b = ', '.join([f'{_.item():.03f}' for _ in b.flatten()])
    print(f'[{name}] lb:', a)
    print(f'[{name}] ub:', b)
    print()
    

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def test1():    
    
    seed = int(sys.argv[1])
    print(f'{seed=}')
    torch.manual_seed(seed)
    
    from example.scripts.test_function import extract_instance
    logger.setLevel(logging.DEBUG)
    
    Settings.setup(None)
    print(Settings)
    
    model_name = 'resnet36B'
    benchmark_dir = 'example/generated_benchmark/resnet_no_bn'
    eps = 0.0005
    
    instances = [l.split(',')[:-1] for l in open(f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/instances.csv').read().strip().split('\n')]
    
    net_path = f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/{instances[seed][0]}'
    vnnlib_path = f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/{instances[seed][1]}'
    
    device = 'cuda'
  
    print(f'{net_path=}')
    print(f'{vnnlib_path=}')
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)

    # TODO: remove
    pytorch_model = resnet_toy()
    
    # Apply fusion
    # print(pytorch_model)
    # fuse_model(pytorch_model)
    # exit()


    pytorch_model.eval()
    print(pytorch_model)
    print(get_model_params(pytorch_model))
    # exit()
     
    verifier = DecompositionalVerifier(
        net=pytorch_model,
        input_shape=input_shape,
        min_layer=6,
        device=device,
    )    

    tic = time.time()
    
    oracle_verify = verifier.decompositional_verify
    # oracle_verify = verifier.original_verify
    
    status = oracle_verify(
        objectives=dnf_objectives, 
        timeout=10000,
        batch=11,
    )
    
    print(status, time.time() - tic)
    
    
    
def test2():    
    
    seed = int(sys.argv[1])
    print(f'{seed=}')
    torch.manual_seed(seed)
    
    from example.scripts.test_function import extract_instance
    logger.setLevel(logging.DEBUG)
    
    Settings.setup(None)
    print(Settings)
    
    model_name = 'cifar10_3'
    eps = 0.0005
    benchmark_dir = f'example/generated_benchmark/{model_name}/eps_{eps:.06f}'

    
    instances = [l.split(',')[:-1] for l in open(f'{benchmark_dir}/instances.csv').read().strip().split('\n')]
    
    net_path = f'{benchmark_dir}/{instances[seed][0]}'
    vnnlib_path = f'{benchmark_dir}/{instances[seed][1]}'
    
    device = 'cuda'
    # device = 'cpu'
  
    print(f'{net_path=}')
    print(f'{vnnlib_path=}')
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
    pytorch_model.eval()
    print(pytorch_model)
    print(get_model_params(pytorch_model))
    # exit()
     
    verifier = DecompositionalVerifier(
        net=pytorch_model,
        input_shape=input_shape,
        min_layer=1,
        device=device,
    )    

    tic = time.time()
    
    oracle_verify = verifier.decompositional_verify
    # oracle_verify = verifier.original_verify
    
    status = oracle_verify(
        objectives=dnf_objectives, 
        timeout=10000,
        batch=50,
    )
    
    print(status, time.time() - tic)

if __name__ == "__main__":
    test2()