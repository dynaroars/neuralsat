from collections import namedtuple
from beartype import beartype
import torch.nn as nn
import traceback
import typing
import torch
import time
import copy
import tqdm
import math
import os


from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from helper.misc.result import ReturnStatus, CoefficientMatrix
from helper.spec.objective import DnfObjectives

from tightener.utils import optimize_dnn, verify_dnf_pairs

from .utils import get_used_gpu_memory
from .verifier import Verifier

from abstractor.auto_LiRPA import PerturbationLpNorm, BoundedTensor, BoundedModule
from abstractor.auto_LiRPA.utils import stop_criterion_batch_any
from abstractor.params import get_initialize_opt_params

from setting import Settings

InOutBounds = namedtuple('InputOutputBounds', ['under_input', 'under_output', 'over_input', 'over_output'], defaults=(None,) * 4)

def redundant_compute_bounds(net, input_lowers, input_uppers, cs, method='backward', device='cpu'):
    # assert method in ['backward', 'crown-optimized']
    print(f'{device=}')
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(input_lowers <= input_uppers)
    new_x = BoundedTensor(input_lowers, PerturbationLpNorm(x_L=input_lowers.clone(), x_U=input_uppers.clone())).to(device)
    abstract = None
    abstract = BoundedModule(
        model=copy.deepcopy(net).to(device), 
        global_input=torch.zeros(input_lowers.shape, device=device),
        bound_opts={'conv_mode': 'patches', 'verbosity': 0, **Settings.verify_extra_opts},
        device=device,
        verbose=False,
    )
    abstract.eval()
    abstract(new_x)
    print(f'{Settings.share_alphas=}')
    l, u, aux_reference_bounds = abstract.init_alpha(
        x=(new_x,), 
        share_alphas=Settings.share_alphas, 
        c=cs.to(device), 
        bound_lower=True,
        bound_upper=True,
    )

    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(l <= u), f'{(l > u).sum()} {l[l > u]} {u[l > u]}'
        
    if method != 'crown-optimized':
        del abstract
        print(l)
        print(u)
        l = l.detach().cpu()
        u = u.detach().cpu()
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
    
    _, u = abstract.compute_bounds(
        x=(new_x,), 
        C=cs,
        method='crown-optimized',
        aux_reference_bounds=aux_reference_bounds, 
        bound_lower=False, 
        bound_upper=True, 
    )
    
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(l <= u)
    del abstract
    return (l, u), None
  

class DecompositionalVerifier:
    
    @beartype
    def __init__(self, net: nn.Module, input_shape: tuple, device: str = 'cpu') -> None:
        self.net = net.to(device) # pytorch model
        self.device = device
        self.input_shape = input_shape
        
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
        perturbed = (diff > 0).int().sum().item()
        input_dim = input_lowers.numel()
        print(f'{cs=} {eps=} {perturbed=} {input_dim=}')
        
        if not output_sequential:
            return redundant_compute_bounds(
                net=abstractor.pytorch_model,
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                cs=cs.to(self.device),
                method=method,
                device=self.device,
            )
            
        assert len(input_lowers) == 1, f'Only support batch=1: {len(input_lowers)=}'
        n_outputs = abstractor.net(input_lowers.to(self.device)).flatten(1).shape[1]
        
        output_lowers, output_uppers = [], []
        output_coeffs = []
        
        
        pbar = tqdm.tqdm(range(0, n_outputs, output_batch), desc=f'Compute bounds: {output_batch=} {method=} {eps=:.06f}')
        for i_, i in enumerate(pbar):
            indices = torch.arange(i, min(i+output_batch, n_outputs))
            ci = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lowers)
            # TODO: reuse optimized slopes
            (lb, ub), coeffs = abstractor.compute_bounds(
                input_lowers=input_lowers.to(self.device),
                input_uppers=input_uppers.to(self.device),
                cs=ci.to(self.device),
                method=method,
            )
            if i_ % 100 == 0:
                print(f'[+] abstractor.compute_bounds {method=}:', get_used_gpu_memory(), 'MB')
                
            gc_cuda()
            
            if os.environ.get('NEURALSAT_ASSERT'):
                assert torch.all(lb <= ub + 1e-5), f'{(lb > ub).sum()} {lb[lb > ub]} {ub[lb > ub]}'
            output_lowers.append(lb.clone())
            output_uppers.append(ub.clone())
            if coeffs:
                output_coeffs.append(coeffs)
            
        output_lowers = torch.cat(output_lowers, dim=-1)
        output_uppers = torch.cat(output_uppers, dim=-1)
        
        if output_coeffs:
            output_coeffs = CoefficientMatrix(
                lA=torch.cat([c.lA for c in output_coeffs], dim=0),
                uA=torch.cat([c.uA for c in output_coeffs], dim=0),
                lbias=torch.cat([c.lbias for c in output_coeffs], dim=-1),
                ubias=torch.cat([c.ubias for c in output_coeffs], dim=-1),
            )
        else:
            output_coeffs = None
            
        return (output_lowers, output_uppers), output_coeffs
        
    @beartype
    def _init_interm_bounds(self, objective: typing.Any, use_extra: bool = True, method='crown-optimized', interm_batch: int = 200) -> tuple:
        print(f'Init interm bounds {method=}')
        # input 
        input_shape = self.sub_networks[0].input_shape
        input_lb_0 = objective.lower_bounds.view(input_shape).to(self.device).detach().cpu()
        input_ub_0 = objective.upper_bounds.view(input_shape).to(self.device).detach().cpu()
        if self.input_output_bounds[0] is None:
            self.input_output_bounds[0] = InOutBounds(over_input=(input_lb_0.clone(), input_ub_0.clone()))
        
        n_subnets = len(self.sub_networks)
        # try computing bounds with sub-networks
        for idx in range(n_subnets):
            # if (idx != n_subnets-1) and (self.input_output_bounds[idx].over_output is not None):
            #     # no need to re-init for different objectives, only need to update last over_output due to cs
            #     # TODO: recheck input property
            #     print(f'Reuse computed bounds subnet {idx=}')
            #     continue
            verifier = self._setup_subnet_verifier(idx)
            print(verifier.net)
            print(f'Processing subnet {idx+1}/{n_subnets} {method=}')
            cs_to_use = objective.cs.clone()
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
            gc_cuda()
            
            if idx == 0 and use_extra and output_coeffs is not None and len(output_coeffs): 
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

            print(f'{output_lb.shape=}')
            
            # update bounds
            assert self.input_output_bounds[idx].over_output is None
            self.input_output_bounds[idx] = self.input_output_bounds[idx]._replace(over_output=(output_lb.clone(), output_ub.clone()))
            assert self.input_output_bounds[idx].over_output is not None
            
            if self.input_output_bounds.get(idx + 1, False) is None:
                self.input_output_bounds[idx + 1] = InOutBounds(over_input=(output_lb.clone(), output_ub.clone()))
        
        for i in range(n_subnets):
            print(i, 'input ', self.input_output_bounds[i].over_input[0].device, self.input_output_bounds[i].over_input[1].device)
            print(i, 'output', self.input_output_bounds[i].over_output[0].device, self.input_output_bounds[i].over_output[1].device)
        
        return self.input_output_bounds[n_subnets-1].over_output
    
    @beartype
    def _under_estimate(self, subnet_idx: int, verify_batch: int, verify_timeout: int | float) -> None:
        gc_cuda()
            
        subnet_input_outputs = self.input_output_bounds[subnet_idx]
        subnet_params = self.sub_networks[subnet_idx]
        # network
        if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
            network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
        else:
            network = subnet_params.network
        network = network.to(self.device)
        network.eval()
        
        if subnet_input_outputs.under_output is None:
            min_i = optimize_dnn(network, subnet_input_outputs.over_input[0].to(self.device), subnet_input_outputs.over_input[1].to(self.device), is_min=True).view(subnet_params.output_shape)
            max_i = optimize_dnn(network, subnet_input_outputs.over_input[0].to(self.device), subnet_input_outputs.over_input[1].to(self.device), is_min=False).view(subnet_params.output_shape)
            assert torch.all(min_i <= max_i), f'{min_i=} {max_i=}'
            self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(min_i.clone(), max_i.clone()))
            subnet_input_outputs = self.input_output_bounds[subnet_idx]

        candidate_neurons = self.extract_candidate(
            subnet_idx=subnet_idx, 
            num_candidate=Settings.verify_candidate_num, # number of tightening candidates,
            interpolate_factor=Settings.verify_interpolate_factor
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
        # TODO: handle attack
        
        new_over_output_lower = subnet_input_outputs.over_output[0].clone()
        new_over_output_upper = subnet_input_outputs.over_output[1].clone()
        
        improved_neuron_indices = []
        for (neuron_idx, neuron_bound, neuron_direction) in verified_candidates:
            assert neuron_direction in ['lt', 'gt']
            if neuron_direction == 'lt': # lower bound
                new_over_output_lower[0].flatten()[neuron_idx] = max(new_over_output_lower[0].flatten()[neuron_idx], neuron_bound)
            else: # upper bound
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
    def extract_candidate(self, subnet_idx: int, num_candidate: int, interpolate_factor: float, eps: float=0.0) -> list:
        
        print(f'[+] Generating candidates using {num_candidate=} {interpolate_factor=}')
        if (subnet_idx not in self.tightening_candidates) or len(self.tightening_candidates[subnet_idx]) < num_candidate // 8:
            subnet_input_outputs = self.input_output_bounds[subnet_idx]
            under_output = subnet_input_outputs.under_output
            over_output = subnet_input_outputs.over_output
            
            assert over_output is not None, f'Unsupported {over_output=}'
            assert under_output is not None, f'Unsupported {under_output=}'
            
            best_interm_min = under_output[0].flatten()
            best_interm_max = under_output[1].flatten()
            
            over_output_min = over_output[0].flatten()
            over_output_max = over_output[1].flatten()
            
            unsorted_candidates = []
            for i in range(len(best_interm_min)):
                score = best_interm_max[i] - best_interm_min[i]
                if score <= 1e-4:
                    continue
                
                candidate_lower = (1 - interpolate_factor) * best_interm_min[i] + interpolate_factor * over_output_min[i] + eps
                candidate_upper = (1 - interpolate_factor) * best_interm_max[i] + interpolate_factor * over_output_max[i] - eps
                if os.environ.get('NEURALSAT_ASSERT'):
                    assert over_output_min[i] <= candidate_lower <= best_interm_min[i] <= best_interm_max[i] <= candidate_upper <= over_output_max[i], f'{over_output_min[i]=}\n{candidate_lower=}\n{best_interm_min[i]=}\n{best_interm_max[i]=}\n{candidate_upper=}\n{over_output_max[i]=}'
                
                unsorted_candidates.append((score, [(i, candidate_lower, 'lt')]))
                unsorted_candidates.append((score, [(i, candidate_upper, 'gt')]))
                
                if len(best_interm_min) < 500:
                    print(f'[{over_output_min[i]:.04f}, {over_output_max[i]:.04f}],\t[{best_interm_min[i]:.04f}, {best_interm_max[i]:.04f}]\t=>\t{unsorted_candidates[-2:]}')
                    
            unsorted_candidates = sorted(unsorted_candidates, key=lambda x: x[0], reverse=True)
            candidates = [_[1] for _ in unsorted_candidates]
            self.tightening_candidates[subnet_idx] = candidates
        
        assert len(self.tightening_candidates[subnet_idx]) > 0
        candidates = self.tightening_candidates[subnet_idx][:num_candidate]
        print(f'Extracted {len(candidates)=} from {len(self.tightening_candidates[subnet_idx])} candidates')
        self.tightening_candidates[subnet_idx] = self.tightening_candidates[subnet_idx][len(candidates):]
    
        return candidates
        
    # @beartype
    def verify_one(self, objective: typing.Any, verify_batch: int, timeout: int | float = 3600, use_extra: bool = True, interm_batch: int = 200) -> str:
        self.iteration = 0
        self.start_time = time.time()
        self.reset()
        
        lb, _ = self._init_interm_bounds(objective, use_extra=use_extra, method=Settings.init_abstraction_method, interm_batch=interm_batch)
        print('[+] _init_interm_bounds:', lb.flatten())
        print('[+] _init_interm_bounds:', get_used_gpu_memory(), 'MB')
        
        stop_criterion_func = stop_criterion_batch_any(objective.rhs.to(lb))
        
        if stop_criterion_func(lb).all().item():
            return ReturnStatus.UNSAT
        
        if Settings.use_decompose_incomplete:
            return ReturnStatus.UNKNOWN
            
        # release memory
        gc_cuda()
        
        n_subnets = len(self.sub_networks)
        while self.iteration < Settings.verify_max_iteration:
            print('[+] Iteration:', self.iteration)
            remain_timeout = timeout - (time.time() - self.start_time)
            if remain_timeout <= 0:
                return ReturnStatus.TIMEOUT
            
            for subnet_idx in range(n_subnets-1):
                if self.iteration > 0:
                    self._under_estimate(
                        subnet_idx=subnet_idx,
                        verify_batch=verify_batch,
                        verify_timeout=Settings.verify_interm_timeout,
                    )
                    print(f'[+] _under_estimate {subnet_idx=}:', get_used_gpu_memory(), 'MB')
                
            # last subnet
            remain_timeout = timeout - (time.time() - self.start_time)
            if remain_timeout <= 0:
                return ReturnStatus.TIMEOUT
            
            status = self._verify_subnet(
                subnet_idx=n_subnets-1, 
                objective=copy.deepcopy(objective), 
                verify_batch=verify_batch,
                timeout=Settings.verify_last_timeout,
            )
            print(f'[+] verify last:', get_used_gpu_memory(), 'MB')
            
            if status not in [ReturnStatus.TIMEOUT, ReturnStatus.SAT, ReturnStatus.UNKNOWN]:
                return status
            
            self.iteration += 1
            
            if time.time() - self.start_time > timeout:
                return status
        return status
    
    # @beartype
    def decompositional_verify(self, objectives: DnfObjectives, timeout: int |float = 3600, 
                               batch: int = 500, interm_batch: int = 200) -> str:
        # decomposition
        gc_cuda()
         
        self.decompose_network(
            net=self.net, 
            input_shape=self.input_shape,
            objectives=copy.deepcopy(objectives),
        )
        self.reset()
        
        while len(objectives):
            objective = objectives.pop(1)
            status = self.verify_one(
                objective=objective,
                verify_batch=batch, # batch size of sub-verifiers
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
                interm_batch=Settings.sequential_batch,
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


    from .dec_utils import (
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
