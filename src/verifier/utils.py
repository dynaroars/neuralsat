from __future__ import annotations
from collections import defaultdict
from beartype import beartype
import numpy as np
import random
import typing
import torch
import time
import copy
import os

if typing.TYPE_CHECKING:
    import verifier
    
from heuristic.restart_heuristics import get_restart_strategy, HIDDEN_SPLIT_RESTART_STRATEGIES
from heuristic.util import compute_masks, _history_to_conflict_clause
from heuristic.decision_heuristics import DecisionHeuristic
from heuristic.domains_list import DomainsList

from tightener.cpu_tightener import MILPTightener
from tightener.gpu_tightener import GPUTightener

from attacker.pgd_attack.general import general_attack
from attacker.mip_attack import MIPAttacker
from attacker.attacker import Attacker

from abstractor.abstractor import NetworkAbstractor

from helper.misc.result import AbstractResults, ReturnStatus
from helper.proof.create_aptp import create_aptp
from helper.spec.objective import DnfObjectives
from helper.misc.check import check_solution
from helper.misc.logger import logger


from setting import Settings


def get_used_gpu_memory(return_percentage: bool = False):
    if not torch.cuda.is_available():
        if return_percentage:
            return 0, 0
        return 0
    
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    torch.cuda.empty_cache()
    if return_percentage:
        return mem_used_MB, (total - free) / total * 100
    return mem_used_MB
    


def _check_invoke_mip_presolving(self):
    if not Settings.use_mip_verify:
        return False
    count_relu = 0
    for layer in self.net.children():
        if not isinstance(layer, (torch.nn.Linear, torch.nn.ReLU, torch.nn.Flatten)):
            print('[!] Found unsupported layer:', type(layer))
            return False
        if isinstance(layer, torch.nn.ReLU):
            count_relu += 1
    
    if os.environ.get('NEURALSAT_DEBUG'):            
        print(f'[+] _check_invoke_mip_presolving {count_relu=}')
        
    if count_relu > Settings.mip_verify_threshold:
        return False
    return True

@beartype
def _prune_objective(self: verifier.verifier.Verifier, objective: typing.Any) -> typing.Any:
    assert self.domains_list is not None
    
    all_remaining_ids = torch.unique(self.domains_list.all_objective_ids.data)
    if not len(all_remaining_ids):
        return objective
    
    # remaining
    indices = torch.tensor([idx for idx, val in enumerate(objective.ids) if val in all_remaining_ids])
    
    # pruning
    objective.ids = objective.ids[indices]
    
    objective.lower_bounds = objective.lower_bounds[indices]
    objective.upper_bounds = objective.upper_bounds[indices]
    
    objective.lower_bounds_f64 = objective.lower_bounds_f64[indices]
    objective.upper_bounds_f64 = objective.upper_bounds_f64[indices]
    
    objective.cs = objective.cs[indices]
    objective.rhs = objective.rhs[indices]
    
    objective.cs_f64 = objective.cs_f64[indices]
    objective.rhs_f64 = objective.rhs_f64[indices]
    
    # assert torch.equal(objective.ids, all_remaining_ids)
    return objective

@beartype
def _prune_domains(domain_params: AbstractResults, remaining_indices: torch.Tensor) -> AbstractResults:
    return AbstractResults(**{
        'objective_ids': domain_params.objective_ids[remaining_indices], 
        'output_lbs': domain_params.output_lbs[remaining_indices] if domain_params.output_lbs is not None else None, 
        'masks': {k: v[remaining_indices] for k, v in domain_params.masks.items()} if domain_params.masks is not None else None, 
        'lAs': {k: v[remaining_indices] for k, v in domain_params.lAs.items()}, 
        'histories': [domain_params.histories[_] for _ in remaining_indices], 
        'lower_bounds': {k: v[remaining_indices] for k, v in domain_params.lower_bounds.items()}, 
        'upper_bounds': {k: v[remaining_indices] for k, v in domain_params.upper_bounds.items()}, 
        'input_lowers': domain_params.input_lowers[remaining_indices], 
        'input_uppers': domain_params.input_uppers[remaining_indices],
        'betas': [domain_params.betas[_] for _ in remaining_indices], 
        'cs': domain_params.cs[remaining_indices], 
        'rhs': domain_params.rhs[remaining_indices], 
        'sat_solvers': [domain_params.sat_solvers[_] for _ in remaining_indices] if domain_params.sat_solvers is not None else None, 
        'slopes': defaultdict(dict, {k: {kk: vv[:, :, remaining_indices] for kk, vv in v.items()} for k, v in domain_params.slopes.items()}), 
    })


@beartype
def _mip_attack(self: verifier.verifier.Verifier, reference_bounds: dict | None) -> tuple[bool, torch.Tensor | None]:
    if not Settings.use_attack:
        return False, None
    
    if not Settings.use_mip_attack:
        return False, None
    
    return self.mip_attacker.run(reference_bounds)
    
    
@beartype
def _preprocess(self: verifier.verifier.Verifier, objectives: typing.Any, force_split: str | None = None) -> tuple:
    # determine search algorithm
    self.refined_betas = None
    
    diff = objectives.upper_bounds - objectives.lower_bounds
    eps = diff.max().item()
    perturbed = (diff > 0).int().sum() // diff.shape[0]
    logger.info(f'[!] eps={eps:.06f}, perturbed={perturbed}')

    if force_split is not None:
        assert force_split in ['input', 'hidden']
        self.input_split = force_split == 'input'
    elif eps > Settings.input_splitting_threshold: # safety properties
        self.input_split = True
    elif np.prod(self.input_shape) <= Settings.safety_num_input_perturbed or perturbed <= Settings.safety_num_input_perturbed: # small inputs
        # if eps < Settings.hidden_splitting_threshold:
        #     self.input_split = False
        # else:
        self.input_split = True
    elif np.prod(self.input_shape) >= 100000: # large inputs, e.g., VGG16
        self.input_split = True
        
    if self.input_split: 
        return objectives, None
    
    if Settings.skip_preprocess:
        return objectives, None
    
    if (not isinstance(objectives.cs, torch.Tensor)) or (not isinstance(objectives.rhs, torch.Tensor)):
        return objectives, None
    
    if not torch.allclose(objectives.lower_bounds.mean(dim=0), objectives.lower_bounds[0], 1e-5, 1e-5):
        return objectives, None
    
    try:
        logger.info(f'[_preprocess] _init_abstractor')
        self._init_abstractor('backward' if np.prod(self.input_shape) < 100000 else 'forward', objectives, preprocess=True)
    except:
        print('[_preprocess] Failed to initialize abstractor')
        return objectives, None
    
    # prune objectives
    tmp_objective = copy.deepcopy(objectives)
    tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1] # raise errors if using beta, use full objectives instead
    tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1] # raise errors if using beta, use full objectives instead
    
    # forward
    try:
        ret = self.abstractor.initialize(tmp_objective, short_cut=True)
    except:
        print('[!] Failed to preprocess objectives')
        if os.environ.get("NEURALSAT_DEBUG"):
            import traceback
            traceback.print_exc()
            raise NotImplementedError('Failed to preprocess objectives')
        return objectives, None

    # pruning
    remaining_index = torch.where((ret.output_lbs.detach().cpu() <= tmp_objective.rhs.detach().cpu()).all(1))[0]
    objectives.lower_bounds = objectives.lower_bounds[remaining_index]
    objectives.upper_bounds = objectives.upper_bounds[remaining_index]
    objectives.cs = objectives.cs[remaining_index]
    objectives.rhs = objectives.rhs[remaining_index]
    objectives.lower_bounds_f64 = objectives.lower_bounds_f64[remaining_index]
    objectives.upper_bounds_f64 = objectives.upper_bounds_f64[remaining_index]
    objectives.cs_f64 = objectives.cs_f64[remaining_index]
    objectives.rhs_f64 = objectives.rhs_f64[remaining_index]
    objectives.ids = objectives.ids[remaining_index]
    
    if None in self.abstractor.split_points:
        # FIXME: disable restart + stabilize for now
        Settings.use_restart = False
        Settings.use_mip_tightening = False
        logger.info(f'Remain {len(objectives)} objectives')
        return objectives, None
    
    # refine
    refined_intermediate_bounds = None
    if len(objectives) and (Settings.use_mip_tightening) and self.abstractor.method == 'backward':
        use_refined = not Settings.use_restart
        if any([isinstance(_, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, 
                               torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)) 
                            for _ in self.net.modules()][1:]):
            # skip refine for Conv layers for now
            use_refined = False

        if use_refined:
            logger.info(f'Refining hidden bounds for {len(objectives)} remaining objectives')
            tmp_objective = copy.deepcopy(objectives)
            tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1].to(self.device)
            tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1].to(self.device)
            
            # build solver
            tic = time.time()
            c_to_use = tmp_objective.cs.transpose(0, 1).to(self.device) if tmp_objective.cs.shape[1] == 1 else None
            self.abstractor.build_lp_solver(
                model_type='mip', 
                input_lower=tmp_objective.lower_bounds.view(self.input_shape), 
                input_upper=tmp_objective.upper_bounds.view(self.input_shape), 
                c=c_to_use,
                refine=use_refined,
                timeout=None,
                timeout_per_neuron=Settings.mip_tightening_timeout_per_neuron,
            )
            logger.debug(f'MIP: {time.time() - tic:.04f}')
            
            # forward with refinement
            refined_intermediate_bounds = self.abstractor.net.get_refined_interm_bounds()
            ret = self.abstractor.initialize(tmp_objective, reference_bounds=refined_intermediate_bounds, short_cut=True)
            
            # pruning
            remaining_index = torch.where((ret.output_lbs.detach().cpu() <= tmp_objective.rhs.detach().cpu()).all(1))[0]
            objectives.lower_bounds = objectives.lower_bounds[remaining_index]
            objectives.upper_bounds = objectives.upper_bounds[remaining_index]
            objectives.cs = objectives.cs[remaining_index]
            objectives.rhs = objectives.rhs[remaining_index]
            objectives.lower_bounds_f64 = objectives.lower_bounds_f64[remaining_index]
            objectives.upper_bounds_f64 = objectives.upper_bounds_f64[remaining_index]
            objectives.cs_f64 = objectives.cs_f64[remaining_index]
            objectives.rhs_f64 = objectives.rhs_f64[remaining_index]
            objectives.ids = objectives.ids[remaining_index]
            
            # TODO: fixme (update found betas from MIP)
            # self.refined_betas = self.abstractor.net.get_betas()
        
        # torch.save(refined_intermediate_bounds, 'refined.pt')
        
    # mip tightener
    if len(objectives) and (not self.input_split):
        # mip attacker
        if Settings.use_mip_attack:
            self.mip_attacker = MIPAttacker(
                abstractor=self.abstractor, 
                objectives=objectives, # full objectives
            )
            
        if Settings.use_mip_tightening:
            self.milp_tightener = MILPTightener(
                abstractor=self.abstractor,
                objectives=objectives,
            )
            
        elif Settings.use_gpu_tightening:
            self.gpu_tightener = GPUTightener(
                verifier=self.other,
                abstractor=self.abstractor,
            )
        
    logger.info(f'Remain {len(objectives)} objectives')
    return objectives, refined_intermediate_bounds

@beartype
def _check_timeout(self: verifier.verifier.Verifier, timeout: int | float) -> bool:
    return time.time() - self.start_time > timeout 


@beartype
def _init_abstractor(self: verifier.verifier.Verifier, method: str, objective: typing.Any, extra_opts: dict = {}, preprocess: bool = False) -> None:
    if hasattr(self, 'abstractor'):
        # del self.abstractor.net
        del self.abstractor

    self.abstractor = NetworkAbstractor(
        pytorch_model=self.net, 
        input_shape=self.input_shape, 
        method=method,
        input_split=self.input_split,
        device=self.device,
    )

    self.abstractor.setup(objective, extra_opts=extra_opts, preprocess=preprocess)
    self.abstractor.net.get_split_nodes()
    

@beartype
def _setup_restart_naive(self: verifier.verifier.Verifier, nth_restart: int, objective: typing.Any) -> None | dict:
    print(f'[+] _setup_restart_naive')
    self.num_restart = nth_restart + 1
    # TODO: select splitting method (input/hidden)
    if objective is not None:
        diff = (objective.upper_bounds - objective.lower_bounds).clone()
        eps = diff.max().item()
        perturbed = (diff > 0).int().sum()
        logger.info(f'[!] eps={eps:.06f}, perturbed={perturbed}')
        logger.info(f'[!] max_sum={objective.upper_bounds.sum():.06f}, min_sum={objective.lower_bounds.sum():.06f}')
        # if eps > 0.5:
        #     self.input_split = True
    
    if Settings.verify_splitting_strategy == 'input':
        params = {'input_split': True, 'abstract_method': 'backward', 'decision_method': 'naive', 'decision_topk': 1, 'extra_opts': Settings.verify_extra_opts}
        # params = {'input_split': True, 'abstract_method': 'crown-optimized', 'decision_method': 'naive', 'decision_topk': 1, 'extra_opts': Settings.verify_extra_opts}
        self.input_split = True
    else:
        if Settings.subverifier_decision_method == 'smart':
            params = {'input_split': False, 'abstract_method': Settings.init_abstraction_method, 'decision_method':  'smart', 'decision_topk':  5, 'extra_opts': Settings.verify_extra_opts}
        elif Settings.subverifier_decision_method == 'greedy':
            params = {'input_split': False, 'abstract_method': Settings.init_abstraction_method, 'decision_method': 'greedy', 'decision_topk': 1000, 'extra_opts': Settings.verify_extra_opts}
        else:
            raise NotImplementedError('Unknown decision method')

    logger.info(f'Params of {nth_restart+1}-th run: {params}')

    # decision heuristic
    assert params['input_split'] == self.input_split
    self.decision = DecisionHeuristic(
        input_split=params['input_split'],
        decision_topk=params['decision_topk'],
        decision_method=params['decision_method'],
    )
        
    logger.info(f'[_setup_restart_naive] _init_abstractor')
    self._init_abstractor(params['abstract_method'], objective, params['extra_opts'])
        


@beartype
def _setup_restart(self: verifier.verifier.Verifier, nth_restart: int, objective: typing.Any) -> None | dict:
    self.num_restart = nth_restart + 1
    params = get_restart_strategy(nth_restart, input_split=self.input_split)
    if params is None:
        raise NotImplementedError()
    
    if np.prod(self.input_shape) >= 100000: # large inputs, e.g., VGG16
        Settings.forward_dynamic = True
        Settings.forward_max_dim = 100
        Settings.backward_batch_size = 16
        
    logger.info(f'Params of {nth_restart+1}-th run: {params}')
    abstract_method = params['abstract_method']

    # decision heuristic
    assert params['input_split'] == self.input_split
    self.decision = DecisionHeuristic(
        input_split=params['input_split'],
        decision_topk=params['decision_topk'],
        decision_method=params['decision_method'],
    )
    
    refined_intermediate_bounds = None
    if (not self.input_split) and Settings.use_restart and self.num_restart == len(HIDDEN_SPLIT_RESTART_STRATEGIES) and Settings.use_mip_tightening:
        if 'forward' in abstract_method:
            pass
        elif not torch.allclose(objective.lower_bounds.mean(dim=0), objective.lower_bounds[0], 1e-5, 1e-5):
            pass
        elif any([isinstance(_, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)) for _ in self.net.modules()][1:]):
            # FIXME: skip refine for Conv layers
            pass
        elif None in self.abstractor.split_points:
            # skip refine for general activation layers
            pass
        else:
            logger.info(f'[_setup_restart] refine')
            self._init_abstractor('backward', objective, preprocess=True)
            
            tmp_objective = copy.deepcopy(objective)
            tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1].to(self.device)
            tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1].to(self.device)
            # tmp_objective.rhs = tmp_objective.rhs.to(self.device) # TODO: check
            c_to_use = tmp_objective.cs.transpose(0, 1).to(self.device) if tmp_objective.cs.shape[1] == 1 else None
            
            ret = self.abstractor.initialize(tmp_objective)
            self.abstractor.build_lp_solver(
                model_type='mip', 
                input_lower=tmp_objective.lower_bounds.view(self.input_shape), 
                input_upper=tmp_objective.upper_bounds.view(self.input_shape), 
                c=c_to_use,
                refine=True,
                timeout=None,
                timeout_per_neuron=Settings.mip_tightening_timeout_per_neuron,
            )
            refined_intermediate_bounds = self.abstractor.net.get_refined_interm_bounds()
            # del self.abstractor
    
    # main abstractor
    if not hasattr(self, 'abstractor') or abstract_method != self.abstractor.method:
        logger.info(f'[_setup_restart] _init_abstractor')
        self._init_abstractor(abstract_method, objective)
        
    return refined_intermediate_bounds


@beartype
def _pre_attack(self: verifier.verifier.Verifier, dnf_objectives: DnfObjectives, 
                timeout: int | float = 10.0) -> tuple[bool, torch.Tensor | None]:
    if Settings.use_attack:
        return Attacker(self.net, dnf_objectives, self.input_shape, device=self.device).run(timeout=timeout)
    return False, None
    
@beartype
def _random_idx(total_samples: int, num_samples: int, device: str = 'cpu') -> torch.Tensor:
    if num_samples >= total_samples:
        return torch.Tensor(range(total_samples)).to(device)
    return torch.Tensor(random.sample(range(total_samples), num_samples)).to(device)


@beartype
def _attack(self: verifier.verifier.Verifier, domain_params: AbstractResults, timeout: int | float,
            n_sample: int = 50, n_interval: int = 1) -> torch.Tensor | None:
    if not Settings.use_attack:
        return None
    
    if self.iteration % n_interval != 0:
        return None

    # random samples
    indices = _random_idx(len(domain_params.cs), n_sample, device=self.device).long()

    input_lowers = domain_params.input_lowers[indices][None]
    input_uppers = domain_params.input_uppers[indices][None]
    # adv_example = (input_lowers + input_uppers) / 2
    adv_example = (input_uppers - input_lowers) * torch.rand(input_lowers.shape, device=self.device) + input_lowers
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.all(adv_example <= input_uppers)
        assert torch.all(adv_example >= input_lowers)
    
    cs = domain_params.cs[indices].view(1, -1, domain_params.cs[indices].shape[-1])
    rhs = domain_params.rhs[indices].view(1, -1)
    cond = [[domain_params.cs.shape[1] for i in range(len(indices))]]
    serialized_conditions = (cs, rhs, cond)
    
    attack_images = general_attack(
        model=self.net, 
        X=adv_example, 
        data_min=input_lowers, 
        data_max=input_uppers, 
        serialized_conditions=serialized_conditions, 
        attack_iters=20, 
        num_restarts=10, 
        only_replicate_restarts=True,
        use_gama=False,
        timeout=timeout,
    )
    if (attack_images is None) and (self.iteration % (5 * n_interval) == 0) and 0:
        attack_images = general_attack(
            model=self.net, 
            X=adv_example, 
            data_min=input_lowers, 
            data_max=input_uppers, 
            serialized_conditions=serialized_conditions, 
            attack_iters=30, 
            num_restarts=10, 
            only_replicate_restarts=True,
            use_gama=True,
            timeout=timeout,
        )

    if attack_images is not None:
        for i in range(attack_images.shape[1]): # restarts
            for j in range(attack_images.shape[2]): # props
                adv = attack_images[:, i, j]
                if check_solution(self.net, adv, domain_params.cs[indices][j], domain_params.rhs[indices][j], input_lowers[:, j], input_uppers[:, j]):
                    return adv
        logger.debug("[!] Invalid counter-example")
        
    return None


@beartype
def _get_learned_conflict_clauses(self: verifier.verifier.Verifier) -> dict:
    if hasattr(self.domains_list, 'all_conflict_clauses'):
        return self.domains_list.all_conflict_clauses
    return {}


@beartype
def _check_invoke_cpu_tightening(self: verifier.verifier.Verifier, patience_limit: int = 10):
    if not hasattr(self, 'milp_tightener'):
        return False
    
    if not Settings.use_mip_tightening:
        return False
    
    if self.input_split:
        return False
    
    if self.tightening_patience < patience_limit:
        return False
    
    if len(self.domains_list) <= self.batch:
        return False
    
    if Settings.use_restart and self.num_restart < len(HIDDEN_SPLIT_RESTART_STRATEGIES):
        return False
    
    # reset counter
    self.tightening_patience = 0
    return True
    
@beartype
def _check_invoke_gpu_tightening(self: verifier.verifier.Verifier, patience_limit: int = 10):
    if not hasattr(self, 'gpu_tightener'):
        return False

    # DEBUG: stabilization during search is still buggy
    return self.iteration == 0 

    if self.input_split:
        return False
    
    if self.tightening_patience < patience_limit:
        return False
    
    if len(self.domains_list) <= self.batch:
        return False
    
    if Settings.use_restart and self.num_restart < len(HIDDEN_SPLIT_RESTART_STRATEGIES):
        return False
    
    self.tightening_patience = 0
    return True
    

@beartype
def _update_tightening_patience(self: verifier.verifier.Verifier, minimum_lowers: float, old_domains_length: int) -> None:
    current_domains_length = len(self.domains_list)
    if (minimum_lowers > self.last_minimum_lowers) or (current_domains_length <= self.batch):
        self.tightening_patience -= 1
        # self.tightening_patience = 0
    elif (current_domains_length <= old_domains_length):
        self.tightening_patience -= 1
    elif minimum_lowers == self.last_minimum_lowers:
        self.tightening_patience += 1
    else:
        self.tightening_patience += 3
        
    self.tightening_patience = max(0, self.tightening_patience)
    self.last_minimum_lowers = minimum_lowers
            
    
@beartype
def _check_full_assignment(self: verifier.verifier.Verifier, domain_params: AbstractResults) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if self.input_split:
        return None, None
    
    if domain_params.lower_bounds is None:
        return None, None
    
    # check all activation layers are ReLU 
    if None in self.abstractor.split_points: 
        # general activation
        return None, None
    
    new_masks = compute_masks(
        lower_bounds=domain_params.lower_bounds, 
        upper_bounds=domain_params.upper_bounds, 
        device='cpu',
        non_blocking=False, # FIXME: setting True makes it return wrong values
    )
    
    n_unstables = torch.stack([v.sum(dim=1) for k, v in new_masks.items()]).sum(dim=0)
    pruning_indices = torch.where(n_unstables == 0)[0]
    
    if not len(pruning_indices):
        return None, None
    
    for idx_ in pruning_indices:
        self.abstractor.build_lp_solver(
            model_type='lp', 
            input_lower=domain_params.input_lowers[idx_][None], 
            input_upper=domain_params.input_uppers[idx_][None], 
            c=domain_params.cs[idx_][None],
            refine=False,
        )

        feasible, adv = self.abstractor.solve_full_assignment(
            input_lower=domain_params.input_lowers[idx_], 
            input_upper=domain_params.input_uppers[idx_], 
            lower_bounds={k: v[idx_] for k, v in domain_params.lower_bounds.items()},
            upper_bounds={k: v[idx_] for k, v in domain_params.upper_bounds.items()},
            c=domain_params.cs[idx_],
            rhs=domain_params.rhs[idx_]
        )
        
        if feasible:
            return adv, None
        
    # save pruned domains
    [self.domains_list.all_conflict_clauses[int(domain_params.objective_ids[i])].append(domain_params.histories[i]) for i in pruning_indices]
    
    # unverified indices
    remaining_indices = torch.where(n_unstables > 0)[0]
    
    return None, remaining_indices

        
@beartype
def _save_stats(self: verifier.verifier.Verifier) -> None:
    for k, v in self._get_learned_conflict_clauses().items():
        if k not in self.all_conflict_clauses:
            self.all_conflict_clauses[k] = []
        self.all_conflict_clauses[k].extend(v)
    if hasattr(self.domains_list, 'visited'):
        self.visited += self.domains_list.visited
        
        
@beartype
def get_stats(self: verifier.verifier.Verifier) -> tuple[int, int]:
    depths = {}
    for k, v in self.all_conflict_clauses.items():
        depths[k] = max(map(lambda x: sum(len(_[0]) for _ in x.values()), v)) if len(v) else 0
    depth = max(list(depths.values())) if len(depths) else 0
    return depth, self.visited


@beartype
def _check_adv_f64(self: verifier.verifier.Verifier, adv: torch.Tensor, objective: typing.Any) -> bool:
    lower_bounds_f64 = objective.lower_bounds_f64.view(-1, *self.input_shape[1:]).to(self.device)
    upper_bounds_f64 = objective.upper_bounds_f64.view(-1, *self.input_shape[1:]).to(self.device)
    cs_f64 = objective.cs_f64.to(self.device)
    rhs_f64 = objective.rhs_f64.to(self.device)
    for i in range(len(lower_bounds_f64)):
        if check_solution(self.net, adv, cs_f64[i], rhs_f64[i], lower_bounds_f64[i:i+1], upper_bounds_f64[i:i+1]):
            return True
    return False


@beartype
def _check_adv(self: verifier.verifier.Verifier, adv: torch.Tensor, objective: typing.Any) -> bool:
    lower_bounds = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
    upper_bounds = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
    cs = objective.cs.to(self.device)
    rhs = objective.rhs.to(self.device)
    for i in range(len(lower_bounds)):
        if check_solution(self.net, adv, cs[i], rhs[i], lower_bounds[i:i+1], upper_bounds[i:i+1]):
            return True
    return False

@beartype
def get_learned_conflict_clauses(self: verifier.verifier.Verifier) -> None | dict:
    if self.all_conflict_clauses is None:
        return None
    learned_conflict_clauses = {k: [] for k in self.all_conflict_clauses}
    if hasattr(self, 'domains_list') and isinstance(self.domains_list, DomainsList):
        for k, v in self.all_conflict_clauses.items():
            [learned_conflict_clauses[k].append(_history_to_conflict_clause(c, self.domains_list.var_mapping)) for c in v]
    return learned_conflict_clauses
        
        
        
@beartype
def get_proof_tree(self: verifier.verifier.Verifier) -> None | dict:
    learned_conflict_clauses = get_learned_conflict_clauses(self)
    if learned_conflict_clauses is None:
        return None
    
    proof_tree = {}
    for obj_idx, conflict_clauses in learned_conflict_clauses.items():
        proof_tree[obj_idx] = [[-1 * lit for lit in clause] for clause in conflict_clauses]
    
    return proof_tree
        
    
@beartype
def export_proof(self: verifier.verifier.Verifier, dnf_objectives: DnfObjectives, output_dir: str) -> None:
    os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Exporting APTP proofs at {output_dir=}')
    proof_dict = self.get_proof_tree()
    for i in range(len(dnf_objectives)):
        # print(objectives.ids[i], objectives.cs[i], objectives.rhs[i], proof[int(objectives.ids[i])])
        aptp_str = create_aptp(
            proof=proof_dict.get(int(dnf_objectives.ids[i]), []), 
            lower=dnf_objectives.lower_bounds[i],
            upper=dnf_objectives.upper_bounds[i],
            cnf_cs=dnf_objectives.cs[i], 
            cnf_rhs=dnf_objectives.rhs[i],
        )
        with open(os.path.join(output_dir, f'proof_{i}.aptp'), 'w') as fp:
            print(aptp_str, file=fp)
