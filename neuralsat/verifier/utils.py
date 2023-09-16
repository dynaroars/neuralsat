import numpy as np
import random
import torch
import time
import copy

from heuristic.restart_heuristics import get_restart_strategy
from heuristic.decision_heuristics import DecisionHeuristic
from heuristic.tightener import Tightener

from attacker.pgd_attack.general import general_attack
from attacker.mip_attack import MIPAttacker
from attacker.attacker import Attacker

from abstractor.abstractor import NetworkAbstractor

from util.misc.check import check_solution
from util.misc.logger import logger

from setting import Settings

DEBUG = False


def new_slopes(slopes, keep_name):
    new_slope = {}
    for relu_layer, alphas in slopes.items():
        new_slope[relu_layer] = {}
        if keep_name in alphas:
            new_slope[relu_layer][keep_name] = alphas[keep_name]
    return new_slope


def _mip_attack(self, reference_bounds):
    if not Settings.use_attack:
        return False, None
    
    if not Settings.use_mip_attack:
        return False, None
    
    return self.mip_attacker.run(reference_bounds)
    
    
def _preprocess(self, objectives):
    # determine search algorithm
    self.refined_betas = None
    
    if Settings.test:
        print(Settings)
        self.input_split = False
        return objectives, None
    
    diff = objectives.upper_bounds - objectives.lower_bounds
    eps = diff.max().item()
    perturbed = (diff > 0).numel()
    logger.info(f'[!] eps = {eps:.06f}, perturbed={perturbed}')
    if eps > Settings.safety_property_threshold: # safety properties
        self.input_split = True
    elif np.prod(self.input_shape) <= 200: # small inputs
        self.input_split = True
    elif np.prod(self.input_shape) >= 100000: # large inputs, e.g., VGG16
        self.input_split = True
        
    if len(objectives) >= 50:
        Settings.use_restart = False
        
    if (not isinstance(objectives.cs, torch.Tensor)) or (not isinstance(objectives.rhs, torch.Tensor)):
        return objectives, None
    
    if not DEBUG:
        if self.input_split and len(objectives) < 50:
            return objectives, None
    else:
        self.input_split = False
        
    try:
        # self._init_abstractor('crown-optimized', objectives)
        self._init_abstractor('backward' if np.prod(self.input_shape) < 100000 else 'forward', objectives)
    except:
        print('Failed to preprocessing objectives')
        return objectives, None
    
    if not torch.allclose(objectives.lower_bounds.mean(dim=0), objectives.lower_bounds[0], 1e-5, 1e-5):
        return objectives, None
    
    # prune objectives
    tmp_objective = copy.deepcopy(objectives)
    tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1] # raise errors if using beta, use full objectives instead
    tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1] # raise errors if using beta, use full objectives instead
        
    # forward
    ret = self.abstractor.initialize(tmp_objective)

    # pruning
    remaining_index = torch.where((ret.output_lbs.detach().cpu() <= tmp_objective.rhs.detach().cpu()).all(1))[0]
    objectives.lower_bounds = objectives.lower_bounds[remaining_index]
    objectives.upper_bounds = objectives.upper_bounds[remaining_index]
    objectives.cs = objectives.cs[remaining_index]
    objectives.rhs = objectives.rhs[remaining_index]
    
    # refine
    refined_intermediate_bounds = None
    if len(objectives) and (Settings.use_mip_refine or Settings.use_mip_attack or Settings.use_mip_tightening) \
            and self.abstractor.method == 'backward':
        logger.info(f'Refining hidden bounds for {len(objectives)} remaining objectives')
        tmp_objective = copy.deepcopy(objectives)
        tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1].to(self.device)
        tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1].to(self.device)
        
        tic = time.time()
        c_to_use = tmp_objective.cs.transpose(0, 1).to(self.device) if tmp_objective.cs.shape[1] == 1 else None
        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=tmp_objective.lower_bounds.view(self.input_shape), 
            input_upper=tmp_objective.upper_bounds.view(self.input_shape), 
            c=c_to_use,
            refine=Settings.use_mip_refine,
            timeout=None,
        )
        logger.debug(f'MIP: {time.time() - tic:.04f}')
        
        # self.abstractor.net.print_betas()

        if Settings.use_mip_refine:
            # forward with refinement
            refined_intermediate_bounds = self.abstractor.net.get_refined_intermediate_bounds()
            ret = self.abstractor.initialize(tmp_objective, reference_bounds=refined_intermediate_bounds)
            
            # pruning
            remaining_index = torch.where((ret.output_lbs.detach().cpu() <= tmp_objective.rhs.detach().cpu()).all(1))[0]
            objectives.lower_bounds = objectives.lower_bounds[remaining_index]
            objectives.upper_bounds = objectives.upper_bounds[remaining_index]
            objectives.cs = objectives.cs[remaining_index]
            objectives.rhs = objectives.rhs[remaining_index]
            
            self.refined_betas = self.abstractor.net.get_betas()
        
        # mip attacker
        if Settings.use_mip_attack:
            self.mip_attacker = MIPAttacker(
                abstractor=self.abstractor, 
                objectives=objectives, 
            )
            
        # mip tightener
        if Settings.use_mip_tightening:
            self.tightener = Tightener(
                abstractor=self.abstractor,
                objectives=objectives, 
            )
            
    logger.info(f'Remain {len(objectives)} objectives')
    return objectives, refined_intermediate_bounds


def _check_timeout(self, timeout):
    return time.time() - self.start_time > timeout 


def _init_abstractor(self, method, objective):
    self.abstractor = NetworkAbstractor(
        pytorch_model=self.net, 
        input_shape=self.input_shape, 
        method=method,
        input_split=self.input_split,
        device=self.device,
    )
    
    self.abstractor.setup(objective)
    
    
def _setup_restart(self, nth_restart, objective):
    params = get_restart_strategy(nth_restart, input_split=self.input_split)
    if params is None:
        return False
    
    if np.prod(self.input_shape) >= 100000: # large inputs, e.g., VGG16
        params['abstract_method'] = 'forward'
        
    logger.info(f'Params of {nth_restart+1}-th run: {params}')
    abstract_method = params['abstract_method']
    decision_topk = params.get('decision_topk', None)
    random_selection = params.get('random_selection', False)
    
    # decision heuristic
    self.decision = DecisionHeuristic(
        decision_topk=decision_topk, 
        input_split=self.input_split,
        random_selection=random_selection,
    )
    
    # abstractor
    if (not hasattr(self, 'abstractor')) or (abstract_method != self.abstractor.method):
        self._init_abstractor(abstract_method, objective)
        
    return True


def _pre_attack(self, dnf_objectives, timeout=0.5):
    if Settings.use_attack:
        return Attacker(self.net, dnf_objectives, self.input_shape, device=self.device).run(timeout=timeout)
    return False, None
    

def _random_idx(total_samples, num_samples, device='cpu'):
    if num_samples >= total_samples:
        return torch.Tensor(range(total_samples)).to(device)
    return torch.Tensor(random.sample(range(total_samples), num_samples)).to(device)


def _attack(self, domain_params, n_sample=50, n_interval=1):
    if not Settings.use_attack:
        return False, None
    
    if self.iteration % n_interval != 0:
        return False, None

    # random samples
    indices = _random_idx(len(domain_params.cs), n_sample, device=self.device).long()

    input_lowers = domain_params.input_lowers[indices][None]
    input_uppers = domain_params.input_uppers[indices][None]
    # adv_example = (input_lowers + input_uppers) / 2
    adv_example = (input_uppers - input_lowers) * torch.rand(input_lowers.shape, device=self.device) + input_lowers
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
        num_restarts=5, 
        only_replicate_restarts=True,
        use_gama=False,
    )
    if (attack_images is None) and (n_interval == 1):
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
        )

    if attack_images is not None:
        for i in range(attack_images.shape[1]): # restarts
            for j in range(attack_images.shape[2]): # props
                adv = attack_images[:, i, j]
                if check_solution(self.net, adv, domain_params.cs[indices][j], domain_params.rhs[indices][j], input_lowers[:, j], input_uppers[:, j]):
                    return True, adv
        logger.debug("[!] Invalid counter-example")
        
    return False, None


def _get_learned_conflict_clauses(self):
    if hasattr(self.domains_list, 'all_conflict_clauses'):
        return self.domains_list.all_conflict_clauses
    return []

