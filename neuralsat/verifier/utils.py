import numpy as np
import random
import torch
import time

from heuristic.decision_heuristics import DecisionHeuristic
from heuristic.restart_heuristics import get_restart_strategy
from attacker.pgd_attack.general import general_attack
from abstractor.abstractor import NetworkAbstractor
from util.misc.check import check_solution
from attacker.attacker import Attacker
from util.misc.logger import logger
from setting import Settings

    
def _preprocess(self, objectives):
    # determine search algorithm
    eps = (objectives.upper_bounds - objectives.lower_bounds).max().item()
    if eps > Settings.safety_property_threshold: # safety properties
        self.input_split = True
    elif np.prod(self.input_shape) <= 100: # small inputs
        self.input_split = True
        
    # debug only
    if 0:
        self.input_split = False
        raise ValueError('Debug only')


def _check_timeout(self, timeout):
    return time.time() - self.start_time > timeout 


def _setup_restart(self, nth_restart):
    params = get_restart_strategy(nth_restart, input_split=self.input_split)
    if params is None:
        return False
    
    logger.info(f'Params of {nth_restart}-th run: {params}')
    abstract_method = params['abstract_method']
    branching_candidates = params.get('branching_candidates', None)
    
    # decision heuristic
    self.decision = DecisionHeuristic(
        branching_candidates=branching_candidates, 
        input_split=self.input_split
    )
    
    # abstractor
    if (not hasattr(self, 'abstractor')) or (abstract_method != self.abstractor.method):
        logger.debug('Initialized abstractor')
        self.abstractor = NetworkAbstractor(
            pytorch_model=self.net, 
            input_shape=self.input_shape, 
            method=abstract_method,
            input_split=self.input_split,
            device=self.device,
        )
    
    return True


def _pre_attack(self, dnf_objectives):
    if Settings.use_attack:
        return Attacker(self.net, dnf_objectives, self.input_shape, device=self.device).run()
    return False, None
    

def _random_idx(total_samples, num_samples, device='cpu'):
    if num_samples >= total_samples:
        return torch.Tensor(range(total_samples)).to(device)
    return torch.Tensor(random.sample(range(total_samples), num_samples)).to(device)


def _attack(self, domain_params, n_sample=50, n_interval=10):
    if not Settings.use_attack:
        return False, None
    
    if self.iteration % n_interval != 0:
        return False, None
    
    # random samples
    indices = _random_idx(len(domain_params.cs), n_sample, device=self.device).long()

    input_lowers = domain_params.input_lowers[indices][None]
    input_uppers = domain_params.input_uppers[indices][None]
    adv_example = (input_lowers + input_uppers) / 2
    
    cs = domain_params.cs[indices].view(1, -1, domain_params.cs[indices].shape[-1])
    rhs = domain_params.rhs[indices].view(1, -1)

    attack_images = general_attack(
        model=self.net, 
        X=adv_example, 
        data_min=input_lowers, 
        data_max=input_uppers, 
        cs_mat=cs, 
        rhs_mat=rhs, 
        cond_mat=[[1 for i in range(domain_params.cs[indices].shape[0])]], 
        same_number_const=True, 
        alpha=(input_uppers - input_lowers).max() / 8, 
        attack_iters=20, 
        num_restarts=5, 
        only_replicate_restarts=True)
    

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


def _fake_learned_clauses():
    return [eval(line) for line in open('./solver/clauses_cnn.txt').read().strip().split('\n')]
    return [eval(line) for line in open('./solver/clauses.txt').read().strip().split('\n')]
    
    