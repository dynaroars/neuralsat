import numpy as np
import random
import torch
import time
import copy

from heuristic.restart_heuristics import get_restart_strategy, HIDDEN_SPLIT_RESTART_STRATEGIES
from heuristic.decision_heuristics import DecisionHeuristic
from heuristic.tightener import Tightener
from heuristic.util import compute_masks

from attacker.pgd_attack.general import general_attack
from attacker.mip_attack import MIPAttacker
from attacker.attacker import Attacker

from abstractor.abstractor import NetworkAbstractor

from util.misc.check import check_solution
from util.misc.logger import logger

from setting import Settings


def _mip_attack(self, reference_bounds):
    if not Settings.use_attack:
        return False, None
    
    if not Settings.use_mip_attack:
        return False, None
    
    return self.mip_attacker.run(reference_bounds)
    
    
def _preprocess(self, objectives, forced_input_split=None):
    # determine search algorithm
    self.refined_betas = None
    
    diff = objectives.upper_bounds - objectives.lower_bounds
    eps = diff.max().item()
    perturbed = (diff > 0).numel()
    logger.info(f'[!] eps = {eps:.06f}, perturbed={perturbed}')
    
    if Settings.test:
        self.input_split = False
    elif forced_input_split is not None:
        self.input_split = forced_input_split
    elif eps > Settings.safety_property_threshold: # safety properties
        self.input_split = True
    elif np.prod(self.input_shape) <= 200: # small inputs
        self.input_split = True
    elif np.prod(self.input_shape) >= 100000: # large inputs, e.g., VGG16
        self.input_split = True
        
    if len(objectives) >= 50:
        Settings.use_restart = False
        
    if (not isinstance(objectives.cs, torch.Tensor)) or (not isinstance(objectives.rhs, torch.Tensor)):
        return objectives, None
    
    if self.input_split and len(objectives) < 50:
        return objectives, None
    
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
    if len(objectives) and (Settings.use_mip_tightening) and self.abstractor.method == 'backward':
        logger.info(f'Refining hidden bounds for {len(objectives)} remaining objectives')
        tmp_objective = copy.deepcopy(objectives)
        tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1].to(self.device)
        tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1].to(self.device)
        
        tic = time.time()
        c_to_use = tmp_objective.cs.transpose(0, 1).to(self.device) if tmp_objective.cs.shape[1] == 1 else None

        use_refined = not Settings.use_restart
        if any([isinstance(_, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)) for _ in self.net.modules()][1:]):
            # TODO: skip refine for Conv layers
            use_refined = False

        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=tmp_objective.lower_bounds.view(self.input_shape), 
            input_upper=tmp_objective.upper_bounds.view(self.input_shape), 
            c=c_to_use,
            refine=use_refined,
            timeout=None,
        )
        logger.debug(f'MIP: {time.time() - tic:.04f}')
        
        # self.abstractor.net.print_betas()

        if use_refined:
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
        
        # torch.save(refined_intermediate_bounds, 'refined.pt')
        
    # mip tightener
    if len(objectives):
        # mip attacker
        if Settings.use_mip_attack:
            self.mip_attacker = MIPAttacker(
                abstractor=self.abstractor, 
                objectives=objectives, # full objectives
            )
            
        if Settings.use_mip_tightening:
            self.tightener = Tightener(
                abstractor=self.abstractor,
                objectives=objectives,
            )
            
    logger.info(f'Remain {len(objectives)} objectives')
    # refined_intermediate_bounds = torch.load('refined.pt')
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
    self.num_restart = nth_restart + 1
    params = get_restart_strategy(nth_restart, input_split=self.input_split)
    if params is None:
        raise
    
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
        seed=nth_restart+2,
    )
    
    refined_intermediate_bounds = None
    if Settings.use_restart and self.num_restart == len(HIDDEN_SPLIT_RESTART_STRATEGIES) and Settings.use_mip_tightening:
        if abstract_method == 'forward':
            return None
        if not torch.allclose(objective.lower_bounds.mean(dim=0), objective.lower_bounds[0], 1e-5, 1e-5):
            return None
        
        if any([isinstance(_, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)) for _ in self.net.modules()][1:]):
            # TODO: skip refine for Conv layers
            return None
            
        self._init_abstractor('backward', objective)
        
        tmp_objective = copy.deepcopy(objective)
        tmp_objective.lower_bounds = tmp_objective.lower_bounds[0:1].to(self.device)
        tmp_objective.upper_bounds = tmp_objective.upper_bounds[0:1].to(self.device)
        
        ret = self.abstractor.initialize(tmp_objective)
        
        c_to_use = tmp_objective.cs.transpose(0, 1).to(self.device) if tmp_objective.cs.shape[1] == 1 else None
        self.abstractor.build_lp_solver(
            model_type='mip', 
            input_lower=tmp_objective.lower_bounds.view(self.input_shape), 
            input_upper=tmp_objective.upper_bounds.view(self.input_shape), 
            c=c_to_use,
            refine=True,
            timeout=None,
        )
        refined_intermediate_bounds = self.abstractor.net.get_refined_intermediate_bounds()
        del self.abstractor
    
    # abstractor
    if (not hasattr(self, 'abstractor')) or (abstract_method != self.abstractor.method):
        self._init_abstractor(abstract_method, objective)
        
    return refined_intermediate_bounds


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
    if (attack_images is None) and (self.iteration % (3 * n_interval) == 0) and 0:
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


def _check_invoke_tightening(self, patience_limit=10):
    if not Settings.use_mip_tightening:
        return False
    
    if Settings.test:
        return True
    
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
    

def _update_tightening_patience(self, minimum_lowers, old_domains_length):
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
            
    
def _check_full_assignment(self, domain_params):
    if domain_params.lower_bounds is None:
        return None
    
    new_masks = compute_masks(lower_bounds=domain_params.lower_bounds, upper_bounds=domain_params.upper_bounds, device='cpu')
    remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]

    for idx_ in remaining_index:
        if sum([layer_mask[idx_].sum() for layer_mask in new_masks]) == 0:
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
                lower_bounds=[l[idx_] for l in domain_params.lower_bounds],
                upper_bounds=[u[idx_] for u in domain_params.upper_bounds],
                c=domain_params.cs[idx_],
                rhs=domain_params.rhs[idx_]
            )
            
            if feasible:
                return adv
    return None

    