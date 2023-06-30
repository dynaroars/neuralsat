import numpy as np
import torch
import time
import copy

from auto_LiRPA.utils import stop_criterion_batch_any

from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from heuristic.domains_list import DomainsList
from util.misc.result import ReturnStatus
from heuristic.util import compute_masks
from abstractor.utils import new_slopes
from util.misc.logger import logger

from setting import Settings

import warnings
warnings.filterwarnings(action='ignore')


class Verifier:
    
    "Branch-and-Bound verifier"
    
    def __init__(self, net, input_shape, batch=1000, device='cpu'):
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device
        
        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)

        # counter-example
        self.adv = None
        
        # debug
        self.iteration = 0
        
        
    def get_objective(self, dnf_objectives):
        if self.input_split:
            objective = dnf_objectives.pop(self.batch)
        elif Settings.use_restart:
            objective = dnf_objectives.pop(1)
        else:
            objective = dnf_objectives.pop(self.batch)
        return objective
    
    
    def verify(self, dnf_objectives, preconditions=[], timeout=3600):
        self.start_time = time.time()
        
        # attack
        is_attacked, self.adv = self._pre_attack(copy.deepcopy(dnf_objectives))
        if is_attacked:
            return ReturnStatus.SAT  
        
        self._preprocess(copy.deepcopy(dnf_objectives))
        
        # verify
        while len(dnf_objectives):
            objective = self.get_objective(dnf_objectives)
            
            # restart variables
            learned_clauses = []
            nth_restart = 0 
            
            # verify objective (multiple times if RESTART is returned)
            while True:
                # get strategy
                if not self._setup_restart(nth_restart):
                    return ReturnStatus.UNKNOWN
                
                # TODO: refinement
                
                # adaptive batch size
                while True: 
                    logger.debug(f'Try batch size {self.batch}')
                    try:
                        # main function
                        status = self._verify(
                            objective=objective, 
                            preconditions=preconditions+learned_clauses, 
                            timeout=timeout
                        )
                    except RuntimeError as exception:
                        if is_cuda_out_of_memory(exception):
                            if self.batch == 1:
                                # cannot find a suitable batch size to fit this device
                                return ReturnStatus.UNKNOWN
                            self.batch = self.batch // 2
                            dnf_objectives.add(objective)
                            objective = self.get_objective(dnf_objectives)
                            continue
                        else:
                            raise NotImplementedError
                    except:
                        raise NotImplementedError
                    else:
                        gc_cuda()
                        break
                    
                # handle returning status
                if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT]:
                    return status 
                if status == ReturnStatus.UNSAT:
                    break # objective is verified
                if status == ReturnStatus.RESTART:
                    logger.debug('Restarting')
                    learned_clauses += self._get_learned_conflict_clauses()
                    nth_restart += 1
                    continue
                raise NotImplementedError()
            
            logger.info(f'Verified: {len(objective.cs)} \t Remain: {len(dnf_objectives)}')
            
        return ReturnStatus.UNSAT  
        
        
    def _initialize(self, objective, preconditions):
        # initialization params
        ret = self.abstractor.initialize(objective)
        
        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs).all():
            return []
        
        # keep last layer's alphas for backward propagation
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)
        
        # remaining domains
        return DomainsList(
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds, 
            upper_bounds=ret.upper_bounds, 
            lAs=ret.lAs, 
            slopes=slopes,
            histories=copy.deepcopy(ret.histories), 
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )
        
        
    def _verify(self, objective, preconditions, timeout):
        # initialization
        self.domains_list = self._initialize(objective=objective, preconditions=preconditions)
        
        # cleaning
        torch.cuda.empty_cache()
        
        # restart threshold
        max_branches = Settings.max_input_branches if self.input_split else Settings.max_hidden_branches
        max_visited_branches = Settings.max_input_visited_branches if self.input_split else Settings.max_hidden_visited_branches
        
        # main loop
        while len(self.domains_list) > 0:
            self._branch_and_bound()
            
            # check adv founded
            if self.adv is not None:
                return ReturnStatus.SAT
            
            # check timeout
            if self._check_timeout(timeout):
                return ReturnStatus.TIMEOUT
            
            # check restart
            if Settings.use_restart:
                if (len(self.domains_list) > max_branches) or (self.domains_list.visited > max_visited_branches):
                    return ReturnStatus.RESTART
        
        return ReturnStatus.UNSAT
            
            
    def _branch_and_bound(self):
        # step 1: pick out
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        
        # step 2: attack
        is_attacked, self.adv = self._attack(pick_ret)
        if is_attacked:
            return

        # step 3: branching
        decisions = self.decision(self.abstractor, pick_ret)
        # print(decisions)

        # step 4: abstraction 
        abstraction_ret = self.abstractor.forward(decisions, pick_ret)
        
        # step 5: pruning
        # 5.1: full assignment
        self.adv = self._check_full_assignment(abstraction_ret)
        if self.adv is not None:
            return
        
        # 5.2: unreachable domains
        self.domains_list.add(decisions, abstraction_ret)
        
        # 5.3: TODO: check full assignment after bcp

        # logging
        self.iteration += 1
        logger.info(f'[{"Input" if self.input_split else "Hidden"} domain] '
                    f'Iteration: {self.iteration:<6} '
                    f'Remaining: {len(self.domains_list):<10} Visited: {self.domains_list.visited}')
        
        # print((pick_ret.input_uppers - pick_ret.input_lowers).sum().detach().cpu(), abstraction_ret.output_lbs.detach().cpu().flatten())
        
        
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

        
    from .utils import (
        _preprocess,
        _check_timeout,
        _setup_restart,
        _pre_attack, _attack, 
        _get_learned_conflict_clauses,
    )
    