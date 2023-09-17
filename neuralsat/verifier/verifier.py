import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import logging
import torch
import time
import copy

from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from auto_LiRPA.utils import stop_criterion_batch_any
from heuristic.domains_list import DomainsList
from util.misc.result import ReturnStatus
from abstractor.utils import new_slopes
from util.misc.logger import logger
from setting import Settings


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
        self.last_minimum_lowers = -1e9
        self.tightening_patience = 0
        
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
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # attack
        is_attacked, self.adv = self._pre_attack(copy.deepcopy(dnf_objectives))
        if is_attacked:
            return ReturnStatus.SAT  

        # refine
        dnf_objectives, reference_bounds = self._preprocess(dnf_objectives)
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # mip attack
        is_attacked, self.adv = self._mip_attack(reference_bounds)
        if is_attacked:
            return ReturnStatus.SAT 
        
        # verify
        while len(dnf_objectives):
            objective = self.get_objective(dnf_objectives)
            
            # restart variables
            learned_clauses = []
            nth_restart = 0 
            
            # verify objective (multiple times if RESTART is returned)
            while True:
                # get strategy
                if not self._setup_restart(nth_restart, objective):
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
                            reference_bounds=reference_bounds,
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
                    except SystemExit:
                        exit()
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
        
        
    def _initialize(self, objective, preconditions, reference_bounds):
        # initialization params
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds, init_betas=self.refined_betas)
        
        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs).all():
            return []
        
        # keep last layer's alphas for backward propagation
        # FIXME: full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)
        
        # remaining domains
        return DomainsList(
            output_lbs=ret.output_lbs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds, 
            upper_bounds=ret.upper_bounds, 
            lAs=ret.lAs, 
            # slopes=ret.slopes, # full slopes
            slopes=slopes, # last layer's slopes
            histories=copy.deepcopy(ret.histories), 
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )
        
        
    def _verify(self, objective, preconditions, reference_bounds, timeout):
        # print('refined bounds:', sum([(v[1] - v[0]).sum().item() for _, v in reference_bounds.items()])) if reference_bounds is not None else None

        # initialization
        self.domains_list = self._initialize(objective=objective, preconditions=preconditions, reference_bounds=reference_bounds)
        
        # cleaning
        torch.cuda.empty_cache()
        if hasattr(self, 'tightener'):
            self.tightener.reset()
        
        # restart threshold
        max_branches = Settings.max_input_branches if self.input_split else Settings.max_hidden_branches
        max_visited_branches = Settings.max_input_visited_branches if self.input_split else Settings.max_hidden_visited_branches
        
        # main loop
        while len(self.domains_list) > 0:
            self._parallel_dpll()
            
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
            
            
    def _parallel_dpll(self):        
        # step 1: MIP attack
        if Settings.use_mip_attack:
            self.mip_attacker.attack_domains(self.domains_list.pick_out_worst_domains(1001, self.device))
        
        # step 2: stabilizing
        old_domains_length = len(self.domains_list)
        if self._check_invoke_tightening(patience_limit=Settings.mip_tightening_patience):
            self.tightener(self.domains_list, topk=64, timeout=15.0, largest=False, solve_both=True)
            
        # step 3: selection
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        
        # step 4: PGD attack
        is_attacked, self.adv = self._attack(pick_ret, n_interval=Settings.attack_interval)
        if is_attacked:
            return

        # step 5: branching
        decisions = self.decision(self.abstractor, pick_ret)
        # print(decisions)
        
        # step 6: abstraction 
        abstraction_ret = self.abstractor.forward(decisions, pick_ret)
        
        # step 7: pruning complete assignments
        self.adv = self._check_full_assignment(abstraction_ret)
        if self.adv is not None:
            return
        
        # step 8: pruning unverified branches
        self.domains_list.add(decisions, abstraction_ret)
        # TODO: check full assignment after bcp

        # statistics
        self.iteration += 1
        minimum_lowers = self.domains_list.minimum_lowers
        self._update_tightening_patience(minimum_lowers, old_domains_length)
        
        # logging
        msg = (
            f'[{"Input" if self.input_split else "Hidden"} domain]     '
            f'Iteration: {self.iteration:<10} '
            f'Remaining: {len(self.domains_list):<10} '
            f'Visited: {self.domains_list.visited:<10} '
            f'Bound: {minimum_lowers:<15.04f} '
            f'Time elapsed: {time.time() - self.start_time:<10.02f}'
        )
        if logger.isEnabledFor(logging.DEBUG):
            msg += (
                f'Tightening patience: {self.tightening_patience}/{Settings.mip_tightening_patience:<10}'
            )
        logger.info(msg)
    
    
    from .utils import (
        _preprocess, _init_abstractor,
        _check_timeout,
        _setup_restart,
        _pre_attack, _attack, _mip_attack,
        _get_learned_conflict_clauses, _check_full_assignment,
        _check_invoke_tightening, _update_tightening_patience
    )
    