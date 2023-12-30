import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import psutil
import torch
import time
import copy

from heuristic.restart_heuristics import HIDDEN_SPLIT_RESTART_STRATEGIES
from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from auto_LiRPA.utils import stop_criterion_batch_any
from heuristic.domains_list import DomainsList
from util.misc.result import ReturnStatus
from verifier.utils import _prune_domains
from abstractor.utils import new_slopes
from util.misc.logger import logger
from util.misc.timer import Timers
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
        
        # stats
        self.all_conflict_clauses = []
        self.visited = 0
        
        
    def get_objective(self, dnf_objectives):
        # FIXME: urgent
        # objective = dnf_objectives.pop(1)
        objective = dnf_objectives.pop(max(1, self.batch))
        return objective
    
        if self.input_split:
            objective = dnf_objectives.pop(max(10, self.batch))
        elif Settings.use_restart:
            objective = dnf_objectives.pop(1)
        else:
            objective = dnf_objectives.pop(max(10, self.batch))
        return objective
    
    
    def compute_stability(self, dnf_objectives):
        print('compute_stability')
        if not (hasattr(self, 'abstractor')):
            self._init_abstractor('backward' if np.prod(self.input_shape) < 100000 else 'forward', dnf_objectives)
            
        return self.abstractor.compute_stability(dnf_objectives)
    
    
    def verify(self, dnf_objectives, preconditions=[], timeout=3600):
        self.start_time = time.time()
        self.status = self._verify(
            dnf_objectives=dnf_objectives,
            preconditions=preconditions,
            timeout=timeout,
        )
        return self.status
    
    
    def _verify(self, dnf_objectives, preconditions=[], timeout=3600):
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # attack
        Timers.tic('Pre-attack') if Settings.use_timer else None
        is_attacked, self.adv = self._pre_attack(copy.deepcopy(dnf_objectives))
        Timers.toc('Pre-attack') if Settings.use_timer else None
        if is_attacked:
            return ReturnStatus.SAT  

        # refine
        Timers.tic('Preprocess') if Settings.use_timer else None
        dnf_objectives, reference_bounds = self._preprocess(dnf_objectives, forced_input_split=None)
        Timers.toc('Preprocess') if Settings.use_timer else None
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # mip attack
        is_attacked, self.adv = self._mip_attack(reference_bounds)
        if is_attacked:
            return ReturnStatus.SAT 
        
        # verify
        while len(dnf_objectives):
            Timers.tic('Get objective') if Settings.use_timer else None
            objective = self.get_objective(dnf_objectives)
            Timers.toc('Get objective') if Settings.use_timer else None
            
            # restart variables
            learned_clauses = []
            nth_restart = 0 
            
            # verify objective (multiple times if RESTART is returned)
            while True:
                # get strategy + refinement
                Timers.tic('Setup restart') if Settings.use_timer else None
                new_reference_bounds = self._setup_restart(nth_restart, objective)
                Timers.toc('Setup restart') if Settings.use_timer else None
                
                # adaptive batch size
                while True: 
                    logger.info(f'Try batch size {self.batch}')
                    try:
                        # main function
                        Timers.tic('Verify one') if Settings.use_timer else None
                        status = self._verify_one(
                            objective=objective, 
                            preconditions=preconditions+learned_clauses, 
                            reference_bounds=reference_bounds if new_reference_bounds is None else new_reference_bounds,
                            timeout=timeout
                        )
                        Timers.toc('Verify one') if Settings.use_timer else None
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
                    
                # stats
                Timers.tic('Save stats') if Settings.use_timer else None
                self._save_stats()
                Timers.toc('Save stats') if Settings.use_timer else None
                
                # handle returning status
                if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT, ReturnStatus.UNKNOWN]:
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
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []
        
        # keep last layer's alphas for backward propagation
        # FIXME: full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)
        
        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
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
        
        
    def _verify_one(self, objective, preconditions, reference_bounds, timeout):
        # initialization
        Timers.tic('Initialization') if Settings.use_timer else None
        self.domains_list = self._initialize(objective=objective, preconditions=preconditions, reference_bounds=reference_bounds)
        Timers.toc('Initialization') if Settings.use_timer else None
            
        # cleaning
        torch.cuda.empty_cache()
        if hasattr(self, 'tightener'):
            self.tightener.reset()
        
        # restart threshold
        max_branches = Settings.max_input_branches if self.input_split else Settings.max_hidden_branches
        max_visited_branches = Settings.max_input_visited_branches if self.input_split else Settings.max_hidden_visited_branches
        
        # main loop
        start_time = time.time()
        while len(self.domains_list) > 0:
            # search
            Timers.tic('Main loop') if Settings.use_timer else None
            self._parallel_dpll()
            Timers.toc('Main loop') if Settings.use_timer else None
                
            # check adv founded
            if self.adv is not None:
                if self._check_adv_f64(self.adv, objective):
                    return ReturnStatus.SAT
                logger.debug("[!] Invalid counter-example")
                self.adv = None
            
            # check timeout
            if self._check_timeout(timeout):
                return ReturnStatus.TIMEOUT
            
            # check restart
            if Settings.use_restart and (self.num_restart < len(HIDDEN_SPLIT_RESTART_STRATEGIES)):
                if (len(self.domains_list) > max_branches) or (self.domains_list.visited > max_visited_branches):
                    return ReturnStatus.RESTART
                
            if psutil.virtual_memory()[2] > 70.0:
                logger.debug('OOM')
                return ReturnStatus.UNKNOWN
            
        logger.debug(f'Main loop: {time.time() - start_time}')
        
        return ReturnStatus.UNSAT
            
            
    def _parallel_dpll(self):
        iter_start = time.time()
        
        # step 1: MIP attack
        if Settings.use_mip_attack:
            self.mip_attacker.attack_domains(self.domains_list.pick_out_worst_domains(1001, 'cpu'))
        
        # step 2: stabilizing
        old_domains_length = len(self.domains_list)
        unstable = self.domains_list.count_unstable_neurons()
        if self._check_invoke_tightening(patience_limit=Settings.mip_tightening_patience):
            Timers.tic('Tightening') if Settings.use_timer else None
            self.tightener(
                domain_list=self.domains_list, 
                topk=Settings.mip_tightening_topk, 
                timeout=Settings.mip_tightening_timeout_per_neuron, 
                largest=False, # stabilize near-stable neurons
                solve_both=True, # stabilize both upper and lower bounds
            )
            Timers.toc('Tightening') if Settings.use_timer else None
            
        # step 3: selection
        Timers.tic('Get domains') if Settings.use_timer else None
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        Timers.toc('Get domains') if Settings.use_timer else None
        
        # step 4: PGD attack
        Timers.tic('Loop attack') if Settings.use_timer else None
        self.adv = self._attack(pick_ret, n_interval=Settings.attack_interval)
        Timers.toc('Loop attack') if Settings.use_timer else None
        if self.adv is not None:
            return

        # step 5: complete assignments
        self.adv, remain_idx = self._check_full_assignment(pick_ret)
        if (self.adv is not None): 
            return
        
        # pruning
        pruned_ret = _prune_domains(pick_ret, remain_idx) if remain_idx is not None else pick_ret
        if not len(pruned_ret.input_lowers): 
            return
            
        # step 6: branching
        Timers.tic('Decision') if Settings.use_timer else None
        decisions = self.decision(self.abstractor, pruned_ret)
        Timers.toc('Decision') if Settings.use_timer else None
        
        # step 7: abstraction 
        Timers.tic('Abstraction') if Settings.use_timer else None
        abstraction_ret = self.abstractor.forward(decisions, pruned_ret)
        Timers.toc('Abstraction') if Settings.use_timer else None

        # step 8: pruning unverified branches
        Timers.tic('Add domains') if Settings.use_timer else None
        self.domains_list.add(abstraction_ret, decisions)
        Timers.toc('Add domains') if Settings.use_timer else None

        # statistics
        self.iteration += 1
        minimum_lowers = self.domains_list.minimum_lowers
        self._update_tightening_patience(minimum_lowers, old_domains_length)
        
        # logging
        msg = (
            f'[{"Input" if self.input_split else "Hidden"} splitting]     '
            f'Iteration: {self.iteration:<10} '
            f'Remaining: {len(self.domains_list):<10} '
            f'Visited: {self.domains_list.visited:<10} '
            f'Bound: {minimum_lowers:<15.04f} '
            f'Iteration elapsed: {time.time() - iter_start:<10.02f} '
            f'Time elapsed: {time.time() - self.start_time:<10.02f} '
        )
        if Settings.use_mip_tightening:
            msg += f'Tightening patience: {self.tightening_patience}/{Settings.mip_tightening_patience:<10}'
            
        if (not self.input_split) and (unstable is not None):
            msg += f'Unstable neurons: {unstable:<10}'
            
        logger.info(msg)
    
    
    from .utils import (
        _preprocess, _init_abstractor,
        _check_timeout,
        _setup_restart,
        _pre_attack, _attack, _mip_attack, _check_adv_f64,
        _get_learned_conflict_clauses, _check_full_assignment,
        _check_invoke_tightening, _update_tightening_patience,
        compute_stability, _save_stats, get_stats,
        get_unsat_core,
    )
    