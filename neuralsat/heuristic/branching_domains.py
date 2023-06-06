from collections import defaultdict
import torch
import copy
import time

from util.misc.tensor_storage import TensorStorage
from util.misc.result import AbstractResults
from .util import compute_masks
from setting import Settings


class DomainsList:
    
    "List of unverified branches"

    def __init__(self, 
                 input_lowers, input_uppers, 
                 lower_bounds, upper_bounds, 
                 lAs, 
                 slopes, histories, 
                 cs, rhs, 
                 input_split=False, preconditions=[]):
        
        ######## clause learning ########
        self.use_restart = Settings.use_restart and (lower_bounds is not None) and len(preconditions)
        if self.use_restart:
            stat = self.init_sat_solver(lower_bounds=lower_bounds, upper_bounds=upper_bounds, histories=histories, preconditions=preconditions)
            if not stat:
                raise ValueError('BCP conflict')
            print(histories)
            self.all_conflict_clauses = []
        ######## end clause learning ########
        
        self.input_split = input_split
        
        # input bounds
        self.all_input_lowers = TensorStorage(input_lowers.cpu())
        self.all_input_uppers = TensorStorage(input_uppers.cpu())
        
        # properties
        self.all_cs = TensorStorage(cs.cpu())
        self.all_rhs = TensorStorage(rhs.cpu())
    
        # alpha
        self.all_slopes = defaultdict(dict)
        for k in slopes:
            self.all_slopes[k] = {}
            for kk, v in slopes[k].items():
                if kk not in self.all_slopes[k]:
                    self.all_slopes[k][kk] = TensorStorage(v.cpu(), concat_dim=2)
                else:
                    self.all_slopes[k][kk].append(v.cpu())

        if self.input_split:
            self.all_lower_bounds = self.all_upper_bounds = self.all_lAs = self.all_histories = self.all_betas = None
        else: # hidden spliting 
            self.all_lAs = [TensorStorage(item.cpu()) for item in lAs]
            # hidden bounds
            self.all_lower_bounds = [TensorStorage(item.cpu()) for item in lower_bounds]
            self.all_upper_bounds = [TensorStorage(item.cpu()) for item in upper_bounds]
            # branching
            self.all_histories = [copy.deepcopy(histories) for _ in range(len(cs))]
            # beta
            self.all_betas = [None for i in range(len(cs))]
            
        self._check_consistent()
        
        
    def _check_consistent(self):
        assert len(self.all_input_lowers) == len(self.all_input_uppers) == len(self.all_cs) == len(self.all_rhs) == len(self)
        if not self.input_split:
            assert len(self.all_betas) == len(self.all_histories) == len(self)
            assert len(self.all_lower_bounds) == len(self.all_upper_bounds) 
            assert all([len(_) == len(self) for _ in self.all_lower_bounds])
            assert all([len(_) == len(self) for _ in self.all_upper_bounds])
            assert all([len(_) == len(self) for _ in self.all_lAs])
            if self.use_restart:
                assert len(self.all_sat_solvers) == len(self), print(f'len(self.all_sat_solvers)={len(self.all_sat_solvers)}, len(self)={len(self)}')
                


    def pick_out(self, batch, device='cpu'):
        # tic = time.time()
        # total_sat_time = 0.0
        
        assert batch > 0
        batch = min(len(self), batch)

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        # input bounds
        input_lowers = self.all_input_lowers.pop(batch).to(device=device, non_blocking=True)
        input_uppers = self.all_input_uppers.pop(batch).to(device=device, non_blocking=True)
        
        # properties
        cs = self.all_cs.pop(batch).to(device=device, non_blocking=True)
        rhs = self.all_rhs.pop(batch).to(device=device, non_blocking=True)
        
        # alpha
        new_slopes = defaultdict(dict)
        for k, v_map in self.all_slopes.items():
            new_slopes[k] = {}
            for kk, vv in v_map.items():
                new_slopes[k][kk] = vv.pop(batch).to(device=device, non_blocking=True)
        
        if self.input_split:
            # input splitting
            lower_bounds = upper_bounds = new_masks = new_lAs = new_betas = new_histories = None
            new_sat_solvers = None
        else: 
            # hidden spliting 
            new_lAs = [lA.pop(batch).to(device=device, non_blocking=True) for lA in self.all_lAs]
            # hidden bounds
            lower_bounds = [lower_bound.pop(batch) for lower_bound in self.all_lower_bounds]
            upper_bounds = [upper_bound.pop(batch) for upper_bound in self.all_upper_bounds]

            lower_bounds = [item.to(device=device, non_blocking=True) for item in lower_bounds]
            upper_bounds = [item.to(device=device, non_blocking=True) for item in upper_bounds]
            
            # pop batch 
            new_betas = self.all_betas[-batch:]
            new_histories = copy.deepcopy(self.all_histories[-batch:])

            # remove batch
            self.all_betas = self.all_betas[:-batch]
            self.all_histories = self.all_histories[:-batch]
            
            if self.use_restart:
                # tic_sat = time.time()
                new_sat_solvers = self.all_sat_solvers[-batch:]
                # new_sat_solvers = copy.deepcopy(self.all_sat_solvers[-batch:])
                self.all_sat_solvers = self.all_sat_solvers[:-batch]
                # total_sat_time += (time.time() - tic_sat)
            else:
                new_sat_solvers = None
            
            new_masks = compute_masks(lower_bounds=lower_bounds, upper_bounds=upper_bounds, device=device)
            
            assert len(new_betas) == len(new_histories) == len(lower_bounds[0]) == len(upper_bounds[0]) == len(new_lAs[0]) == batch 
        
        self._check_consistent()
        
        # print('Pick out time:', time.time() - tic, 'Sat time:', total_sat_time)

        return AbstractResults(**{
            'input_lowers': input_lowers, 
            'input_uppers': input_uppers, 
            'masks': new_masks, 
            'lAs': new_lAs, 
            'lower_bounds': lower_bounds, 
            'upper_bounds': upper_bounds, 
            'slopes': new_slopes, 
            'betas': new_betas,
            'histories': new_histories,
            'cs': cs,
            'rhs': rhs,
            'sat_solvers': new_sat_solvers,
        })


    def add(self, branching_decisions, domain_params):
        # tic = time.time()
        # total_sat_time = 0.0
        
        assert branching_decisions is not None
        batch = len(branching_decisions)
        assert batch > 0
        
        remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]
        
        # hidden splitting
        if not self.input_split:
            new_masks = compute_masks(lower_bounds=domain_params.lower_bounds, upper_bounds=domain_params.upper_bounds, device='cpu')
            
            extra_conflict_index = []
            for idx_ in remaining_index:
                # check full assignment
                if sum([layer_mask[idx_].sum() for layer_mask in new_masks]) == 0:
                    # already check
                    extra_conflict_index.append(idx_)
                    continue
                                
                # new decision
                idx = idx_ % batch
                new_history = copy.deepcopy(domain_params.histories[idx])
                new_history[branching_decisions[idx][0]][0].append(branching_decisions[idx][1])
                new_history[branching_decisions[idx][0]][1].append(+1.0 if idx_ < batch else -1.0)

                # repetition
                if branching_decisions[idx][1] in domain_params.histories[idx][branching_decisions[idx][0]][0]:
                    print(branching_decisions[idx], domain_params.histories[idx])
                    raise RuntimeError('Repeated split')
                
                # bcp
                if self.use_restart:
                    new_sat_solver = self.boolean_propagation(
                        domain_params=domain_params,
                        branching_decisions=branching_decisions,
                        new_history=new_history,
                        batch_idx=idx_
                    )
                    if new_sat_solver is None:
                        extra_conflict_index.append(idx_)
                        continue
                            
                    self.all_sat_solvers.append(new_sat_solver)

                self.all_histories.append(new_history)
                self.all_betas.append(domain_params.betas[idx_])
                
            if len(extra_conflict_index):
                assert len(extra_conflict_index) == len(list(set(extra_conflict_index)))
                for eci in extra_conflict_index:
                    remaining_index = remaining_index[remaining_index != eci]
            
            # conflict clauses
            if self.use_restart:
                self.save_conflict_clauses(branching_decisions=branching_decisions, domain_params=domain_params, remaining_index=remaining_index)
            
            # lower A matrix
            [lA.append(new_lA[remaining_index]) for lA, new_lA in zip(self.all_lAs, domain_params.lAs)]
            
            # hidden bounds
            [lb.append(new_lb[remaining_index]) if new_lb is not None else None for lb, new_lb in zip(self.all_lower_bounds, domain_params.lower_bounds)]
            [ub.append(new_ub[remaining_index]) if new_ub is not None else None for ub, new_ub in zip(self.all_upper_bounds, domain_params.upper_bounds)]
        
        # input bounds
        self.all_input_lowers.append(domain_params.input_lowers[remaining_index])
        self.all_input_uppers.append(domain_params.input_uppers[remaining_index])
        
        # properties
        self.all_cs.append(domain_params.cs[remaining_index])
        self.all_rhs.append(domain_params.rhs[remaining_index])
        
        # alpha
        if domain_params.slopes is not None:
            for k, v in self.all_slopes.items():
                for kk, vv in v.items():
                    vv.append(domain_params.slopes[k][kk][:,:,remaining_index])
            
        # checking
        self._check_consistent()
        # print('Add time:', time.time() - tic, 'Sat time:', total_sat_time)
        
        
    def __len__(self):
        return len(self.all_input_lowers)


    from .util import init_sat_solver, update_hidden_bounds_histories, boolean_propagation, save_conflict_clauses