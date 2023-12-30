from collections import defaultdict
import torch
import copy
import time

from util.misc.tensor_storage import TensorStorage
from util.misc.result import AbstractResults
from abstractor.utils import _copy_history
from .util import compute_masks
from setting import Settings

from util.misc.logger import logger

class DomainsList:
    
    "List of unverified branches"

    def __init__(self, 
                 net,
                 output_lbs,
                 input_lowers, input_uppers, 
                 lower_bounds, upper_bounds, 
                 lAs, 
                 slopes, histories, 
                 cs, rhs, 
                 input_split=False, preconditions=[]):

        self.net = net
        self.final_name = self.net.final_node_name

        self.input_split = input_split
        self.visited = 0
        self.all_conflict_clauses = []
        
        # FIXME: len(input_lowers) > 1
        self.use_restart = Settings.use_restart and (lower_bounds is not None) and (len(input_lowers) == 1) # and len(preconditions)
        if self.use_restart:
            tic = time.time()
            stat = self.init_sat_solver(
                lower_bounds=lower_bounds, 
                upper_bounds=upper_bounds, 
                histories=histories, 
                preconditions=preconditions
            )
            logger.info(f'Initialize {len(preconditions)} learned clauses in {time.time() - tic:.03f} seconds')
            if not stat:
                raise ValueError('BCP conflict')
        
        # input bounds
        self.all_input_lowers = TensorStorage(input_lowers.cpu())
        self.all_input_uppers = TensorStorage(input_uppers.cpu())
        
        # output bounds
        self.all_output_lowers = TensorStorage(output_lbs.cpu())
        
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
            self.all_lAs = {k: TensorStorage(v.cpu()) for k, v in lAs.items()}
            # hidden bounds
            self.all_lower_bounds = {k: TensorStorage(v.cpu()) for k, v in lower_bounds.items() if k != self.final_name}
            self.all_upper_bounds = {k: TensorStorage(v.cpu()) for k, v in upper_bounds.items() if k != self.final_name}
            # branching
            self.all_histories = [_copy_history(histories) for _ in range(len(cs))]
            # beta
            self.all_betas = [None for i in range(len(cs))]
            
        self._check_consistent()
        
        
    @property
    def var_mapping(self):
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for layer in self.net.split_nodes:
                for nid in range(layer.lower.flatten(start_dim=1).shape[-1]):
                    self._var_mapping[layer.name, nid] = count
                    count += 1
        return self._var_mapping
    
    
    @property
    def reversed_var_mapping(self):
        if not hasattr(self, '_reversed_var_mapping'):
            self._reversed_var_mapping = {v: k for k, v in self.var_mapping.items()}
        return self._reversed_var_mapping
    
        
    def _check_consistent(self):
        assert len(self.all_input_lowers) == len(self.all_input_uppers) == len(self.all_output_lowers) == len(self), \
            print(len(self.all_input_lowers), len(self.all_input_uppers), len(self.all_output_lowers), len(self))
        assert len(self.all_cs) == len(self.all_rhs) == len(self)
        if not self.input_split:
            assert len(self.all_betas) == len(self.all_histories) == len(self)
            assert len(self.all_lower_bounds) == len(self.all_upper_bounds) 
            assert all([len(_) == len(self) for _ in self.all_lower_bounds.values()])
            assert all([len(_) == len(self) for _ in self.all_upper_bounds.values()])
            assert all([len(_) == len(self) for _ in self.all_lAs.values()])
            if self.use_restart:
                assert len(self.all_sat_solvers) == len(self), print(f'len(self.all_sat_solvers)={len(self.all_sat_solvers)}, len(self)={len(self)}')


    def pick_out(self, batch, device='cpu'):
        assert batch > 0
        batch = min(len(self), batch)
        self.visited += batch

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()

        # input bounds
        new_input_lowers = self.all_input_lowers.pop(batch).to(device=device, non_blocking=True)
        new_input_uppers = self.all_input_uppers.pop(batch).to(device=device, non_blocking=True)
        
        # output bounds
        new_output_lowers = self.all_output_lowers.pop(batch).to(device=device, non_blocking=True)
        
        # properties
        new_cs = self.all_cs.pop(batch).to(device=device, non_blocking=True)
        new_rhs = self.all_rhs.pop(batch).to(device=device, non_blocking=True)
        
        # alpha
        new_slopes = defaultdict(dict)
        for k, v in self.all_slopes.items():
            new_slopes[k] = {kk: vv.pop(batch).to(device=device, non_blocking=True) for (kk, vv) in v.items()}
            
        if self.input_split:
            # input splitting
            new_lower_bounds = new_upper_bounds = None
            new_masks = new_lAs = new_betas = new_histories = None
            new_sat_solvers = None
        else: 
            # hidden spliting 
            new_lAs = {k: lA.pop(batch).to(device=device, non_blocking=True) for (k, lA) in self.all_lAs.items()}
            new_lower_bounds = {k: lb.pop(batch).to(device=device, non_blocking=True) for (k, lb) in self.all_lower_bounds.items()}
            new_upper_bounds = {k: ub.pop(batch).to(device=device, non_blocking=True) for (k, ub) in self.all_upper_bounds.items()}
            
            # pop batch
            new_betas = self.all_betas[-batch:]
            new_histories = self.all_histories[-batch:]
            # remove batch
            self.all_betas = self.all_betas[:-batch]
            self.all_histories = self.all_histories[:-batch]
            
            if self.use_restart:
                new_sat_solvers = self.all_sat_solvers[-batch:]
                # new_sat_solvers = copy.deepcopy(self.all_sat_solvers[-batch:])
                self.all_sat_solvers = self.all_sat_solvers[:-batch]
            else:
                new_sat_solvers = None
            
            new_masks = compute_masks(
                lower_bounds=new_lower_bounds, 
                upper_bounds=new_upper_bounds, 
                device=device,
            )
            
            assert len(new_betas) == len(new_histories) == batch
            assert len(new_input_lowers) == len(new_input_lowers) == batch
            assert len(new_lower_bounds[list(new_lower_bounds.keys())[0]]) == batch
            assert len(new_upper_bounds[list(new_upper_bounds.keys())[0]]) == batch
            assert len(new_lAs[list(new_lAs.keys())[0]]) == batch 
        
        self._check_consistent()
        
        return AbstractResults(**{
            'output_lbs': new_output_lowers,
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers, 
            'masks': new_masks, 
            'lAs': new_lAs, 
            'lower_bounds': new_lower_bounds, 
            'upper_bounds': new_upper_bounds, 
            'slopes': new_slopes, 
            'betas': new_betas,
            'histories': new_histories,
            'cs': new_cs,
            'rhs': new_rhs,
            'sat_solvers': new_sat_solvers,
        })


    def add(self, domain_params, decisions=None):
        # assert decisions is not None
        batch = len(domain_params.input_lowers)
        assert batch > 0
        
        # unverified indices
        remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]

        # hidden splitting
        if not self.input_split:
            # bcp
            if self.use_restart:
                assert len(domain_params.sat_solvers) == batch
                assert decisions is not None
                
                extra_conflict_index = []
                for idx_ in remaining_index:
                    # bcp
                    new_sat_solver = self.boolean_propagation(
                        domain_params=domain_params, 
                        decisions=decisions, 
                        batch_idx=idx_,
                    )
                    if new_sat_solver is None:
                        extra_conflict_index.append(idx_)
                        continue
                            
                    self.all_sat_solvers.append(new_sat_solver)
                    
                if len(extra_conflict_index):
                    logger.debug(f'BCP removes {len(extra_conflict_index)} domains')
                    assert len(extra_conflict_index) == len(list(set(extra_conflict_index)))
                    for eci in extra_conflict_index:
                        remaining_index = remaining_index[remaining_index != eci]
            
            # decision histories
            self.all_histories.extend([domain_params.histories[i] for i in remaining_index])
            self.all_betas.extend([domain_params.betas[i] for i in remaining_index])
            
            # conflict clauses
            self.save_conflict_clauses(
                domain_params=domain_params, 
                remaining_index=remaining_index,
            )
            
            # hidden bounds
            [v.append(domain_params.lower_bounds[k][remaining_index]) for k, v in self.all_lower_bounds.items()]
            [v.append(domain_params.upper_bounds[k][remaining_index]) for k, v in self.all_upper_bounds.items()]
            [v.append(domain_params.lAs[k][remaining_index]) for k, v in self.all_lAs.items()]
        
        # input bounds
        self.all_input_lowers.append(domain_params.input_lowers[remaining_index])
        self.all_input_uppers.append(domain_params.input_uppers[remaining_index])
        
        # output bounds
        self.all_output_lowers.append(domain_params.output_lbs[remaining_index])
        
        # properties
        self.all_cs.append(domain_params.cs[remaining_index])
        self.all_rhs.append(domain_params.rhs[remaining_index])
        
        # alpha
        [vv.append(domain_params.slopes[k][kk][:,:,remaining_index]) for (k, v) in self.all_slopes.items() for (kk, vv) in v.items()]
            
        # checking
        self._check_consistent()
        

    def __len__(self):
        return len(self.all_input_lowers)


    @property
    def minimum_lowers(self):
        indices = (self.all_output_lowers - self.all_rhs).max(dim=1)[0].argsort()
        if len(indices):
            return self.all_output_lowers[indices[0]].max().detach().item()
        return 1e-6


    def pick_out_worst_domains(self, batch, device='cpu'):
        indices = (self.all_output_lowers - self.all_rhs).max(dim=1)[0].argsort()[:batch]

        new_lower_bounds = {k: v[indices].to(device=device, non_blocking=True) for k, v in self.all_lower_bounds.items()}
        new_upper_bounds = {k: v[indices].to(device=device, non_blocking=True) for k, v in self.all_upper_bounds.items()}

        self._check_consistent()
        
        return AbstractResults(**{
            'lower_bounds': new_lower_bounds, 
            'upper_bounds': new_upper_bounds, 
        })
        
        
    def update_refined_bounds(self, domain_params):
        # updating
        for key in domain_params.lower_bounds:
            orig_shape = self.all_lower_bounds[key].size()[1:] # skip batch dim

            self.all_lower_bounds[key].copy_(
                torch.where(
                    domain_params.lower_bounds[key].view(orig_shape) > self.all_lower_bounds[key].data, 
                    domain_params.lower_bounds[key].view(orig_shape), 
                    self.all_lower_bounds[key].data
                )
            )
            
            self.all_upper_bounds[key].copy_(
                torch.where(
                    domain_params.upper_bounds[key].view(orig_shape) < self.all_upper_bounds[key].data, 
                    domain_params.upper_bounds[key].view(orig_shape), 
                    self.all_upper_bounds[key].data
                )
            )

        # checking
        self._check_consistent()
        
    
    @torch.no_grad()
    def count_unstable_neurons(self):
        if self.all_lower_bounds is None:
            return None
        
        if not len(self):
            return None
        
        if not len(self.net.relus):
            return None
        
        new_masks = compute_masks(
            lower_bounds={k: v.data for k, v in self.all_lower_bounds.items()}, 
            upper_bounds={k: v.data for k, v in self.all_upper_bounds.items()}, 
            device='cpu',
        )
        # print(len(self), [_.shape for _ in new_masks.values()])
        n_unstable = sum([_.sum() for _ in new_masks.values()]).int()
        return n_unstable // len(self)

        
    from .util import init_sat_solver, update_hidden_bounds_histories, boolean_propagation, save_conflict_clauses