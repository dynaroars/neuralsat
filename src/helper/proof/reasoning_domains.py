from __future__ import annotations
from beartype import beartype
import torch
import json
import os

from heuristic.util import _history_to_conflict_clause
from helper.misc.tensor_storage import TensorStorage
from helper.proof.create_aptp import create_aptp
from helper.misc.result import AbstractResults

class ReasoningStep:
    
    @beartype
    def __init__(self, 
                 input_lower: torch.Tensor, 
                 input_upper: torch.Tensor, 
                 output_lb: torch.Tensor,
                 lower_bound: dict, 
                 upper_bound: dict, 
                 c: torch.Tensor, 
                 rhs: torch.Tensor, 
                 history: dict | None) -> None:

        self.input_lower = input_lower
        self.input_upper = input_upper
        self.output_lb = output_lb
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.c = c
        self.rhs = rhs
        self.history = history
        
    @beartype
    def to_json(self):
        data = {
            'input_lower': self.input_lower.detach().cpu().numpy().tolist(),
            'input_upper': self.input_upper.detach().cpu().numpy().tolist(),
            'output_lower': self.output_lb.detach().cpu().numpy().tolist(),
            'c': self.c.detach().cpu().numpy().tolist(),
            'rhs': self.rhs.detach().cpu().numpy().tolist(),
            'hidden_lower': {k: v.detach().cpu().numpy().tolist() for k, v in self.lower_bound.items()},
            'hidden_upper': {k: v.detach().cpu().numpy().tolist() for k, v in self.upper_bound.items()},
            'history': {k: [v_.detach().cpu().numpy().tolist() if isinstance(v_, torch.Tensor) else v_ for v_ in v] for k, v in self.history.items()} if self.history is not None else None,
        }
        return data
    

class ReasoningDomains:
    
    @beartype
    def __init__(
        self, 
        objective_ids: torch.Tensor,
        input_lowers: torch.Tensor, 
        input_uppers: torch.Tensor, 
        output_lbs: torch.Tensor,
        lower_bounds: dict, 
        upper_bounds: dict, 
        cs: torch.Tensor, 
        rhs: torch.Tensor, 
        histories: list | None, 
        input_split: bool, 
        select_index: torch.Tensor,
        var_mapping: dict,
    ) -> None:
        
        # objective indices
        self.all_objective_ids = TensorStorage(objective_ids[select_index].cpu())
        
        # input bounds
        self.all_input_lowers = TensorStorage(input_lowers[select_index].cpu())
        self.all_input_uppers = TensorStorage(input_uppers[select_index].cpu())
    
        # hidden bounds
        self.all_lower_bounds = {k: TensorStorage(v[select_index].cpu()) for k, v in lower_bounds.items()}
        self.all_upper_bounds = {k: TensorStorage(v[select_index].cpu()) for k, v in upper_bounds.items()}
        
        # output bounds
        self.all_output_lowers = TensorStorage(output_lbs[select_index].cpu())
        
        # properties
        self.all_cs = TensorStorage(cs[select_index].cpu())
        self.all_rhs = TensorStorage(rhs[select_index].cpu())
        
        # hidden spliting 
        if not input_split:
            # decisions
            self.all_histories = [histories[_] for _ in select_index]
        
        self.input_split = input_split
        self.var_mapping = var_mapping
        
        self._check_consistent()
        
        
    @beartype
    def _check_consistent(self) -> None:
        # print('Checking domains:', len(self))
        assert len(self.all_input_lowers) == len(self.all_input_uppers) == len(self.all_output_lowers) == len(self)
        assert len(self.all_cs) == len(self.all_rhs) == len(self.all_objective_ids) == len(self)
        assert len(self.all_lower_bounds) == len(self.all_upper_bounds) 
        assert all([len(_) == len(self) for _ in self.all_lower_bounds.values()])
        assert all([len(_) == len(self) for _ in self.all_upper_bounds.values()])
        if not self.input_split:
            assert len(self.all_histories) == len(self), f'{len(self.all_histories)=} {len(self)=}'


    @beartype
    def add(self, domain_params: AbstractResults, select_index: torch.Tensor) -> None:
        assert len(domain_params.input_lowers) > 0
        
        # objective indices
        self.all_objective_ids.append(domain_params.objective_ids[select_index])
        
        # input bounds
        self.all_input_lowers.append(domain_params.input_lowers[select_index])
        self.all_input_uppers.append(domain_params.input_uppers[select_index])
        
        # hidden bounds
        [v.append(domain_params.lower_bounds[k][select_index]) for k, v in self.all_lower_bounds.items()]
        [v.append(domain_params.upper_bounds[k][select_index]) for k, v in self.all_upper_bounds.items()]
        
        # output bounds
        self.all_output_lowers.append(domain_params.output_lbs[select_index])
        
        # properties
        self.all_cs.append(domain_params.cs[select_index])
        self.all_rhs.append(domain_params.rhs[select_index])
        
        if not self.input_split:
            # decision histories
            self.all_histories.extend([domain_params.histories[i] for i in select_index])
            
        # checking
        self._check_consistent()
    

    @beartype
    def get_reasoning_steps(self) -> dict:
        device = 'cpu'
        
        batch = len(self)
        assert batch > 0
        
        # objective ids
        new_objective_ids = self.all_objective_ids.pop(batch).to(device=device, non_blocking=True)
        
        # input bounds
        new_input_lowers = self.all_input_lowers.pop(batch).to(device=device, non_blocking=True)
        new_input_uppers = self.all_input_uppers.pop(batch).to(device=device, non_blocking=True)

        # hidden bounds
        new_lower_bounds = {k: v.pop(batch).to(device=device, non_blocking=True) for k, v in self.all_lower_bounds.items()}
        new_upper_bounds = {k: v.pop(batch).to(device=device, non_blocking=True) for k, v in self.all_upper_bounds.items()}

        if not self.input_split:
            # decision
            new_histories = self.all_histories[-batch:]
            self.all_histories = self.all_histories[:-batch]
            
        # output
        new_output_lowers = self.all_output_lowers.pop(batch).to(device=device, non_blocking=True)
        
        # properties
        new_cs = self.all_cs.pop(batch).to(device=device, non_blocking=True)
        new_rhs = self.all_rhs.pop(batch).to(device=device, non_blocking=True)
        
        # proof
        list_objectives = new_objective_ids.unique().int()
        data = {
            'reasoning_steps': {
                int(i): [] for i in list_objectives
            }
        }
        
        for i in range(batch):
            oid = int(new_objective_ids[i].item())
            step = ReasoningStep(
                input_lower=new_input_lowers[i],
                input_upper=new_input_uppers[i],
                output_lb=new_output_lowers[i],
                lower_bound={k: v[i] for k, v in new_lower_bounds.items()},
                upper_bound={k: v[i] for k, v in new_upper_bounds.items()},
                history=new_histories[i] if not self.input_split else None,
                c=new_cs[i],
                rhs=new_rhs[i],
            )
            data['reasoning_steps'][oid].append(step.to_json())

        print(f'[!] Got {batch} reasoning steps')
        self._check_consistent()
        return data
            
    @beartype
    def export_json(self, path: str):
        # write
        data = self.get_reasoning_steps()
        os.remove(path) if os.path.exists(path) else None
        with open(path, 'w') as fp:
            json.dump(data, fp, indent=2)
        
    @beartype
    def export_aptp(self, output_dir: str):
        data = self.get_reasoning_steps()
        for oid, steps in data['reasoning_steps'].items():
            proof_tree = []
            cs, rhs, input_lower, input_upper = None, None, None, None
            for step in steps:
                if self.input_split:
                    proof_step = (torch.tensor(step['input_lower']), torch.tensor(step['input_upper']))
                else:
                    proof_step = [(-1 * lit) for lit in _history_to_conflict_clause(step['history'], self.var_mapping)]
                proof_tree.append(proof_step)
                if cs is None:
                    cs = step['c']
                    rhs = step['rhs']
                    input_lower = step['input_lower']
                    input_upper = step['input_upper']
            aptp_str = create_aptp(
                proof=proof_tree, 
                input_lower=torch.tensor(input_lower),
                input_upper=torch.tensor(input_upper),
                cnf_cs=torch.tensor(cs), 
                cnf_rhs=torch.tensor(rhs),
                input_split=self.input_split,
            )
            # print(aptp_str)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'proof_{oid}.aptp'), 'w') as fp:
                print(aptp_str, file=fp)
            
    @beartype
    def __len__(self) -> int:
        return len(self.all_input_lowers)