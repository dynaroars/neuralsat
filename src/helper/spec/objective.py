from beartype import beartype
import numpy as np
import torch

from helper.network.read_onnx import custom_quirks
from helper.spec.read_vnnlib import read_vnnlib


class Objective:
    
    "Single objective in CNF"
    
    @beartype
    def __init__(self: 'Objective', prop: tuple[list, tuple[np.ndarray, np.ndarray]]) -> None:
        input_bounds, mat = prop
        self.dtype = torch.get_default_dtype()
        
        bounds = torch.tensor(input_bounds, dtype=self.dtype)
        self.lower_bound = bounds[:, 0]
        self.upper_bound = bounds[:, 1]
        assert torch.all(self.lower_bound <= self.upper_bound)
        
        bounds_f64 = torch.tensor(input_bounds, dtype=torch.float64)
        self.lower_bound_f64 = bounds_f64[:, 0]
        self.upper_bound_f64 = bounds_f64[:, 1]
        assert torch.all(self.lower_bound_f64 <= self.upper_bound_f64)
        
        self._extract(mat)
        
        
    @beartype
    def _extract(self: 'Objective', mat: tuple[np.ndarray, np.ndarray]) -> None:
        assert len(mat) == 2, print(len(mat))
        prop_mat, prop_rhs = mat

        # f32
        self.cs = torch.tensor(prop_mat, dtype=self.dtype)
        self.rhs = torch.tensor(prop_rhs, dtype=self.dtype)
        
        # f64
        self.cs_f64 = torch.tensor(prop_mat, dtype=torch.float64)
        self.rhs_f64 = torch.tensor(prop_rhs, dtype=torch.float64)
        
        if custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
            assert (self.rhs == 0).all()
    
    
    @beartype
    def get_info(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cs, self.rhs
    

    @beartype
    def get_info_f64(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cs_f64, self.rhs_f64
        

class DnfObjectives:
    
    "List of objectives"
    
    @beartype
    def __init__(self, objectives: list[Objective], input_shape: tuple) -> None:
        self.objectives = objectives
        self.input_shape = input_shape
        
        self._extract()
        
        self.num_used = 0
        
        
    @beartype
    def __len__(self: 'DnfObjectives') -> int:
        return len(self.lower_bounds[self.num_used:])
    
    
    @beartype
    def pop(self: 'DnfObjectives', batch: int):
        if isinstance(self.cs, torch.Tensor):
            batch = min(batch, len(self))
        else:
            batch = 1
        # print('\t- popping:', batch)
        class TMP:
            pass
        
        assert len(self.lower_bounds) == len(self.upper_bounds) == len(self.cs) == len(self.rhs)
        assert len(self.lower_bounds_f64) == len(self.upper_bounds_f64) == len(self.cs_f64) == len(self.rhs_f64)
        assert len(self.cs) == len(self.cs_f64)

        # f32
        lower_bounds = self.lower_bounds[self.num_used : self.num_used + batch]
        upper_bounds = self.upper_bounds[self.num_used : self.num_used + batch]
        
        # f64
        lower_bounds_f64 = self.lower_bounds_f64[self.num_used : self.num_used + batch]
        upper_bounds_f64 = self.upper_bounds_f64[self.num_used : self.num_used + batch]
        
        objective = TMP()
        
        # indices for distinguishing restart
        objective.ids = self.ids[self.num_used : self.num_used + batch]
        
        # input bounds
        objective.lower_bounds = lower_bounds
        objective.upper_bounds = upper_bounds
        
        objective.lower_bounds_f64 = lower_bounds_f64
        objective.upper_bounds_f64 = upper_bounds_f64
        
        # specs
        objective.cs = self.cs[self.num_used : self.num_used + batch]
        objective.cs_f64 = self.cs_f64[self.num_used : self.num_used + batch]
        if not isinstance(objective.cs, torch.Tensor):
            objective.cs = torch.cat(objective.cs)[None]
            objective.cs_f64 = torch.cat(objective.cs_f64)[None]
            
        objective.rhs = self.rhs[self.num_used : self.num_used + batch]
        objective.rhs_f64 = self.rhs_f64[self.num_used : self.num_used + batch]
        if not isinstance(objective.rhs, torch.Tensor):
            objective.rhs = torch.cat(objective.rhs)[None]
            objective.rhs_f64 = torch.cat(objective.rhs_f64)[None]
            
        self.num_used += batch
        
        assert objective.cs.dtype == objective.rhs.dtype == objective.lower_bounds.dtype == objective.upper_bounds.dtype
        assert objective.cs_f64.dtype == objective.rhs_f64.dtype == objective.lower_bounds_f64.dtype == objective.upper_bounds_f64.dtype

        return objective
        
    
    @beartype
    def _extract(self: 'DnfObjectives') -> None:
        self.cs, self.rhs = [], []
        self.lower_bounds, self.upper_bounds = [], []
        
        self.cs_f64, self.rhs_f64 = [], []
        self.lower_bounds_f64, self.upper_bounds_f64 = [], []
        
        for objective in self.objectives:
            # f32
            self.lower_bounds.append(objective.lower_bound)
            self.upper_bounds.append(objective.upper_bound)
            
            c_, rhs_ = objective.get_info()
            self.cs.append(c_)
            self.rhs.append(rhs_)
            
            # f64
            self.lower_bounds_f64.append(objective.lower_bound_f64)
            self.upper_bounds_f64.append(objective.upper_bound_f64)

            c_f64, rhs_f64 = objective.get_info_f64()
            self.cs_f64.append(c_f64)
            self.rhs_f64.append(rhs_f64)
            
        # input bounds
        self.lower_bounds = torch.stack(self.lower_bounds)
        self.upper_bounds = torch.stack(self.upper_bounds)
        
        self.lower_bounds_f64 = torch.stack(self.lower_bounds_f64)
        self.upper_bounds_f64 = torch.stack(self.upper_bounds_f64)

        # ids
        magic_number = 3
        self.ids = torch.arange(0, len(self.cs)) + magic_number
        
        assert torch.all(self.lower_bounds <= self.upper_bounds)
        assert torch.all(self.lower_bounds_f64 <= self.upper_bounds_f64)
            
        # properties
        if all([_.shape[0] == self.cs[0].shape[0] for _ in self.cs]):
            self.cs = torch.stack(self.cs)
            self.cs_f64 = torch.stack(self.cs_f64)
        if all([_.shape[0] == self.rhs[0].shape[0] for _ in self.rhs]):
            self.rhs = torch.stack(self.rhs)
            self.rhs_f64 = torch.stack(self.rhs_f64)
            
            
    @beartype
    def add(self: 'DnfObjectives', objective) -> None:
        self.num_used -= len(objective.cs)
        


def parse_vnnlib(vnnlib_path, input_shape):
    vnnlibs = read_vnnlib(vnnlib_path)
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape)
    return objectives
