import numpy as np
import torch
import time

from util.network.read_onnx import custom_quirks

class DnfObjectives:
    
    "List of objectives"
    
    def __init__(self, objectives, input_shape, is_nhwc) -> None:
        self.objectives = objectives
        self.is_nhwc = is_nhwc
        self.input_shape = input_shape
        
        self._extract()
        
        self.num_used = 0
        
        
    def __len__(self):
        return len(self.lower_bounds[self.num_used:])
    
    
    def pop(self, batch):
        if isinstance(self.cs, torch.Tensor):
            batch = min(batch, len(self))
        else:
            batch = 1
        # print('\t- popping:', batch)
        class TMP:
            pass
        
        # f32
        lower_bounds = self.lower_bounds[self.num_used : self.num_used + batch]
        upper_bounds = self.upper_bounds[self.num_used : self.num_used + batch]
        
        # f64
        lower_bounds_f64 = self.lower_bounds_f64[self.num_used : self.num_used + batch]
        upper_bounds_f64 = self.upper_bounds_f64[self.num_used : self.num_used + batch]
        
        if self.is_nhwc:
            n_, c_, h_, w_ = self.input_shape
            orig_input_shape = (-1, h_, w_, c_)
            # f32
            lower_bounds = lower_bounds.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            upper_bounds = upper_bounds.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            # f64
            lower_bounds_f64 = lower_bounds_f64.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            upper_bounds_f64 = upper_bounds_f64.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            
            assert torch.all(lower_bounds <= upper_bounds)
            assert torch.all(lower_bounds_f64 <= upper_bounds_f64)
        
        objective = TMP()
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
        
    
    def _extract(self):
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

        # properties
        if all([_.shape[0] == self.cs[0].shape[0] for _ in self.cs]):
            self.cs = torch.stack(self.cs)
            self.cs_f64 = torch.stack(self.cs_f64)
        if all([_.shape[0] == self.rhs[0].shape[0] for _ in self.rhs]):
            self.rhs = torch.stack(self.rhs)
            self.rhs_f64 = torch.stack(self.rhs_f64)
            
            
    def add(self, objective):
        self.num_used -= len(objective.cs)
        

    def get_info(self):
        return self.cs, self.rhs
    

    def get_info_f64(self):
        return self.cs_f64, self.rhs_f64
    
    
class Objective:
    
    "Single objective in CNF"
    
    def __init__(self, prop) -> None:
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
        # FIXME: unsound errors
        
        self._extract(mat)
        
        
    def _extract(self, mat) -> None:
        # print('preprocess vnnlib spec')
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
    
    def get_info(self):
        return self.cs, self.rhs
    
    
    def get_info_f64(self):
        return self.cs_f64, self.rhs_f64
        
