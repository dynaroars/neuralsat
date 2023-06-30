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
        
        lower_bounds = self.lower_bounds[self.num_used : self.num_used + batch]
        upper_bounds = self.upper_bounds[self.num_used : self.num_used + batch]
        if self.is_nhwc:
            n_, c_, h_, w_ = self.input_shape
            orig_input_shape = (-1, h_, w_, c_)
            lower_bounds = lower_bounds.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            upper_bounds = upper_bounds.view(orig_input_shape).permute(0, 3, 1, 2).flatten(1)
            assert torch.all(lower_bounds <= upper_bounds)
        
        objective = TMP()
        # input bounds
        objective.lower_bounds = lower_bounds
        objective.upper_bounds = upper_bounds
        
        # specs
        objective.cs = self.cs[self.num_used : self.num_used + batch]
        if not isinstance(objective.cs, torch.Tensor):
            objective.cs = torch.cat(objective.cs)[None]
            
        objective.rhs = self.rhs[self.num_used : self.num_used + batch]
        if not isinstance(objective.rhs, torch.Tensor):
            objective.rhs = torch.cat(objective.rhs)[None]
            
        self.num_used += batch
        return objective
        
    
    def _extract(self):
        self.cs, self.rhs = [], []
        self.lower_bounds, self.upper_bounds = [], []
        
        for objective in self.objectives:
            self.lower_bounds.append(objective.lower_bound)
            self.upper_bounds.append(objective.upper_bound)

            c_, rhs_ = objective.get_info()
            self.cs.append(c_)
            self.rhs.append(rhs_)
            
        # input bounds
        self.lower_bounds = torch.stack(self.lower_bounds)
        self.upper_bounds = torch.stack(self.upper_bounds)
        
        # properties
        if all([_.shape[0] == self.cs[0].shape[0] for _ in self.cs]):
            self.cs = torch.stack(self.cs)
        if all([_.shape[0] == self.rhs[0].shape[0] for _ in self.rhs]):
            self.rhs = torch.stack(self.rhs)
            
            
    def add(self, objective):
        self.num_used -= len(objective.cs)
        

    def get_info(self):
        return self.cs, self.rhs
    
    
class Objective:
    
    "Single objective in CNF"
    
    def __init__(self, prop) -> None:
        bounds, self.mat = prop
        self.dtype = torch.get_default_dtype()
        
        bounds = torch.tensor(bounds, dtype=self.dtype)
        self.lower_bound = bounds[:, 0]
        self.upper_bound = bounds[:, 1]
        assert torch.all(self.lower_bound <= self.upper_bound)
        
        self._extract()
        
        
    def _extract(self) -> None:
        # print('preprocess vnnlib spec')
        assert len(self.mat) == 2, print(len(self.mat))
        prop_mat, prop_rhs = self.mat

        self.cs = torch.tensor(prop_mat, dtype=self.dtype)
        self.rhs = torch.tensor(prop_rhs, dtype=self.dtype)
        
        if custom_quirks.get('Softmax', {}).get('skip_last_layer', False):
            assert (self.rhs == 0).all()

        true_labels, target_labels = [], []
        for m in prop_mat:
            true_label = np.where(m == 1)[-1]
            if len(true_label) != 0:
                assert len(true_label) == 1
                true_labels.append(true_label[0])
            else:
                true_labels.append(None)

            target_label = np.where(m == -1)[-1]
            if len(target_label) != 0:
                assert len(target_label) == 1
                target_labels.append(target_label[0])
            else:
                target_labels.append(None)

        self.true_labels = np.array(true_labels)
        self.target_labels = np.array(target_labels)

    
    def get_info(self):
        return self.cs, self.rhs
        
