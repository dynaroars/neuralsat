import numpy as np
import torch
import time


class DnfObjectives:
    
    "List of objectives"
    
    def __init__(self, objectives: list) -> None:
        self.objectives = objectives
        
        self._extract()
        
        self.num_used = 0
        
        
    def __len__(self):
        return len(self.lower_bounds[self.num_used:])
    
    
    def pop(self, batch):
        batch = min(batch, len(self))
        # print('\t- popping:', batch)
        class TMP:
            pass
        objective = TMP()
        objective.lower_bounds = self.lower_bounds[self.num_used : self.num_used + batch]
        objective.upper_bounds = self.upper_bounds[self.num_used : self.num_used + batch]
        objective.cs = self.cs[self.num_used : self.num_used + batch]
        objective.rhs = self.rhs[self.num_used : self.num_used + batch]
        objective.true_labels = self.true_labels[self.num_used : self.num_used + batch]
        objective.target_labels = self.target_labels[self.num_used : self.num_used + batch]
        self.num_used += batch
        return objective
        
    
    def _extract(self):
        self.cs, self.rhs, self.true_labels, self.target_labels = [], [], [], []
        self.lower_bounds, self.upper_bounds = [], []
        
        for objective in self.objectives:
            self.lower_bounds.append(objective.lower_bound)
            self.upper_bounds.append(objective.upper_bound)

            c_, rhs_, true_label_, target_label_ = objective.get_info()
            self.cs.append(c_)
            self.rhs.append(rhs_)
            self.true_labels.append(true_label_)
            self.target_labels.append(target_label_)
            
        # input bounds
        self.lower_bounds = torch.stack(self.lower_bounds)
        self.upper_bounds = torch.stack(self.upper_bounds)
        
        # properties
        self.cs = torch.stack(self.cs, dim=0)
        self.rhs = torch.stack(self.rhs)
        self.true_labels = np.array(self.true_labels)
        self.target_labels = np.array(self.target_labels)
            
    
    def get_info(self):
        return self.cs, self.rhs, self.true_labels, self.target_labels
    
    
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
        return self.cs, self.rhs, self.true_labels, self.target_labels
        
