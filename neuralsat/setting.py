import torch

class GlobalSettings:

    def __init__(self):
        
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart thresholds, restart if exceeded
        self.max_hidden_branches = 1e5 
        self.max_hidden_visited_branches = 2e5
        
        self.max_input_branches  = 1e5
        self.max_input_visited_branches = 5e6
        
        # MIP refinement
        self.use_mip_refine = 0
        self.use_mip_refine_domain_bounds = 0
        
        # attack
        self.use_attack = 1
        self.attack_interval = 10
        
        self.use_mip_attack = 0
        
        # restart
        self.use_restart = 0
        
        # abstraction
        self.use_hidden_bounds_optimization = 0
        self.hidden_bounds_optimization_interval = 10
        
        # threshold for automatically switching between input and hidden splitting
        self.safety_property_threshold = 0.5

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


Settings = GlobalSettings()
