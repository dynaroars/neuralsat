import torch

class GlobalSettings:

    def __init__(self):
        
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart thresholds, restart if exceeded
        self.max_hidden_branches = 1e5 
        self.max_hidden_visited_branches = 2e6
        
        self.max_input_branches  = 1e5
        self.max_input_visited_branches = 5e6
        
        # MIP refinement
        self.use_mip_refine = 0
        self.use_mip_refine_domain_bounds = 0
        
        self.use_mip_tightening = 0
        self.mip_tightening_patience = 10
        
        # attack
        self.use_attack = 1
        self.attack_interval = 10
        
        self.use_mip_attack = 1
        
        # restart
        self.use_restart = 0
        
        # optimization
        self.use_hidden_bounds_optimization = 0
        self.hidden_bounds_optimization_interval = 10
        # TODO: haven't worked yet, disable for now
        assert (not self.use_hidden_bounds_optimization)
        
        # threshold for automatically switching between input and hidden splitting
        self.safety_property_threshold = 0.5
        
        # motivation example
        self.test = 0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def setup_test(self):
        self.max_hidden_branches = 1
        self.max_hidden_visited_branches = 2
        self.use_mip_refine = 0
        self.use_mip_tightening = 1
        self.use_restart = 0
        self.use_attack = 1
        self.test = 1
        

    def __repr__(self):
        return (
            '\n[!] Current settings:\n'
            f'\t- max_hidden_branches           : {int(self.max_hidden_branches)}\n'
            f'\t- max_hidden_visited_branches   : {int(self.max_hidden_visited_branches)}\n'
            # f'\t- max_input_branches            : {int(self.max_input_branches)}\n'
            # f'\t- max_input_visited_branches    : {int(self.max_input_visited_branches)}\n'
            f'\t- use_attack                    : {bool(self.use_attack)}\n'
            f'\t- use_restart                   : {bool(self.use_restart)}\n'
            f'\t- use_mip_refine                : {bool(self.use_mip_refine)}\n'
            f'\t- use_mip_tightening            : {bool(self.use_mip_tightening)}\n'
            f'\t- test                          : {bool(self.test)}\n'
            f'\n'
        )

Settings = GlobalSettings()
