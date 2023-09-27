import torch

class GlobalSettings:

    def __init__(self):
        
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart
        self.use_restart = 1
        
        self.max_hidden_branches = 5000
        self.max_hidden_visited_branches = 20000
        
        self.max_input_branches  = 1e5
        self.max_input_visited_branches = 5e6
        
        # stabilize
        self.use_mip_tightening = 1
        self.mip_tightening_patience = 10
        
        # attack
        self.use_attack = 1
        self.use_mip_attack = 0
        
        self.attack_interval = 10
        
        # optimization
        self.use_hidden_bounds_optimization = 0
        self.hidden_bounds_optimization_interval = 1
        # TODO: haven't worked yet, disable for now
        # assert (not self.use_hidden_bounds_optimization)
        
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
        self.use_mip_tightening = 1
        self.use_restart = 0
        self.use_attack = 1
        self.test = 1
    
    def setup(self, args):
        self.use_restart = args.disable_restart
        self.use_mip_tightening = args.disable_stabilize
        
    def __repr__(self):
        return (
            '\n[!] Current settings:\n'
            f'\t- max_hidden_branches           : {int(self.max_hidden_branches)}\n'
            f'\t- max_hidden_visited_branches   : {int(self.max_hidden_visited_branches)}\n'
            # f'\t- max_input_branches            : {int(self.max_input_branches)}\n'
            # f'\t- max_input_visited_branches    : {int(self.max_input_visited_branches)}\n'
            f'\t- use_attack                    : {bool(self.use_attack)}\n'
            f'\t- use_restart                   : {bool(self.use_restart)}\n'
            f'\t- use_stabilize                 : {bool(self.use_mip_tightening)}\n'
            # f'\t- test                          : {bool(self.test)}\n'
            f'\n'
        )

Settings = GlobalSettings()
