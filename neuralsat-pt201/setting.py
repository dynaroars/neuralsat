import numpy as np
import torch

try:
    import gurobipy as grb
    grb.Model('test')
    USE_GUROBI = True
except:
    USE_GUROBI = False
    

class GlobalSettings:

    def __init__(self):
        
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart
        self.use_restart = 1
        
        self.max_hidden_branches = 5000
        self.max_hidden_visited_branches = 20000
        
        self.max_input_branches  = 100000
        self.max_input_visited_branches = 300000
        
        self.max_restart_runtime = 50.0
        
        # stabilize
        self.use_mip_tightening = 1
        self.mip_tightening_patience = 10
        self.mip_tightening_timeout_per_neuron = 15.0
        self.mip_tightening_topk = 64
        
        # attack
        self.use_attack = 1
        self.use_mip_attack = 0 # in progress
        
        self.attack_interval = 10
        
        # timing statistic
        self.use_timer = 0
        
        # property
        self.safety_property_threshold = 0.5 # threshold for input/hidden splitting
        
        # motivation example
        self.test = 0
        
        # abstraction
        self.share_alphas = 0
        self.backward_batch_size = np.inf
        self.forward_max_dim = 10000
        self.forward_dynamic = 0

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def setup_test(self):
        self.max_hidden_branches = 100
        self.max_hidden_visited_branches = 200
        self.use_mip_tightening = 0
        self.use_restart = 1
        self.use_attack = 1
        self.test = 1
    
    def setup(self, args):
        if args is not None:
            if hasattr(args, 'disable_restart'):
                self.use_restart = args.disable_restart
            if hasattr(args, 'disable_stabilize'):
                self.use_mip_tightening = args.disable_stabilize and USE_GUROBI
        else:
            self.use_mip_tightening = USE_GUROBI
        
        # FIXME: remove after debugging
        # self.max_hidden_visited_branches = 100
        # self.use_timer = 1
        self.use_attack = 0
        self.use_restart = 0
        self.use_mip_tightening = 0
        # self.max_input_visited_branches = 100000
        # self.mip_tightening_timeout_per_neuron = 2.0
        # self.backward_batch_size = 256
        # self.max_restart_runtime = 20.0
        # self.forward_dynamic = 1
        # self.forward_max_dim = 100
            
        
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
