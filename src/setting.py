import torch
import json
import os

try:
    import gurobipy as grb
    grb.Model('test')
    USE_GUROBI = True
except:
    print("[!] Gurobi License not found!")
    USE_GUROBI = False
    
from configure.advanced import BaseSettings, AdvancedSettings

class GlobalSettings(BaseSettings):

    def __init__(self):
        # data precision
        torch.set_default_dtype(torch.float32)
        
        # restart
        self.use_restart = True
        
        # stabilize
        self.use_mip_tightening = True
        
        # attack
        self.use_attack = True
        self.attack_interval = 10
        
        # mip verify
        self.use_mip_verify = True 
        
        # threshold for input/hidden splitting: 
        self.input_splitting_threshold = 0.5  # > 0.5: use input splitting
        self.hidden_splitting_threshold = 0.01  # < 0.01: use hidden splitting
        self.safety_num_input_perturbed = 200 # < 200: use input splitting
        
        # preprocess
        self.skip_preprocess = False
        
        # proof
        self.use_save_reasoning_step = False
        
        # early stopping
        self.max_iterations = 1e9
        self.skip_initial_worst_bound = -1e6
        self.max_domains = 1e9
        
        # decomposition
        self.use_decompose = False
        
    
    def _add_advanced_settings(self, args=None):
        advanced = AdvancedSettings(args)
        for key, value in advanced.__dict__.items():
            if not key.startswith('_') and key != 'advanced_settings':
                setattr(self, key, value)
    
    def setup(self, args=None):
        if args is not None:
            if hasattr(args, 'disable_attack'):
                self.use_attack = args.disable_attack
            if hasattr(args, 'disable_restart'):
                self.use_restart = args.disable_restart
            if hasattr(args, 'disable_stabilize'):
                self.use_mip_tightening = args.disable_stabilize and USE_GUROBI
        else:
            self.use_mip_tightening = USE_GUROBI
        
        # add advanced settings
        self._add_advanced_settings(args)
        
        # TODO: remove this
        # self.use_mip_verify = False
        # self.use_mip_tightening = False
        # self.use_restart = False
        # self.use_attack = False
        # self.mip_tightening_timeout_per_neuron = 2.0
        # self.restart_visited_hidden_branches = 100
        # self.share_alphas = True
        # self.skip_preprocess = False
        
        # load specific settings from json
        if args is not None and args.setting_file is not None:
            assert os.path.exists(args.setting_file), f"Setting file not found: {args.setting_file=}"
            settings = json.load(open(args.setting_file))
            for key, value in settings.items():
                assert hasattr(self, key), f"Unknown setting: {key=}"
                setattr(self, key, value)

        if self.use_save_reasoning_step:
            torch.set_default_dtype(torch.float64)
            print(f'[!] Using float64 for proof generation')
            
Settings = GlobalSettings()
