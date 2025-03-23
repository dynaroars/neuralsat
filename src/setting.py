import numpy as np
import torch
import os

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
        
        self.restart_current_hidden_branches = 1000
        self.restart_visited_hidden_branches = 20000
        
        self.restart_current_input_branches  = 100000
        self.restart_visited_input_branches = 300000
        
        self.restart_max_runtime = 50.0
        
        # cpu stabilize
        self.use_mip_tightening = 1
        self.mip_tightening_patience = 10
        self.mip_tightening_timeout_per_neuron = 15.0
        self.mip_tightening_topk = 64
        
        # gpu stabilize
        self.use_gpu_tightening = 0
        self.gpu_tightening_current_hidden_branches = 1000
        self.gpu_tightening_visited_hidden_branches = 5000
        self.gpu_tightening_timeout = 10.0
        self.gpu_tightening_patience = 10
        self.gpu_tightening_topk = 500
        
        # attack
        self.use_attack = 1
        self.use_mip_attack = 0 # in progress
        self.attack_interval = 10
        
        # MIP verify
        self.use_mip_verify = 1
        self.mip_verify_threshold = 2
        
        # timing statistic
        self.use_timer = 0
        
        # property
        self.safety_property_threshold = 0.5 # threshold for input/hidden splitting
        
        # motivation example
        self.test = 0
        
        # preprocess
        self.skip_preprocess = 0
        
        # abstraction
        self.share_alphas = 0 
        self.backward_batch_size = np.inf
        self.forward_max_dim = 10000
        self.forward_dynamic = 0
        
        # decomposition
        self.use_decompose = 0
        self.use_decompose_incomplete = 0
        self.init_abstraction_method = 'crown-optimized'
        self.min_layer = 100
        self.subverifier_decision_method = 'smart'
        self.verify_candidate_batch = 3
        self.verify_candidate_num = 128
        self.verify_last_timeout = 200.0
        self.verify_interm_timeout = 10.0
        
        # debug
        self.max_iterations = 1e9
        self.skip_initial_worst_bound = -1e6

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def setup_test(self):
        self.restart_current_hidden_branches = 100
        self.restart_visited_hidden_branches = 200
        self.use_mip_tightening = 0
        self.use_restart = 1
        self.use_attack = 1
        self.test = 1
        self.use_gpu_tightening = 0
        
    
    def setup(self, args):
        if args is not None:
            if hasattr(args, 'disable_restart'):
                self.use_restart = args.disable_restart
            if hasattr(args, 'disable_stabilize'):
                self.use_mip_tightening = args.disable_stabilize and USE_GUROBI
        else:
            self.use_mip_tightening = USE_GUROBI
        
        # FIXME: remove after debugging
        # self.use_gpu_tightening = 1
        # self.gpu_tightening_timeout = 2
        # self.restart_visited_hidden_branches = 100
        # self.use_timer = 1
        # self.use_mip_verify = 0
        # self.use_attack = 0
        # self.use_restart = 0
        # self.use_mip_tightening = 0
        # self.restart_visited_input_branches = 100000
        # self.mip_tightening_timeout_per_neuron = 1.0
        # self.backward_batch_size = 256
        # self.restart_max_runtime = 20.0
        # self.forward_dynamic = 1
        # self.forward_max_dim = 100
        # self.share_alphas = 0 # sharing alphas may lose precision
        # self.max_iterations = 100
        # self.skip_initial_worst_bound = -5.0
            
        
    def setup_dec_test(self, args):
        print('[+] setup_dec_test')
        self.use_decompose = 1
        self.min_layer = 4
        self.share_alphas = 1 # sharing alphas may lose precision
        self.init_abstraction_method = 'backward'
        self.skip_preprocess = 1
        self.use_decompose_incomplete = 0

        self.verify_candidate_batch = 1
        self.verify_candidate_num = 3
        self.verify_last_timeout = 20.0
        self.verify_interm_timeout = 5.0
    
    def setup_crown(self, args):
        print('[+] setup_crown')
        self.use_decompose = 1
        self.min_layer = 1
        self.share_alphas = 0
        # self.init_abstraction_method = 'backward'
        self.init_abstraction_method = 'crown-optimized'
        self.skip_preprocess = 1
        self.verify_candidate_batch = 1
        
    def setup_vae_small(self, args):
        print('[+] setup_vae_small')
        self.use_decompose = 1
        self.min_layer = 1
        self.share_alphas = 0 
        self.init_abstraction_method = 'crown-optimized'
        self.skip_preprocess = 1
        
        
    def setup_vae_large(self, args):
        print('[+] setup_vae_large')
        self.use_decompose = 1
        self.min_layer = 1
        self.share_alphas = 1 # sharing alphas may lose precision
        # self.init_abstraction_method = 'backward'
        self.init_abstraction_method = 'crown-optimized'
        self.subverifier_decision_method = 'greedy'
        self.skip_preprocess = 1
    
    
    def setup_resnet_small(self, args):
        print('[+] setup_resnet_small')
        self.use_decompose = 0
        self.min_layer = 100
        self.share_alphas = 1 # sharing alphas may lose precision
        self.skip_preprocess = 1
        
        
    def setup_resnet_small_oom(self, args):
        print('[+] setup_resnet_small')
        self.use_decompose = 1
        self.min_layer = 6
        self.share_alphas = 1 # sharing alphas may lose precision
        self.init_abstraction_method = 'backward'
        self.skip_preprocess = 1
        
    def setup_resnet_large(self, args):
        print('[+] setup_resnet_large')
        self.use_decompose = 1
        self.min_layer = 6
        self.share_alphas = 1 # sharing alphas may lose precision
        self.init_abstraction_method = 'backward'
        self.skip_preprocess = 1
        self.verify_candidate_batch = 1
        self.use_decompose_incomplete = 1
        
        
    def __repr__(self):
        str = (
            '\n[!] Current settings:\n'
            f'\t- restart_current_hidden_branches        : {int(self.restart_current_hidden_branches)}\n'
            f'\t- restart_visited_hidden_branches        : {int(self.restart_visited_hidden_branches)}\n'
            f'\t- restart_current_input_branches         : {int(self.restart_current_input_branches)}\n'
            f'\t- restart_visited_input_branches         : {int(self.restart_visited_input_branches)}\n'
            f'\t- gpu_tightening_current_hidden_branches : {int(self.gpu_tightening_current_hidden_branches)}\n'
            f'\t- gpu_tightening_visited_hidden_branches : {int(self.gpu_tightening_visited_hidden_branches)}\n'
            f'\t- attack                                 : {bool(self.use_attack)}\n'
            f'\t- restart                                : {bool(self.use_restart)}\n'
            f'\t- stabilize (CPU)                        : {bool(self.use_mip_tightening)}\n'
            f'\t- stabilize (GPU)                        : {bool(self.use_gpu_tightening)}\n'
            f'\t- assertion                              : {bool(os.environ.get("NEURALSAT_ASSERT"))}\n'
            f'\t- debug                                  : {bool(os.environ.get("NEURALSAT_DEBUG"))}\n'
        )
        if Settings.use_decompose:
            str += (
                f'\n[!] Decomposition:\n'
                f'\t- use_decompose                          : {bool(self.use_decompose)}\n'
                f'\t- share_alphas                           : {bool(self.share_alphas)}\n'
                f'\t- skip_preprocess                        : {bool(self.skip_preprocess)}\n'
                f'\t- min_layer                              : {self.min_layer}\n'
                f'\t- init_abstraction_method                : {self.init_abstraction_method}\n'
                f'\t- subverifier_decision_method            : {self.subverifier_decision_method}\n'
                f'\t- use_decompose_incomplete               : {self.use_decompose_incomplete}\n'
                f'\n'
            )
        return str

Settings = GlobalSettings()
