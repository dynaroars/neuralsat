import numpy as np

class BaseSettings:
    
    def __init__(self):
        pass
    
    def __repr__(self):
        str = f'\n [!] {self.__class__.__name__}:\n'
        for k, v in self.__dict__.items():
            str += f'\t- {k:<40}: {v}\n'
        return str

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
    

class RestartSettings(BaseSettings):
    
    def __init__(self, args=None):
        # hidden splitting
        self.restart_current_hidden_branches = 2000
        self.restart_visited_hidden_branches = 20000
        
        # input splitting
        self.restart_current_input_branches = 20000
        self.restart_visited_input_branches = 200000
        
        # restart time threshold
        self.restart_max_runtime = 60.0
        self.restart_max_runtime_percentage = 0.4
        

class MIPSettings(BaseSettings):
    
    def __init__(self, args=None):
        # cpu stabilize
        self.mip_tightening_patience = 10
        self.mip_tightening_timeout_per_neuron = 15.0
        self.mip_tightening_topk = 64
        
        # gpu stabilize
        self.use_gpu_tightening = False # TODO: in progress
        self.gpu_tightening_current_hidden_branches = 1000
        self.gpu_tightening_visited_hidden_branches = 5000
        self.gpu_tightening_timeout = 10.0
        self.gpu_tightening_patience = 10
        self.gpu_tightening_topk = 500
        
        # mip verify
        self.use_mip_attack = False # TODO: in progress
        self.mip_verify_threshold = 2
        
        
class AbstractionSettings(BaseSettings):
    
    def __init__(self, args=None):
        self.default_abstraction_method = 'backward'
        self.share_alphas = False 
        self.backward_batch_size = np.inf
        self.forward_max_dim = 10000
        self.forward_dynamic = False
        
        
class DecompositionSettings(BaseSettings):
    
    def __init__(self, args=None):
        self.use_decompose_incomplete = False
        self.init_abstraction_method = 'crown-optimized'
        self.subverifier_decision_method = 'smart'
        self.verify_candidate_batch = 3
        self.verify_last_timeout = 200.0
        self.verify_interm_timeout = 20.0
        self.sequential_batch = 100
        self.use_sequential_abstract_forward = False
        
        self.verify_extra_opts = {'sparse_intermediate_bounds': True}
        self.verify_splitting_strategy = 'hidden'
        self.use_extra_substitution = False # TODO: in progress
        
        self.verify_max_iteration = 4
        self.verify_candidate_num = 128
        self.verify_interpolate_factor = 0.5
        
        self.setup_decompose(args)
        
    def setup_decompose(self, args):
        if not hasattr(args, 'category'):
            return
        
        self.use_attack = False
        self.use_restart = False
        self.use_mip_tightening = False
        self.share_alphas = False 
        self.skip_preprocess = True
        
        category = args.category.lower()
        if category in ['resnet6']:
            self.setup_resnet_small(args)
        elif category in ['resnet12', 'resnet18']:
            self.setup_resnet_large(args)
        elif category in ['resnet36']:
            self.setup_resnet_extra_large(args)
        elif category in ['vae_base', 'vae_wide']:
            self.setup_vae_base(args)
        elif category in ['vae_deep']:
            self.setup_vae_deep(args)
        else:
            raise ValueError(f'[!] Unsupported settings for {category=}')
        
        
    def setup_vae_base(self, args):
        print('[+] setup_vae_base')
        self.use_decompose = True
        self.share_alphas = False 
        self.init_abstraction_method = 'crown-optimized'
        
    def setup_vae_deep(self, args):
        print('[+] setup_vae_deep')
        self.use_decompose = True
        self.share_alphas = True # sharing alphas may lose precision
        self.init_abstraction_method = 'crown-optimized'
        self.subverifier_decision_method = 'greedy'
        self.verify_candidate_batch = 32
        self.verify_interm_timeout = 30.0
        self.verify_last_timeout = 300.0
    
    def setup_resnet_small(self, args):
        print('[+] setup_resnet_small')
        self.use_decompose = False
        self.share_alphas = True # sharing alphas may lose precision
        self.use_restart = True
        
    def setup_resnet_large(self, args):
        print('[+] setup_resnet_large')
        self.use_decompose = True
        self.share_alphas = True # sharing alphas may lose precision
        self.init_abstraction_method = 'backward'
        self.verify_candidate_batch = True
        self.verify_last_timeout = 20.0
        self.verify_splitting_strategy = 'input'
        self.verify_interm_timeout = 10.0
        self.use_sequential_abstract_forward = True
        
    def setup_resnet_extra_large(self, args):
        self.setup_resnet_large(args)

        
class AdvancedSettings(BaseSettings):
    
    def __init__(self, args=None):
        super().__init__()
        
        self.advanced_settings = [
            RestartSettings(args),
            MIPSettings(args),
            AbstractionSettings(args),
            DecompositionSettings(args)
        ]
        
        self._add_settings(args)
    
    def _add_settings(self, args=None):
        for setting_obj in self.advanced_settings:
            for key, value in setting_obj.__dict__.items():
                setattr(self, key, value)
    
    def __repr__(self):
        str = ''
        for setting_obj in self.advanced_settings:
            str += setting_obj.__repr__()
        return str