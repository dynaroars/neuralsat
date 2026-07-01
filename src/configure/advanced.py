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
        self.init_alpha_iteration = 100
        self.init_alpha_lr = 0.1
        self.clip_input_domain = False
        self.clip_input_domain_complete = False
        self.clip_input_domain_iters = 1
        self.loss_reduction_min = False
        self.use_maxpool_to_relu = False
        
        
WIDE_OUTPUT_THRESHOLD = 10000


def is_wide_output(output_shape: tuple | None = None, cs=None) -> bool:
    if cs is not None and hasattr(cs, 'shape') and len(cs.shape) > 0:
        return cs.shape[-1] >= WIDE_OUTPUT_THRESHOLD
    if output_shape is not None and len(output_shape) >= 2:
        return output_shape[-1] >= WIDE_OUTPUT_THRESHOLD
    return False


def configure_from_output_shape(settings, output_shape: tuple, batch: int) -> int:
    """Tune solver for wide output heads (many output constraints)."""
    if not is_wide_output(output_shape=output_shape):
        return batch
    settings.share_alphas = True
    settings.backward_batch_size = 128
    settings.use_attack = False
    settings.use_mip_tightening = False
    settings.use_restart = False
    settings.init_alpha_iteration = 50
    settings.init_alpha_lr = 0.3
    settings.verify_extra_opts = {
        'use_full_conv_alpha': False,
        'use_shared_alpha': True,
    }
    return min(batch, 128)


def model_has_maxpool(model) -> bool:
    import torch.nn as nn
    return any(isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)) for m in model.modules())


def configure_from_input_shape(settings, input_shape: tuple) -> None:
    """Tune bound propagation for very large input tensors (e.g. VGG-scale images)."""
    import numpy as np
    if np.prod(input_shape) < 100000:
        return
    settings.forward_dynamic = True
    settings.forward_max_dim = 100
    settings.backward_batch_size = 16


def configure_from_model(settings, model) -> None:
    """Tune bound propagation for model architecture (MaxPool, Softmax, etc.)."""
    if not any(type(m).__name__ in ('Softmax', 'LogSoftmax') for m in model.modules()):
        if not model_has_maxpool(model):
            return
    if model_has_maxpool(model):
        settings.forward_max_dim = 100
        settings.backward_batch_size = 64
    if not any(type(m).__name__ in ('Softmax', 'LogSoftmax') for m in model.modules()):
        return
    settings.init_abstraction_method = 'crown-optimized'
    settings.init_alpha_iteration = 50
    settings.init_alpha_lr = 0.5
    settings.loss_reduction_min = True
    opts = dict(getattr(settings, 'verify_extra_opts', None) or {})
    opts.update({
        'conv_mode': 'matrix',
        'softmax': 'complex',
        'disable_optimization': ['Exp'],
        'forward_before_compute_bounds': True,
        'fixed_reducemax_index': True,
        'sparse_intermediate_bounds': False,
        'sparse_conv_intermediate_bounds': False,
    })
    settings.verify_extra_opts = opts


def configure_for_input_split(settings) -> None:
    settings.clip_input_domain = True
    settings.init_abstraction_method = 'backward'
    settings.use_mip_tightening = False
    settings.input_split_decision_method = 'smart'
    settings.verify_extra_opts = {}


def configure_for_vggnet(settings, perturbed: int) -> None:
    settings.use_maxpool_to_relu = True
    settings.loss_reduction_min = True
    settings.forward_dynamic = True
    settings.forward_max_dim = 100
    settings.backward_batch_size = 64
    settings.use_mip_tightening = False
    settings.skip_initial_worst_bound = float('-inf')
    settings.verify_extra_opts = {}
    if perturbed > 100:
        settings.default_abstraction_method = 'crown-optimized'
        settings.init_abstraction_method = 'crown-optimized'
    else:
        settings.default_abstraction_method = 'backward'
        settings.init_abstraction_method = 'backward'


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