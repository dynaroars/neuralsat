from beartype import beartype
import typing
import torch

from setting import Settings

@beartype
def get_check_abstractor_params() -> dict:
    return {
        'crown_batch_size': Settings.backward_batch_size,
        'forward_max_dim': Settings.forward_max_dim,
        'dynamic_forward': Settings.forward_dynamic,
        'optimize_bound_args': {
            'iteration': 1,
            'stop_criterion_func': lambda x: False,
            'enable_beta_crown': False
        }
    }

@beartype
def get_branching_opt_params() -> dict:
    return {
        'crown_batch_size': Settings.backward_batch_size,
        'optimize_bound_args': {
            'enable_beta_crown': False, 
            'fix_interm_bounds': True,
        }
    }   
    
    
@beartype
def get_initialize_opt_params(stop_criterion_func: typing.Callable) -> dict:
    return {
        'crown_batch_size': Settings.backward_batch_size,
        'optimize_bound_args': {
            'enable_alpha_crown': True,
            'enable_beta_crown': False, 
            'use_shared_alpha': Settings.share_alphas, 
            'init_alpha': False,
            'fix_interm_bounds': True,
            'stop_criterion_func': stop_criterion_func,
            'iteration': 50, 
            'lr_alpha': 0.1, 
            'lr_beta': 0.1,
            'lr_decay': 0.98, 
        }
    }
    
    
@beartype
def get_beta_opt_params(stop_criterion_func: typing.Callable) -> dict:
    return {
        'crown_batch_size': Settings.backward_batch_size,
        'optimize_bound_args': {
            'enable_alpha_crown': True,
            'enable_beta_crown': True, 
            'use_shared_alpha': Settings.share_alphas, 
            'fix_interm_bounds': True, 
            'iteration': 20,
            'lr_alpha': 0.1, 
            'lr_beta': 0.1,
            'lr_decay': 0.98, 
            'stop_criterion_func': stop_criterion_func,
        }
    }

    
@beartype
def get_input_opt_params(stop_criterion_func: typing.Callable) -> dict:
    return {
        'crown_batch_size': Settings.backward_batch_size,
        'forward_max_dim': Settings.forward_max_dim,
        'dynamic_forward': Settings.forward_dynamic,
        'optimize_bound_args': {
            'enable_beta_crown': False, 
            'fix_interm_bounds': True, 
            'iteration': 20,
            'lr_alpha': 0.1, 
            'lr_decay': 0.98, 
            'stop_criterion_func': stop_criterion_func,
        }
    }
