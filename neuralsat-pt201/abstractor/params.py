from beartype import beartype
import typing
import torch


@beartype
def get_branching_opt_params() -> dict:
    return {'optimize_bound_args': {
                'enable_beta_crown': False, 
                'fix_intermediate_layer_bounds': True,
                'pruning_in_iteration': True,
            }}   
    
    
@beartype
def get_initialize_opt_params(share_slopes: bool, stop_criterion_func: typing.Callable) -> dict:
    return {'optimize_bound_args': {
                'enable_alpha_crown': True,
                'enable_beta_crown': False, 
                'use_shared_alpha': share_slopes, 
                'early_stop': False,
                'init_alpha': False,
                'fix_intermediate_layer_bounds': True,
                'stop_criterion_func': stop_criterion_func,
                'iteration': 50, 
                'lr_alpha': 0.1, 
                'lr_beta': 0.1,
                'lr_decay': 0.98, 
            }}
    
    
@beartype
def get_beta_opt_params(stop_criterion_func: typing.Callable) -> dict:
    return {'optimize_bound_args': {
                'enable_alpha_crown': True,
                'enable_beta_crown': True, 
                'fix_intermediate_layer_bounds': True, 
                'iteration': 20,
                'lr_alpha': 0.1, 
                'lr_beta': 0.1,
                'lr_decay': 0.98, 
                'pruning_in_iteration': True,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': lambda x: torch.all(x, dim=-1),
            }}
    
    
@beartype
def get_input_opt_params(stop_criterion_func: typing.Callable) -> dict:
    return {'optimize_bound_args': {
                'enable_beta_crown': False, 
                'fix_intermediate_layer_bounds': True, 
                'iteration': 20,
                'lr_alpha': 0.1, 
                'lr_decay': 0.98, 
                'pruning_in_iteration': True,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': lambda x: torch.all(x, dim=-1),
            }}
