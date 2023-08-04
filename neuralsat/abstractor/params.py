import torch

def get_branching_opt_params():
    return {'optimize_bound_args': {
                'enable_beta_crown': False, 
                'fix_intermediate_layer_bounds': True,
                'pruning_in_iteration': True,
            }}   
    
    
def get_initialize_opt_params(share_slopes, stop_criterion_func):
    return {'optimize_bound_args': {
                'enable_beta_crown': False, 
                'enable_alpha_crown': True,
                'use_shared_alpha': share_slopes, 
                'early_stop': False,
                'init_alpha': False,
                'fix_intermediate_layer_bounds': True,
                'stop_criterion_func': stop_criterion_func,
                'iteration': 50, 
                'lr_alpha': 0.1, 
                'lr_decay': 0.98, 
            }}
    
    
def get_beta_opt_params(use_beta, stop_criterion_func):
    return {'optimize_bound_args': {
                'enable_beta_crown': use_beta, 
                'enable_alpha_crown': True,
                'fix_intermediate_layer_bounds': True, 
                'iteration': 50,
                'lr_alpha': 0.1, 
                'lr_decay': 0.98, 
                'lr_beta': 0.1,
                'pruning_in_iteration': False,
                'stop_criterion_func': stop_criterion_func,
                'multi_spec_keep_func': lambda x: torch.all(x, dim=-1),
            }}
    
    
def get_input_opt_params(stop_criterion_func):
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
