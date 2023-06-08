import torch.nn as nn
import numpy as np
import random
import torch
import copy

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any
from auto_LiRPA import BoundedModule, BoundedTensor

from util.misc.result import AbstractResults
from .params import *


class NetworkAbstractor:
    
    "Over-approximation method"

    def __init__(self, pytorch_model, input_shape, method, input_split=False, device='cpu'):

        self.pytorch_model = copy.deepcopy(pytorch_model)
        self.device = device
        self.input_shape = input_shape
        
        # search domain
        self.input_split = input_split
        
        # computation algorithm
        self.mode = 'patches'
        assert method in ['backward', 'crown-optimized']
        self.method = method
        
        # try matrix mode
        self._init_module(self.mode)
        self._check()
        
        print(f'[!] Using mode="{self.mode}", method="{self.method}"')

        # check conversion correctness
        dummy = torch.randn(input_shape, device=self.device)
        try:
            assert torch.allclose(pytorch_model(dummy), self.net(dummy), atol=1e-4, rtol=0)
        except AssertionError:
            print(f'torch allclose failed: norm {torch.norm(pytorch_model(dummy) - self.net(dummy))}')
            exit()
            
            
    def _init_module(self, mode):
        self.net = BoundedModule(
            model=self.pytorch_model, 
            global_input=torch.zeros(self.input_shape, device=self.device),
            bound_opts={'relu': 'adaptive', 'conv_mode': mode},
            device=self.device
        )
        self.net.eval()
        
        
    def _check(self):
        ptb = PerturbationLpNorm(x_L=torch.zeros(self.input_shape), x_U=torch.ones(self.input_shape))
        x = BoundedTensor(ptb.x_L, ptb).to(self.device)
        
        try:
            self.net.compute_bounds(x=(x,), method=self.method)
        except IndexError:
            self.mode = 'matrix'
            # self.method = 'backward'
            self._init_module(self.mode)            
        except:
            raise NotImplementedError()
        

    def initialize(self, objective, share_slopes=False):
        objective.cs = objective.cs.to(self.device)
        objective.rhs = objective.rhs.to(self.device)
        
        # input property
        input_lowers = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        
        # stop function used when optimizing abstraction
        stop_criterion_func = stop_criterion_batch_any(objective.rhs)
        
        self.x = BoundedTensor(input_lowers, PerturbationLpNorm(x_L=input_lowers, x_U=input_uppers)).to(self.device)
        
        if self.method == 'crown-optimized':
            # setup optimization parameters
            self.net.set_bound_opts(get_initialize_opt_params(share_slopes, stop_criterion_func))

            # initial bounds
            lb, _, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=objective.cs)
            print('Initial bounds:', lb.detach().cpu().flatten())
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})

            lb, _ = self.net.compute_bounds(
                x=(self.x,), 
                C=objective.cs, 
                method=self.method,
                aux_reference_bounds=aux_reference_bounds
            )
            print('Initial optimized bounds:', lb.detach().cpu().flatten())
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})
            
            # reorganize tensors
            lower_bounds, upper_bounds = self.get_hidden_bounds(self.net, lb)

            return AbstractResults(**{
                'output_lbs': lower_bounds[-1], 
                'lAs': self.get_lAs(self.net), 
                'lower_bounds': lower_bounds, 
                'upper_bounds': upper_bounds, 
                'slopes': self.get_slope(self.net), 
                'histories': [[[], []] for _ in range(len(self.net.relus))], 
                'cs': objective.cs,
                'rhs': objective.rhs,
                'input_lowers': input_lowers,
                'input_uppers': input_uppers,
            })
                
        elif self.method == 'backward':
            with torch.no_grad():
                lb, _ = self.net.compute_bounds(
                    x=(self.x,), 
                    C=objective.cs, 
                    method=self.method, 
                )
            print('Initial bounds:', lb.detach().cpu().flatten())
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})
            
            return AbstractResults(**{
                'output_lbs': lb, 
                'slopes': self.get_slope(self.net), 
                'cs': objective.cs,
                'rhs': objective.rhs,
                'input_lowers': input_lowers,
                'input_uppers': input_uppers,
            })
            
        else:
            print(self.method)
            raise NotImplementedError()
        

    def _naive_forward_hidden(self, domain_params, branching_decisions):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        decision = np.array(branching_decisions)
        batch = len(decision)
        assert batch > 0
        
        # update layer bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decision=decision
        )
        
        # sample-wise for supporting handling multiple targets in one batch
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers], dim=0)
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers], dim=0)
        assert torch.all(double_input_lowers <= double_input_uppers)
        
        # create new inputs
        new_x = BoundedTensor(double_input_lowers, PerturbationLpNorm(x_L=double_input_lowers, x_U=double_input_uppers))
        
        # set slope here again
        if len(domain_params.slopes) > 0: 
            self.set_slope(self.net, domain_params.slopes)

        # setup optimization parameters
        self.net.set_bound_opts(get_branching_opt_params())

        with torch.no_grad():
            lb, _, = self.net.compute_bounds(
                x=(new_x,), 
                C=double_cs, 
                method='backward', 
                reuse_alpha=True,
                intermediate_layer_bounds=new_intermediate_layer_bounds
            )
        return AbstractResults(**{'output_lbs': lb})
    

    def _forward_hidden(self, domain_params, branching_decisions, use_beta=True):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        decision = np.array(branching_decisions)
        batch = len(decision)
        assert batch > 0
        
        # update betas with new decisions
        splits_per_example = self.set_beta(
            model=self.net, 
            betas=domain_params.betas, 
            histories=domain_params.histories, 
            decision=decision,
            use_beta=use_beta,
        )
        
        # update hidden bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decision=decision
        )
        
        # 2 * batch
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers], dim=0)
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers], dim=0)
        assert torch.all(double_input_lowers <= double_input_uppers)
        
        # create new inputs
        new_x = BoundedTensor(double_input_lowers, PerturbationLpNorm(x_L=double_input_lowers, x_U=double_input_uppers))
        
        # update slopes
        if len(domain_params.slopes) > 0: 
            self.set_slope(self.net, domain_params.slopes)

        # setup optimization parameters
        self.net.set_bound_opts(get_beta_opt_params(use_beta, stop_criterion_batch_any(double_rhs)))
        
        lb, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            intermediate_layer_bounds=new_intermediate_layer_bounds,
            decision_thresh=double_rhs
        )

        # process output on CPU instead of GPU
        with torch.no_grad():
            lAs = self.get_batch_lAs(model=self.net, size=len(double_input_lowers), to_cpu=True)
            lb = lb.to(device='cpu')
            cpu_net = self.transfer_to_cpu(net=self.net, non_blocking=False)
            # slopes
            ret_s = self.get_slope(cpu_net) if len(domain_params.slopes) > 0 else [[] for _ in range(batch * 2)]
            # betas
            ret_b = self.get_beta(cpu_net, splits_per_example) if use_beta else [[] for _ in range(batch * 2)]
            # hidden bounds
            ret_l, ret_u = self.get_batch_hidden_bounds(cpu_net, lb)
            
        assert all([_.shape[0] == 2*batch for _ in ret_l]), print([_.shape for _ in ret_l])
        assert all([_.shape[0] == 2*batch for _ in ret_u]), print([_.shape for _ in ret_u])
            
        return AbstractResults(**{
            'input_lowers': double_input_lowers, 
            'input_uppers': double_input_uppers,
            'output_lbs': ret_l[-1], 
            'lAs': lAs, 
            'lower_bounds': ret_l, 
            'upper_bounds': ret_u, 
            'slopes': ret_s, 
            'betas': ret_b, 
            'histories': domain_params.histories,
            'cs': double_cs,
            'rhs': double_rhs,
            'sat_solvers': domain_params.sat_solvers,
        })
        
        
    def _forward_input(self, domain_params, branching_decisions):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        batch = len(branching_decisions)
        assert batch > 0
        
        # splitting input by branching_decisions (perform splitting)
        new_input_lowers, new_input_uppers = self.input_split_idx(
            input_lowers=domain_params.input_lowers, 
            input_uppers=domain_params.input_uppers, 
            split_idx=branching_decisions,
        )
        assert torch.all(new_input_lowers <= new_input_uppers)
        
        # create new inputs
        new_x = BoundedTensor(new_input_lowers, PerturbationLpNorm(x_L=new_input_lowers, x_U=new_input_uppers))
        
        # sample-wise for supporting handling multiple targets in one batch
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        
        # set slope again since batch might change
        if len(domain_params.slopes) > 0: 
            self.set_slope(self.net, domain_params.slopes, set_all=True)
        
        # set optimization parameters
        self.net.set_bound_opts(get_input_opt_params(stop_criterion_batch_any(double_rhs)))
        
        lb, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs
        )

        with torch.no_grad():
            # indexing on CPU
            cpu_net = self.transfer_to_cpu(
                net=self.net, 
                non_blocking=False, 
                slope_only=True,
            )
            # slopes
            ret_s = self.get_slope(cpu_net) if len(domain_params.slopes) > 0 else [[] for _ in range(batch * 2)]

        return AbstractResults(**{
            'output_lbs': lb, 
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers,
            'slopes': ret_s, 
            'cs': double_cs, 
            'rhs': double_rhs, 
        })
        
        
    def forward(self, branching_decisions, domain_params):
        if self.input_split:
            return self._forward_input(
                domain_params=domain_params,
                branching_decisions=branching_decisions,
            )
        return self._forward_hidden(
            domain_params=domain_params,
            branching_decisions=branching_decisions,
        )


    from .utils import (
        get_slope, set_slope,
        get_beta, set_beta, reset_beta, 
        get_hidden_bounds, get_batch_hidden_bounds,
        get_lAs, get_batch_lAs, 
        transfer_to_cpu, 
        hidden_split_idx, input_split_idx,
        build_lp_solver, solve_full_assignment,
    )
