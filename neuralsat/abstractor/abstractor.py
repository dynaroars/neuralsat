import torch.nn as nn
import numpy as np
import traceback
import random
import torch
import copy

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import stop_criterion_batch_any
from auto_LiRPA import BoundedModule, BoundedTensor

from util.misc.result import AbstractResults
from util.misc.logger import logger
from .params import *


class NetworkAbstractor:
    
    "Over-approximation method: https://github.com/Verified-Intelligence/alpha-beta-CROWN"

    def __init__(self, pytorch_model, input_shape, method, input_split=False, device='cpu'):

        self.pytorch_model = copy.deepcopy(pytorch_model)
        self.device = device
        self.input_shape = input_shape
        
        # search domain
        self.input_split = input_split
        
        # computation algorithm
        self.method = method
        
        
    def setup(self, objective):
        assert self.select_params(objective), print('Initialization failed')
        
        logger.info(f'Initialized abstractor: mode="{self.mode}", method="{self.method}"')

        # check conversion correctness
        dummy = objective.lower_bounds[0].view(self.input_shape).to(self.device)
        try:
            assert torch.allclose(self.pytorch_model(dummy), self.net(dummy), atol=1e-5, rtol=1e-5)
        except AssertionError:
            print(f'torch allclose failed: norm {torch.norm(self.pytorch_model(dummy) - self.net(dummy))}')
            exit()
            
            
    def select_params(self, objective):
        params = [
            ['patches', self.method], # default
            ['matrix', self.method],
            
            ['patches', 'backward'],
            ['matrix', 'backward'],
            
            ['patches', 'forward'],
            ['matrix', 'forward'],
        ]
        
        for mode, method in params:
            logger.debug(f'Try {mode}, {method}')
            self._init_module(mode)
            if self._check_module(method, objective):
                self.mode = mode
                self.method = method
                return True
            
        return False
            
    def _init_module(self, mode):
        bound_opts = {'relu': 'adaptive', 'conv_mode': mode}
        
        # logger.debug(f'Trying bound_opts: {bound_opts}')
        self.net = BoundedModule(
            model=self.pytorch_model, 
            global_input=torch.randn(self.input_shape, device=self.device),
            bound_opts=bound_opts,
            device=self.device,
            verbose=False,
        )
        self.net.eval()
        
        
    def _check_module(self, method, objective):
        if np.prod(self.input_shape) >= 100000:
            return True
        
        # at least can run with batch=1
        x_L = objective.lower_bounds[0].view(self.input_shape)
        x_U = objective.upper_bounds[0].view(self.input_shape)
        x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(self.device)
        
        try:
            self.net.compute_bounds(x=(x,), method=method)
        except KeyboardInterrupt:
            exit()
        except:
            # traceback.print_exc()
            return False
        else:
            return True
        

    def initialize(self, objective, share_slopes=False, reference_bounds=None):
        objective.cs = objective.cs.to(self.device)
        objective.rhs = objective.rhs.to(self.device)
        
        # input property
        input_lowers = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
       
        # stop function used when optimizing abstraction
        stop_criterion_func = stop_criterion_batch_any(objective.rhs)
        
        self.x = BoundedTensor(input_lowers, PerturbationLpNorm(x_L=input_lowers, x_U=input_uppers)).to(self.device)
        
        # update initial reference bounds for later use
        self.init_reference_bounds = reference_bounds
        
        if self.method == 'crown-optimized':
            # setup optimization parameters
            self.net.set_bound_opts(get_initialize_opt_params(share_slopes, stop_criterion_func))

            # initial bounds
            lb, _, aux_reference_bounds = self.net.init_slope((self.x,), share_slopes=share_slopes, c=objective.cs)
            logger.info(f'Initial bounds: {lb.detach().cpu().flatten()}')
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})
            
            lb, _ = self.net.compute_bounds(
                x=(self.x,), 
                C=objective.cs, 
                method=self.method,
                aux_reference_bounds=aux_reference_bounds, 
                reference_bounds=reference_bounds,
            )
            logger.info(f'Initial optimized bounds: {lb.detach().cpu().flatten()}')
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
                
        elif self.method in ['forward', 'backward']:
            with torch.no_grad():
                lb, _ = self.net.compute_bounds(
                    x=(self.x,), 
                    C=objective.cs, 
                    method=self.method, 
                    reference_bounds=reference_bounds,
                )
            logger.info(f'Initial bounds: {lb.detach().cpu().flatten()}')
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
        

    def _naive_forward_hidden(self, domain_params, decisions):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        decision_np = np.array(decisions)
        batch = len(decision_np)
        assert batch > 0
        
        # update layer bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decision=decision_np
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
    

    def _forward_hidden(self, domain_params, decisions, use_beta=True):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        decision_np = np.array(decisions)
        batch = len(decision_np)
        assert batch > 0
        
        # update betas with new decisions
        num_splits = self.set_beta(
            model=self.net, 
            betas=domain_params.betas, 
            histories=domain_params.histories, 
            decision=decision_np,
            use_beta=use_beta,
        )
        
        # update hidden bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decision=decision_np
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
            self.set_slope(self.net, domain_params.slopes, set_all=True)

        # setup optimization parameters
        self.net.set_bound_opts(get_beta_opt_params(use_beta, stop_criterion_batch_any(double_rhs)))
        
        new_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            intermediate_layer_bounds=new_intermediate_layer_bounds,
            decision_thresh=double_rhs,
            # reference_bounds=new_intermediate_layer_bounds,
        )

        # reorganize output
        with torch.no_grad():
            double_lAs = self.get_batch_lAs(model=self.net, size=len(double_input_lowers), to_cpu=True)
            new_output_lbs = new_output_lbs.to(device='cpu')
            cpu_net = self.transfer_to_cpu(net=self.net, non_blocking=False)
            # slopes
            double_slopes = self.get_slope(cpu_net) if len(domain_params.slopes) > 0 else [[] for _ in range(batch * 2)]
            # betas
            double_betas = self.get_beta(cpu_net, num_splits) if use_beta else [[] for _ in range(batch * 2)]
            # hidden bounds
            double_lower_bounds, double_upper_bounds = self.get_batch_hidden_bounds(cpu_net, new_output_lbs)
            
        assert all([_.shape[0] == 2*batch for _ in double_lower_bounds]), print([_.shape for _ in double_lower_bounds])
        assert all([_.shape[0] == 2*batch for _ in double_upper_bounds]), print([_.shape for _ in double_upper_bounds])
            
        return AbstractResults(**{
            'input_lowers': double_input_lowers, 
            'input_uppers': double_input_uppers,
            'output_lbs': double_lower_bounds[-1], 
            'lAs': double_lAs, 
            'lower_bounds': double_lower_bounds, 
            'upper_bounds': double_upper_bounds, 
            'slopes': double_slopes, 
            'betas': double_betas, 
            'histories': domain_params.histories,
            'cs': double_cs,
            'rhs': double_rhs,
            'sat_solvers': domain_params.sat_solvers,
        })
        
        
    def _forward_input(self, domain_params, decisions):
        assert len(domain_params.cs) == len(domain_params.rhs) == len(domain_params.input_lowers) == len(domain_params.input_uppers)
        batch = len(decisions)
        assert batch > 0
        
        # splitting input by decisions (perform splitting)
        new_input_lowers, new_input_uppers = self.input_split_idx(
            input_lowers=domain_params.input_lowers, 
            input_uppers=domain_params.input_uppers, 
            split_idx=decisions,
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
        
        new_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs,
            reference_bounds=self.init_reference_bounds,
        )

        with torch.no_grad():
            # indexing on CPU
            cpu_net = self.transfer_to_cpu(
                net=self.net, 
                non_blocking=False, 
                slope_only=True,
            )
            # slopes
            double_slopes = self.get_slope(cpu_net) if len(domain_params.slopes) > 0 else [[] for _ in range(batch * 2)]

        return AbstractResults(**{
            'output_lbs': new_output_lbs, 
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers,
            'slopes': double_slopes, 
            'cs': double_cs, 
            'rhs': double_rhs, 
        })
        
        
    def forward(self, decisions, domain_params):
        if self.input_split:
            return self._forward_input(
                domain_params=domain_params,
                decisions=decisions,
            )
        return self._forward_hidden(
            domain_params=domain_params,
            decisions=decisions,
        )


    def __repr__(self):
        return f'{self.__class__.__name__}({self.mode}, {self.method})'
        
    from .utils import (
        get_slope, set_slope,
        get_beta, set_beta, reset_beta, 
        get_hidden_bounds, get_batch_hidden_bounds,
        get_lAs, get_batch_lAs, 
        transfer_to_cpu, 
        hidden_split_idx, input_split_idx,
        build_lp_solver, solve_full_assignment,
    )
