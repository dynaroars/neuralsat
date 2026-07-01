import warnings
warnings.filterwarnings(action='ignore')

from beartype import beartype
import traceback
import logging
import typing
import torch
import tqdm
import copy
import math
import time
import os

from .auto_LiRPA.utils import stop_criterion_batch_any
from .auto_LiRPA import BoundedModule
from .graph_optimizer import merge_relu_lookup_table, merge_sign

from helper.misc.result import AbstractResults, CoefficientMatrix
from helper.network.onnx2pytorch import ConvertModel
from helper.misc.torch_cuda_memory import gc_cuda
from helper.misc.logger import logger

from .params import *
from configure.advanced import is_wide_output
from abstractor.input_split_clip import clip_domains
from abstractor.auto_LiRPA.concretize_func import construct_constraints

class NetworkAbstractor:

    @beartype
    def __init__(self: 'NetworkAbstractor', pytorch_model: ConvertModel | torch.nn.Module, 
                 input_shape: tuple, method: str, input_split: bool = False, device: str = 'cpu') -> None:

        self.pytorch_model = copy.deepcopy(pytorch_model)
        self.device = device
        self.input_shape = input_shape
        
        # search domain
        self.input_split = input_split
        
        # computation algorithm
        self.method = method
        
        # debug
        self.iteration = 0

    @beartype
    def _conv_modes(self: 'NetworkAbstractor') -> list[str]:
        has_conv = any(
            isinstance(m, (
                torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
            ))
            for m in self.pytorch_model.modules()
        )
        return ['patches', 'matrix'] if has_conv else ['matrix', 'patches']
        
    @beartype
    @property
    def split_points(self):
        if not hasattr(self, '_split_points'):
            self._split_points = [self.net.split_activations[k.name][0][0].get_split_point() for k in self.net.split_nodes]
        return self._split_points
        
    @beartype
    def setup(self: 'NetworkAbstractor', objective: typing.Any, extra_opts: dict = {}, preprocess: bool = False) -> None:
        if objective is not None and is_wide_output(cs=getattr(objective, 'cs', None)):
            wide_opts = copy.deepcopy(extra_opts)
            wide_opts.update(getattr(Settings, 'verify_extra_opts', {}) or {})
            Settings.use_restart = False
            Settings.use_attack = False
            if self.select_params(objective, extra_opts=wide_opts):
                return None

        if self.select_params(objective, extra_opts=extra_opts):
            return None
        
        if preprocess:
            raise NotImplementedError('Preprocess initialization failed')
        
        # special settings 
        Settings.use_restart = False
        Settings.use_attack = False
        
        # FIXME: try special settings for large CNNs
        new_extra_opts = copy.deepcopy(extra_opts)
        new_extra_opts.update({'use_full_conv_alpha': False})
        if self.select_params(objective, extra_opts=new_extra_opts):
            return None
        
        # FIXME: try special settings for ViT
        Settings.backward_batch_size = float('inf')
        new_extra_opts = copy.deepcopy(extra_opts)
        new_extra_opts.update({'sparse_intermediate_bounds': False})
        if self.select_params(objective, extra_opts=new_extra_opts):
            return None
        
        # FIXME: try smaller backward batch size
        new_extra_opts = copy.deepcopy(extra_opts)
        # new_extra_opts.update({'use_full_conv_alpha': False, 'use_shared_alpha': True})
        for backward_batch_size in [512, 128, 32]:
            Settings.backward_batch_size = backward_batch_size
            if self.select_params(objective, extra_opts=new_extra_opts):
                return None 
            
        # FIXME: try special settings for large-output networks
        Settings.backward_batch_size = 128
        Settings.share_alphas = True
        new_extra_opts = copy.deepcopy(extra_opts)
        new_extra_opts.update({'use_full_conv_alpha': False, 'use_shared_alpha': True})
        if self.select_params(objective, extra_opts=new_extra_opts):
            return None

        # Last resort: fall back to plain backward (no alpha optimization)
        saved_method = self.method
        self.method = 'backward'
        Settings.backward_batch_size = float('inf')
        Settings.share_alphas = False
        if self.select_params(objective, extra_opts={}):
            logger.info('[setup] Fell back to plain backward (no alpha optimization)')
            return None
        self.method = saved_method

        logger.info('[setup] Initialization failed')
        raise NotImplementedError('Initialization failed')

    @beartype
    def _get_clip_lA_lbias(self: 'NetworkAbstractor') -> tuple[torch.Tensor | None, torch.Tensor | None]:
        input_name = self.net.input_name[0]
        node = self.net[input_name]
        lA = getattr(node, 'lA', None)
        if lA is None:
            return None, None
        lA = lA.transpose(0, 1).flatten(2)
        lbias = torch.zeros(lA.shape[0], lA.shape[1], device=lA.device, dtype=lA.dtype)
        return lA, lbias

    @beartype
    def _clip_input_domain(self: 'NetworkAbstractor', input_lowers: torch.Tensor, input_uppers: torch.Tensor,
                           rhs: torch.Tensor, output_lbs: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        if not Settings.clip_input_domain:
            return input_lowers, input_uppers, None
        lA, lbias = self._get_clip_lA_lbias()
        if lA is None:
            return input_lowers, input_uppers, None
        new_lowers, new_uppers = clip_domains(
            input_lowers, input_uppers, rhs, lA, lbias,
            dm_lb=output_lbs, num_iters=Settings.clip_input_domain_iters,
        )
        constraints = None
        if Settings.clip_input_domain_complete:
            batch_size = len(new_lowers)
            x_dim = new_lowers.flatten(1).shape[1]
            constraints = construct_constraints(lA, lbias, rhs, batch_size, x_dim)
        return new_lowers, new_uppers, constraints
            
    @beartype
    def select_params(self: 'NetworkAbstractor', objective: typing.Any, extra_opts: dict = {}) -> bool:
        params = [[mode, self.method] for mode in self._conv_modes()]
        if self.input_split and (self.method != 'backward'):
            params += [[mode, 'backward'] for mode in self._conv_modes()]
        
        for mode, method in params:
            gc_cuda()
            logger.debug(f'[select_params] Try {self.input_split=} {Settings.backward_batch_size=} {extra_opts=} {mode=} {method=}')
            self._init_module(mode=mode, objective=objective, extra_opts=extra_opts)
            if self._check_module(method=method, objective=objective):
                self.mode = mode
                self.method = method
                logger.info(f'[select_params] Success: {self.input_split=} {Settings.backward_batch_size=} {extra_opts=} {mode=} {method=}')
                return True
            else:
                logger.info(f'[_check_module] Failed: {self.input_split=} {Settings.backward_batch_size=} {extra_opts=} {mode=} {method=}')
        return False
            
    @beartype
    def _init_module(self: 'NetworkAbstractor', mode: str, objective: typing.Any, extra_opts: dict = {}) -> None:
        bound_opts = {'conv_mode': mode, 'verbosity': 0, **extra_opts}
        logger.debug(f'[_init_module] Try {bound_opts=}')
        self.net = BoundedModule(
            model=self.pytorch_model, 
            global_input=torch.zeros(self.input_shape, device=self.device),
            bound_opts=bound_opts,
            device=self.device,
            verbose=False,
        )
        merged = merge_relu_lookup_table(self.net)
        merge_sign(self.net)
        self.net.eval()
        self.net.get_split_nodes()
        
        # check conversion correctness
        if objective:
            dummy = objective.lower_bounds[0].clone().view(self.input_shape).to(self.device)
        else:
            logger.debug(f'[_init_module] Use random dummy input for checking correctness')
            dummy = torch.randn(self.input_shape, device=self.device) 
            
        if (os.environ.get('NEURALSAT_DEBUG')
                and merged == 0
                and not getattr(self.pytorch_model, '_lookup_onnx_load', False)
                and not getattr(self.pytorch_model, '_gtrsb_nhwc', None)):
            self.net.to('cpu')
            self.pytorch_model.to('cpu')
            dummy = dummy.to('cpu')
            assert torch.allclose(self.pytorch_model(dummy), self.net(dummy), atol=1e-4, rtol=1e-4)
            self.net.to(self.device)
            self.pytorch_model.to(self.device)
        
        
    @beartype
    def _check_module(self: 'NetworkAbstractor', method: str, objective: typing.Any) -> bool:
        if objective is not None and is_wide_output(cs=getattr(objective, 'cs', None)):
            return True

        # at least can run with batch=1
        if objective:
            x_L = objective.lower_bounds[0].view(self.input_shape).to(self.device)
            x_U = objective.upper_bounds[0].view(self.input_shape).to(self.device)
        else:
            logger.debug(f'[_check_module] Use random dummy input for checking correctness')
            x_L = torch.randn(self.input_shape, device=self.device) 
            x_U = x_L + 0.1
        
        x = self.new_input(x_L=x_L, x_U=x_U)
        
        # forward to save architectural information
        self.net(x) 
    
        if math.prod(self.input_shape) >= 100000:
            return True
        
        if Settings.use_decompose or Settings.skip_preprocess:
            return True
        
        try:
            self.net.set_bound_opts(get_check_abstractor_params())
            # stop_criterion_func = stop_criterion_batch_any(objective.rhs.to(self.device))
            # self.net.set_bound_opts(get_initialize_opt_params(stop_criterion_func))
            # initial bounds
            if method == 'crown-optimized':
                _, _, aux_reference_bounds = self.net.init_alpha(
                    x=(x,), 
                    share_alphas=Settings.share_alphas, 
                    c=objective.cs.to(self.device), 
                    bound_upper=False,
                ) 
                lb, _ = self.net.compute_bounds(
                    x=(x,), 
                    C=objective.cs.to(self.device), 
                    method='crown-optimized',
                    aux_reference_bounds=aux_reference_bounds, 
                    bound_upper=False,
                )
            # elif method == 'backward':
            #     lb, _, _ = self.net.init_alpha(
            #         x=(x,), 
            #         share_alphas=Settings.share_alphas, 
            #         c=objective.cs.to(self.device) if objective is not None else None, 
            #         bound_upper=False,
            #     ) 
            else:
                lb, _ = self.net.compute_bounds(x=(x,), method=method, bound_upper=False) 
            assert not torch.isnan(lb).any()
        except RuntimeError:
            if os.environ.get('NEURALSAT_DEBUG'):
                raise
            return False # FIXME: might affect other benchmarks
        except KeyboardInterrupt:
            exit()
        except SystemExit:
            exit()
        except:
            if os.environ.get('NEURALSAT_DEBUG'):
                raise
            # raise
            if logger.level <= logging.DEBUG:
                traceback.print_exc()
            return False
        else:
            return True
        

    @beartype
    def initialize(self: 'NetworkAbstractor', objective: typing.Any, reference_bounds: dict | None = None, short_cut: bool = False) -> AbstractResults:
        objective.cs = objective.cs.to(self.device)
        objective.rhs = objective.rhs.to(self.device)
        
        # input property
        input_lowers = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
       
        # stop function used when optimizing abstraction
        stop_criterion_func = stop_criterion_batch_any(objective.rhs)

        # setup intialization parameters
        self.net.set_bound_opts(get_initialize_opt_params(stop_criterion_func))
        
        # create input
        x = self.new_input(x_L=input_lowers, x_U=input_uppers)
        
        # update initial reference bounds for later use
        self.init_reference_bounds = reference_bounds
        
        # get split nodes
        self.net.get_split_nodes()
        
        if self.method not in ['crown-optimized']:
            with torch.no_grad():
                lb, _ = self.net.compute_bounds(
                    x=(x,), 
                    C=objective.cs, 
                    method=self.method, 
                    reference_bounds=reference_bounds,
                    bound_upper=False,
                )
            logger.info(f'Initial bounds (first 10): {lb.detach().cpu().flatten()[:10]}')
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})

            input_lowers, input_uppers, _ = self._clip_input_domain(
                input_lowers, input_uppers, objective.rhs, lb)

            if short_cut:
                return AbstractResults(**{
                    'objective_ids': getattr(objective, 'ids', None),
                    'output_lbs': lb, 
                    'lAs': self.get_lAs(), 
                    'slopes': self.get_slope(), 
                    'cs': objective.cs,
                    'rhs': objective.rhs,
                    'input_lowers': input_lowers,
                    'input_uppers': input_uppers,
                })
                
            # reorganize tensors
            with torch.no_grad():
                lower_bounds, upper_bounds = self.get_hidden_bounds(lb)
            self.net.get_split_nodes()
                
            return AbstractResults(**{
                'objective_ids': getattr(objective, 'ids', None),
                'output_lbs': lb, 
                'lAs': self.get_lAs(), 
                'lower_bounds': lower_bounds, 
                'upper_bounds': upper_bounds, 
                'slopes': self.get_slope(), 
                'histories': {_.name: ([], [], []) for _ in self.net.split_nodes}, 
                'cs': objective.cs,
                'rhs': objective.rhs,
                'input_lowers': input_lowers,
                'input_uppers': input_uppers,
            })

        # initial bounds
        lb_init, _, aux_reference_bounds = self.net.init_alpha(
            x=(x,), 
            share_alphas=Settings.share_alphas, 
            c=objective.cs, 
            bound_upper=False,
        )
        
        if os.environ.get('NEURALSAT_DEBUG'):
            print(f'[Init alpha] {x.shape=} {Settings.share_alphas=} {objective.cs.shape=} {lb_init.flatten()=}',)
            
        logger.info(f'Initial bounds (first 10): {lb_init.detach().cpu().flatten()[:10]}')
        
        if stop_criterion_func(lb_init).all().item():
            return AbstractResults(**{'output_lbs': lb_init})

        # self.update_refined_beta(init_betas, batch=len(objective.cs))
        lb, _ = self.net.compute_bounds(
            x=(x,), 
            C=objective.cs, 
            method='crown-optimized',
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
            bound_upper=False,
        )
        logger.info(f'Initial optimized bounds (first 10): {lb.detach().cpu().flatten()[:10]}')
        if stop_criterion_func(lb).all().item():
            return AbstractResults(**{'output_lbs': lb})

        input_lowers, input_uppers, _ = self._clip_input_domain(
            input_lowers, input_uppers, objective.rhs, lb)
        
        # reorganize tensors
        with torch.no_grad():
            lower_bounds, upper_bounds = self.get_hidden_bounds(lb)
        self.net.get_split_nodes()

        return AbstractResults(**{
            'objective_ids': objective.ids,
            'output_lbs': lower_bounds[self.net.final_name], 
            'lAs': self.get_lAs(), 
            'lower_bounds': lower_bounds, 
            'upper_bounds': upper_bounds, 
            'slopes': self.get_slope(), 
            'histories': {_.name: ([], [], []) for _ in self.net.split_nodes}, 
            'cs': objective.cs,
            'rhs': objective.rhs,
            'input_lowers': input_lowers,
            'input_uppers': input_uppers,
        })
            
    
    @beartype
    def _forward_hidden(self: 'NetworkAbstractor', domain_params: AbstractResults, decisions: list, simplify: bool) -> AbstractResults:
        assert len(decisions) == len(domain_params.cs) == len(domain_params.rhs) == \
               len(domain_params.input_lowers) == len(domain_params.input_uppers), \
               print(f'len(decisions)={len(decisions)}, len(domain_params.input_lowers)={len(domain_params.input_lowers)}')
            
        batch = len(decisions)
        assert batch > 0
        
        # 2 * batch
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers], dim=0) # TODO: torch compile
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers], dim=0) # TODO: torch compile
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(double_input_lowers <= double_input_uppers)
        
        # update hidden bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decisions=decisions
        )
        
        # create new inputs
        new_x = self.new_input(x_L=double_input_lowers, x_U=double_input_uppers)
        
        # update slopes
        if len(domain_params.slopes) > 0: 
            self.set_slope(domain_params.slopes)
            
        # simplify for decision heuristics
        if simplify:
            # setup optimization parameters
            self.net.set_bound_opts(get_branching_opt_params())
            
            # compute outputs
            with torch.no_grad():
                double_output_lbs, _, = self.net.compute_bounds(
                    x=(new_x,), 
                    C=double_cs, 
                    method='backward', 
                    reuse_alpha=self.method == 'crown-optimized',
                    interm_bounds=new_intermediate_layer_bounds,
                    bound_upper=False,
                )
            return AbstractResults(**{'output_lbs': double_output_lbs})

        # 2 * batch
        assert len(decisions) == len(domain_params.objective_ids)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        double_objective_ids = torch.cat([domain_params.objective_ids, domain_params.objective_ids], dim=0)
        double_sat_solvers = domain_params.sat_solvers * 2 if domain_params.sat_solvers is not None else None
         
        # update new decisions
        double_histories = self.update_histories(histories=domain_params.histories, decisions=decisions)
        double_betas = domain_params.betas * 2
        num_splits = self.set_beta(betas=double_betas, histories=double_histories)
        
        # setup optimization parameters
        self.net.set_bound_opts(get_beta_opt_params(stop_criterion_batch_any(double_rhs)))
        
        # compute outputs
        double_ref_output_lbs = torch.cat([domain_params.output_lbs, domain_params.output_lbs], dim=0) # TODO: torch compile
        reference_bounds = {self.net.final_name: [double_ref_output_lbs, double_ref_output_lbs + torch.inf]}
        double_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs,
            interm_bounds=new_intermediate_layer_bounds,
            reference_bounds=reference_bounds,
            bound_upper=False,
        )

        # reorganize output
        with torch.no_grad():
            # lAs
            double_lAs = self.get_lAs()
            # outputs
            double_output_lbs = double_output_lbs.detach().to(device='cpu')
            # slopes
            double_slopes = self.get_slope() if len(domain_params.slopes) > 0 else {}
            # betas
            double_betas = self.get_beta(num_splits)
            # hidden bounds
            double_lower_bounds, double_upper_bounds = self.get_hidden_bounds(double_output_lbs)
            
        assert all([_.shape[0] == 2 * batch for _ in double_lower_bounds.values()]), print([_.shape for _ in double_lower_bounds.values()])
        assert all([_.shape[0] == 2 * batch for _ in double_upper_bounds.values()]), print([_.shape for _ in double_upper_bounds.values()])
        assert all([_.shape[0] == 2 * batch for _ in double_lAs.values()]), print([_.shape for _ in double_lAs.values()])
        assert len(double_histories) == len(double_betas) == 2 * batch
            
        return AbstractResults(**{
            'objective_ids': double_objective_ids,
            'output_lbs': double_lower_bounds[self.net.final_name], 
            'input_lowers': double_input_lowers, 
            'input_uppers': double_input_uppers,
            'lAs': double_lAs, 
            'lower_bounds': double_lower_bounds, 
            'upper_bounds': double_upper_bounds, 
            'slopes': double_slopes, 
            'betas': double_betas, 
            'histories': double_histories,
            'cs': double_cs,
            'rhs': double_rhs,
            'sat_solvers': double_sat_solvers,
        })
        
        
    @beartype
    def _forward_input(self: 'NetworkAbstractor', domain_params: AbstractResults, decisions: torch.Tensor, simplify: bool) -> AbstractResults:
        assert len(decisions) == len(domain_params.cs) == len(domain_params.rhs) == \
               len(domain_params.input_lowers) == len(domain_params.input_uppers) == len(domain_params.objective_ids)
               
        batch = len(decisions)
        assert batch > 0
        
        # splitting input by decisions (perform splitting)
        new_input_lowers, new_input_uppers = self.input_split_idx(
            input_lowers=domain_params.input_lowers, 
            input_uppers=domain_params.input_uppers, 
            split_idx=decisions,
        )
        
        # create new inputs
        constraints = getattr(domain_params, 'constraints', None)
        no_return_inf = Settings.clip_input_domain and Settings.clip_input_domain_complete and constraints is not None
        if no_return_inf:
            self.net.init_infeasible_bounds_constraints(len(new_input_lowers), self.device)
        new_x = self.new_input(
            x_L=new_input_lowers, x_U=new_input_uppers,
            constraints=constraints, no_return_inf=no_return_inf,
        )
        
        # 2 * batch
        double_objective_ids = torch.cat([domain_params.objective_ids, domain_params.objective_ids], dim=0)
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        
        # set slope again since batch might change
        if len(domain_params.slopes) > 0: 
            self.set_slope(domain_params.slopes)
        
        # set optimization parameters
        self.net.set_bound_opts(get_input_opt_params(stop_criterion_batch_any(double_rhs)))
        
        double_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs,
            reference_bounds=self.init_reference_bounds,
            bound_upper=False,
        )
        if self.net.infeasible_bounds_constraints is not None:
            infeasible = self.net.infeasible_bounds_constraints
            double_output_lbs[infeasible] = torch.inf

        new_input_lowers, new_input_uppers, constraints = self._clip_input_domain(
            new_input_lowers, new_input_uppers, double_rhs, double_output_lbs)

        with torch.no_grad():
            # slopes
            double_slopes = self.get_slope() if len(domain_params.slopes) > 0 else {}
            double_lAs = self.get_lAs()
            
            # hidden bounds
            double_lower_bounds, double_upper_bounds = None, None
            if Settings.use_save_reasoning_step:
                double_lower_bounds, double_upper_bounds = self.get_hidden_bounds(double_output_lbs)

        return AbstractResults(**{
            'objective_ids': double_objective_ids,
            'output_lbs': double_output_lbs, 
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers,
            'lower_bounds': double_lower_bounds, 
            'upper_bounds': double_upper_bounds, 
            'slopes': double_slopes, 
            'lAs': double_lAs, 
            'cs': double_cs, 
            'rhs': double_rhs, 
            'constraints': constraints,
        })
        
    @beartype
    def _forward_input_sequential(self: 'NetworkAbstractor', domain_params: AbstractResults, decisions: torch.Tensor, simplify: bool) -> AbstractResults:
        assert len(decisions) == len(domain_params.cs) == len(domain_params.rhs) == \
               len(domain_params.input_lowers) == len(domain_params.input_uppers) == len(domain_params.objective_ids)
               
        batch = len(decisions)
        assert batch > 0
        
        # splitting input by decisions (perform splitting)
        new_input_lowers, new_input_uppers = self.input_split_idx(
            input_lowers=domain_params.input_lowers, 
            input_uppers=domain_params.input_uppers, 
            split_idx=decisions,
        )
        
        # 2 * batch
        double_objective_ids = torch.cat([domain_params.objective_ids, domain_params.objective_ids], dim=0)
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        
        sequential_output_lbs = []
        for b in tqdm.tqdm(range(2*batch), desc=f'_forward_input_sequential {double_cs.device}'):
            # create new inputs
            x_b = self.new_input(x_L=new_input_lowers[b:b+1], x_U=new_input_uppers[b:b+1])
            self.net.set_bound_opts(get_input_opt_params(stop_criterion_batch_any(double_rhs[b:b+1])))
            
            output_lb, _ = self.net.compute_bounds(
                x=(x_b,), 
                C=double_cs[b:b+1], 
                method=self.method,
                decision_thresh=double_rhs[b:b+1],
                reference_bounds=self.init_reference_bounds,
            )
            sequential_output_lbs.append(output_lb[0])
        sequential_output_lbs = torch.stack(sequential_output_lbs)

        return AbstractResults(**{
            'objective_ids': double_objective_ids,
            'output_lbs': sequential_output_lbs, 
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers,
            'cs': double_cs, 
            'rhs': double_rhs, 
        })
        
    @beartype
    def forward(self: 'NetworkAbstractor', decisions: list | torch.Tensor, domain_params: AbstractResults) -> AbstractResults:
        self.iteration += 1
        if self.input_split:
            if Settings.use_sequential_abstract_forward:
                forward_func = self._forward_input_sequential
            else:
                forward_func = self._forward_input  
        else:
            forward_func = self._forward_hidden
        return forward_func(domain_params=domain_params, decisions=decisions, simplify=False)

    
    # TODO: experimental function
    def compute_bounds(self, input_lowers, input_uppers, method, cs=None, rhs=None, reference_bounds=None, reuse_alpha=False):
        assert method in ['backward', 'crown-optimized']
        assert torch.all(input_lowers <= input_uppers)
        if os.environ.get('NEURALSAT_ASSERT'):
            assert not torch.equal(input_lowers, input_uppers)
        
        # gc_cuda()
        x = self.new_input(x_L=input_lowers, x_U=input_uppers)

        self.net.get_split_nodes()
        self.net.set_bound_opts(get_initialize_opt_params(lambda x: False))
        
        # backward mode
        lb, ub, aux_reference_bounds = self.net.init_alpha(
            x=(x,), 
            c=cs, 
            share_alphas=Settings.share_alphas, 
            bound_upper=True,
        )
        # FIXME: numerical error
        ub[lb > ub] = lb[lb > ub]
        
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(lb <= ub), f'{(lb > ub).sum()}\nlower: {lb[lb > ub].detach().cpu().tolist()}\nupper:{ub[lb > ub].detach().cpu().tolist()}\nnorm: {torch.norm(lb[lb > ub] - ub[lb > ub])}'
        
        # save to CPU
        lb = lb.detach().cpu()
        ub = ub.detach().cpu()
        
        if not Settings.use_extra_substitution:
            return (lb, ub), None
        
        if method == 'backward':
            lA, uA, lbias, ubias = self.get_input_A(self.device)
            coeffs = CoefficientMatrix(lA=lA, uA=uA, lbias=lbias, ubias=ubias)
            return (lb, ub), coeffs

        # lower bound
        lb, _ = self.net.compute_bounds(
            x=(x,), 
            C=cs,
            method=method,
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
            bound_lower=True,
            bound_upper=False,
        )
        lb = lb.detach().cpu()
        lA, _, lbias, _ = self.get_input_A('cpu')
        
        # upper bound
        _, ub = self.net.compute_bounds(
            x=(x,), 
            C=cs,
            method=method,
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
            bound_lower=False,
            bound_upper=True,
        )
        ub = ub.detach().cpu()
        _, uA, _, ubias = self.get_input_A('cpu')
        coeffs = CoefficientMatrix(lA=lA, uA=uA, lbias=lbias, ubias=ubias)
        # assert torch.all(lb <= ub + 1e-6), f'{(lb > ub).sum()} {lb[lb > ub]} {ub[lb > ub]}'
        
        return (lb, ub), coeffs
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.mode}, {self.method})'
        
        
    from .utils import (
        new_input,
        get_slope, set_slope,
        get_beta, set_beta, reset_beta, update_refined_beta,
        get_hidden_bounds,
        get_lAs, get_input_A,
        update_histories,
        hidden_split_idx, input_split_idx,
        build_lp_solver, solve_full_assignment,
        compute_stability,
    )
