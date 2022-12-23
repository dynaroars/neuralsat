from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch

from .auto_LiRPA import BoundedTensor, PerturbationLpNorm
from .auto_LiRPA.utils import stop_criterion_sum
from .beta_CROWN_solver import LiRPAConvNet

from core.activation.relu_domain import ReLUDomain, add_domain_parallel
from util.misc.logger import logger

import arguments

class ABCrownAbstraction:

    def __init__(self, net, spec):
        self.net = net
        self.spec = spec

        self.device = net.device
        self.dtype = arguments.Config['dtype']

        self.preprocess_spec()
        self.init_arguments()
        self.init_crown()


    def init_arguments(self):
        arguments.Config['solver'] = {}
        arguments.Config['solver']['alpha-crown'] = {}
        arguments.Config['solver']['alpha-crown']['lr_alpha'] = 0.01
        arguments.Config['solver']['alpha-crown']['iteration'] = 10
        arguments.Config['solver']['alpha-crown']['share_slopes'] = True
        # arguments.Config['solver']['alpha-crown']['no_joint_opt'] = False

        arguments.Config['solver']['beta-crown'] = {}
        arguments.Config['solver']['beta-crown']['beta'] = True
        arguments.Config['solver']['beta-crown']['beta_warmup'] = True
        arguments.Config['solver']['beta-crown']['lr_alpha'] = 0.01
        arguments.Config['solver']['beta-crown']['lr_beta'] = 0.05
        arguments.Config['solver']['beta-crown']['lr_decay'] = 0.98
        arguments.Config['solver']['beta-crown']['optimizer'] = 'adam'
        arguments.Config['solver']['beta-crown']['iteration'] = 10

        arguments.Config['general'] = {}
        arguments.Config['general']['loss_reduction_func'] = 'mean'
        # arguments.Config['general']['deterministic'] = False
        # arguments.Config['general']['double_fp'] = False

        arguments.Config['bab'] = {}
        arguments.Config['bab']['get_upper_bound'] = False

        arguments.Config['bab']['branching'] = {}
        arguments.Config['bab']['branching']['reduceop'] = 'min'


    def preprocess_spec(self):
        prop_mat, prop_rhs = self.spec.mat[0]

        assert len(prop_mat) == 1
        y = np.where(prop_mat[0] == 1)[0]
        if len(y) != 0:
            y = int(y)
        else:
            y = None
        target = np.where(prop_mat[0] == -1)[0]  # target label
        target = int(target) if len(target) != 0 else None  # Fix constant specification with no target label.
        if y is not None and target is None:
            y, target = target, y  # Fix vnnlib with >= const property.
        decision_threshold = prop_rhs[0]


        if y is not None:
            if self.net.n_output > 1:
                c = torch.zeros((1, 1, self.net.n_output), dtype=self.dtype, device=self.device)  # we only support c with shape of (1, 1, n)
                c[0, 0, y] = 1
                c[0, 0, target] = -1
            else:
                # Binary classifier, only 1 output. Assume negative label means label 0, postive label means label 1.
                c = (float(y) - 0.5) * 2 * torch.ones(size=(1, 1, 1), dtype=self.dtype, device=self.device)
        else:
            # if there is no ture label, we only verify the target output
            c = torch.zeros((1, 1, self.net.n_output), dtype=self.dtype, device=self.device)  # we only support c with shape of (1, 1, n)
            c[0, 0, target] = -1

        # print(c, decision_threshold)

        self.y = y
        self.target = target
        self.c = c
        self.decision_threshold = decision_threshold


    def init_crown(self):
        input_shape = self.net.input_shape
        x_range = torch.tensor(self.spec.bounds, dtype=self.dtype, device=self.device)
        data_min = x_range[:, 0].reshape(input_shape)
        data_max = x_range[:, 1].reshape(input_shape)
        data = x_range.mean(1).reshape(input_shape)

        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=data_min, x_U=data_max)
        self.x = BoundedTensor(data, ptb).to(self.device)

        self.lirpa = LiRPAConvNet(self.net.layers, 
                                  self.y, 
                                  self.target, 
                                  device=self.device, 
                                  in_size=input_shape, 
                                  deterministic=False, 
                                  conv_mode='patches', 
                                  c=self.c)


        self.assignment_mapping = {}
        for lid, lnodes in self.net.layers_mapping.items():
            for jj, node in enumerate(lnodes):
                self.assignment_mapping[(lid, jj)] = node

    
    def forward(self, input_lower, input_upper, extra_params=None):
        # print('ABcrown forward hehe')
        if extra_params is None: # initialize
            output_ub, output_lb, _, _, primals, updated_mask, lA, all_lowers, all_uppers, pre_relu_indices, slope, history = self.lirpa.build_the_model(None, self.x, stop_criterion_func=stop_criterion_sum(self.decision_threshold))

            # save ReLU node indices
            self.pre_relu_indices = pre_relu_indices

            # Keep only the alpha for the last layer.
            new_slope = defaultdict(dict)
            if slope is not None:
                for relu_layer, alphas in slope.items():
                    new_slope[relu_layer][self.lirpa.net.final_name] = alphas[self.lirpa.net.final_name]

            # TODO: Fixme to handle general cases
            init_domain = ReLUDomain(lA, 
                                     output_lb, 
                                     output_ub, 
                                     all_lowers, 
                                     all_uppers, 
                                     new_slope, 
                                     history=history, 
                                     primals=primals, 
                                     assignment_mapping=self.assignment_mapping).to_device(self.net.device, partial=True)

            if output_lb >= self.decision_threshold:
                init_domain.unsat = True

            return init_domain

        else:
            mask, lAs, all_lowers, all_uppers, slopes, betas, intermediate_betas, selected_domains, branching_decision = extra_params

            history = [sd.history for sd in selected_domains]

            split_history = [sd.split_history for sd in selected_domains]
            split = {}
            split["decision"] = [[bd] for bd in branching_decision]
            split["coeffs"] = [[1.] for i in range(len(branching_decision))]
            split["diving"] = 0

            ret = self.lirpa.get_lower_bound(all_lowers,
                                             all_uppers, 
                                             split, 
                                             slopes=slopes, 
                                             history=history, 
                                             split_history=split_history, 
                                             layer_set_bound=True, 
                                             betas=betas, 
                                             single_node_split=True, 
                                             intermediate_betas=intermediate_betas)
            
            dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = ret
            batch = len(selected_domains)
            domain_list = add_domain_parallel(lA=lAs[:2*batch], 
                                              lb=dom_lb[:2*batch], 
                                              ub=dom_ub[:2*batch], 
                                              lb_all=dom_lb_all[:2*batch], 
                                              up_all=dom_ub_all[:2*batch],
                                              selected_domains=selected_domains[:batch], 
                                              slope=slopes[:2*batch], 
                                              beta=betas[:2*batch],
                                              branching_decision=branching_decision, 
                                              decision_thresh=self.decision_threshold,
                                              split_history=split_history[:2*batch], 
                                              intermediate_betas=intermediate_betas[:2*batch],
                                              primals=primals[:2*batch] if primals is not None else None)

            return domain_list