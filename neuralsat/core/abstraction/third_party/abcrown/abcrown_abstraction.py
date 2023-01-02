from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch

from .auto_LiRPA.utils import stop_criterion_sum, stop_criterion_batch_any
from .auto_LiRPA import BoundedTensor, PerturbationLpNorm
from .lirpa import LiRPA

from core.activation.relu_domain import ReLUDomain, add_domain_parallel
from util.misc.logger import logger

import arguments

class ABCrownAbstraction:

    def __init__(self, net, spec):
        self.net = net
        self.spec = spec

        self.device = net.device
        self.dtype = arguments.Config['dtype']

        self.init_arguments()
        self.init_crown()


    def init_arguments(self):
        arguments.Config['solver'] = {}
        arguments.Config['solver']['alpha-crown'] = {}
        arguments.Config['solver']['alpha-crown']['lr_alpha'] = 0.1
        arguments.Config['solver']['alpha-crown']['iteration'] = 50
        arguments.Config['solver']['alpha-crown']['share_slopes'] = False

        arguments.Config['solver']['beta-crown'] = {}
        arguments.Config['solver']['beta-crown']['beta'] = True
        arguments.Config['solver']['beta-crown']['beta_warmup'] = True
        arguments.Config['solver']['beta-crown']['lr_alpha'] = 0.01
        arguments.Config['solver']['beta-crown']['lr_beta'] = 0.03
        arguments.Config['solver']['beta-crown']['lr_decay'] = 0.98
        arguments.Config['solver']['beta-crown']['optimizer'] = 'adam'
        arguments.Config['solver']['beta-crown']['iteration'] = 20

        arguments.Config['general'] = {}
        arguments.Config['general']['loss_reduction_func'] = 'sum'

        arguments.Config['bab'] = {}
        arguments.Config['bab']['branching'] = {}
        arguments.Config['bab']['branching']['reduceop'] = 'min'


    def init_crown(self):
        input_shape = self.net.input_shape
        x_range = torch.tensor(self.spec.bounds, dtype=self.dtype, device=self.device)
        data_min = x_range[:, 0].reshape(input_shape)
        data_max = x_range[:, 1].reshape(input_shape)
        data = x_range.mean(1).reshape(input_shape)

        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=data_min, x_U=data_max)
        self.x = BoundedTensor(data, ptb).to(self.device)

        c, self.decision_threshold, y, pidx = self.spec.extract()
        self.lirpa = LiRPA(model_ori=self.net.layers, 
                           input_shape=input_shape, 
                           device=self.device, 
                           c=c,
                           rhs=self.decision_threshold)

        self.assignment_mapping = {}
        for lid, lnodes in self.net.layers_mapping.items():
            for jj, node in enumerate(lnodes):
                self.assignment_mapping[(lid, jj)] = node

    
    def forward(self, input_lower, input_upper, extra_params=None):
        logger.debug('\t\tabstraction forward')
        if extra_params is None: # initialize
            output_ub, output_lb, _, _, primals, updated_mask, lA, all_lowers, all_uppers, self.pre_relu_indices, slope, history = self.lirpa.build_the_model(None, self.x, stop_criterion_func=stop_criterion_batch_any(self.decision_threshold))

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

            if (output_lb > self.decision_threshold).any():
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
            domain_list = add_domain_parallel(lA=lAs, 
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