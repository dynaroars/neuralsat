from attack import attack_with_general_specs, test_conditions
from core.abstraction.polytope import PolytopeAbstraction
from .abcrown_new.lirpa_naive import LiRPANaive
from .input_domain import InputDomainList

from util.misc.logger import logger
import arguments

from sortedcontainers import SortedList
from torch import nn
import numpy as np
import torch
import math
import time


def stop_criterion_batch_any(threshold=0):
    return lambda x: (x > threshold).any(dim=1)


class InputSolver:

    def __init__(self, net, spec):
        self.net = net
        self.spec = spec

        self.batch = arguments.Config['batch']
        self.device = arguments.Config['device']
        self.dtype = arguments.Config['dtype']

        self.total_domains = 0
        self.iteration = 0
        self.assignment = None # cex

        self.domains = InputDomainList()

        self.init_abstractor()


    def get_assignment(self):
        return self.assignment


    def init_abstractor(self):
        self.c, self.rhs, _, _ = self.spec.extract()
        self.abstractor = LiRPANaive(model_ori=self.net.layers, 
                                     input_shape=self.net.input_shape, 
                                     device=self.device, 
                                     c=self.c,
                                     rhs=self.rhs)


    def process_init(self):
        # extract input bounds
        x_range = torch.tensor(self.spec.bounds, dtype=self.dtype, device=self.device)
        self.input_lb = x_range[:, 0].reshape(self.net.input_shape)
        self.input_ub = x_range[:, 1].reshape(self.net.input_shape)

        # compute abstraction
        output_lb, output_ub = self.compute_abstraction(self.input_lb, self.input_ub)

        # choose split dimension
        split_idx = self.input_select(self.input_lb, self.input_ub, topk=1)

        # add domains
        self.add_domains(self.input_lb, self.input_ub, output_lb, output_ub, split_idx)


    def process_batch(self):
        # step 1: pick domains
        dm_l_all, dm_u_all, split_idx = self.pick_domains()
        assert torch.all(dm_l_all <= dm_u_all)

        # step 2: split
        new_dm_l_all, new_dm_u_all = self.input_split(dm_l_all, dm_u_all, split_idx)
        assert torch.all(new_dm_l_all <= new_dm_u_all)

        # step 3: compute abstraction
        output_lb, output_ub = self.compute_abstraction(new_dm_l_all, new_dm_u_all)

        # step 4: choose split dimension
        split_idx = self.input_select(new_dm_l_all, new_dm_u_all, topk=1)

        # step 5: add domains
        self.add_domains(new_dm_l_all.detach(), new_dm_u_all.detach(), output_lb, output_ub, split_idx)

        if arguments.Config['print_progress']:
            logger.info(f'Process batch (iteration={self.iteration}) \t domains={len(self.domains)}/{self.total_domains}')


    def solve(self):
        logger.info('Input splitting')
        
        self.process_init()

        while len(self.domains) > 0:
            self.iteration += 1

            # step 1: check adv
            is_attacked, self.assignment = self.attack(topk=20)
            if is_attacked:
                return arguments.ReturnStatus.SAT

            # step 2: process one batch
            self.process_batch()

            # step 3: check early stopping
            if len(self.domains) > arguments.Config['max_input_branch']:
                return arguments.ReturnStatus.UNKNOWN

        return arguments.ReturnStatus.UNSAT


    def add_domains(self, input_lower, input_upper, output_lower, output_upper, split_idx):
        self.total_domains += len(input_lower)
        self.domains.add_batch(input_lower=input_lower, 
                               input_upper=input_upper, 
                               output_lower=output_lower, 
                               output_upper=output_upper, 
                               c=self.c,
                               rhs=self.rhs, 
                               split_idx=split_idx)



    def pick_domains(self):
        return self.domains.pick_out_batch(self.batch, self.device)


    @torch.no_grad()    
    def input_select(self, dm_l_all, dm_u_all, topk=1):
        dm_l_all = dm_l_all.flatten(1)
        dm_u_all = dm_u_all.flatten(1)
        split_idx = torch.topk(dm_u_all - dm_l_all, topk, -1).indices
        return split_idx


    @torch.no_grad()
    def input_split(self, dm_l_all, dm_u_all, split_idx, split_depth=1):
        dm_l_all = dm_l_all.flatten(1)
        dm_u_all = dm_u_all.flatten(1)

        dm_l_all_cp = dm_l_all.clone()
        dm_u_all_cp = dm_u_all.clone()

        remaining_depth = split_depth
        input_dim = dm_l_all.shape[1]
        while remaining_depth > 0:
            for i in range(min(input_dim, remaining_depth)):
                indices = torch.arange(dm_l_all_cp.shape[0])
                copy_num = dm_l_all_cp.shape[0] // dm_l_all.shape[0]
                idx = split_idx[:,i].repeat(copy_num).long()

                dm_l_all_cp_tmp = dm_l_all_cp.clone()
                dm_u_all_cp_tmp = dm_u_all_cp.clone()

                mid = (dm_l_all_cp[indices, idx] + dm_u_all_cp[indices, idx]) / 2

                dm_l_all_cp[indices, idx] = mid
                dm_u_all_cp_tmp[indices, idx] = mid
                dm_l_all_cp = torch.cat([dm_l_all_cp, dm_l_all_cp_tmp])
                dm_u_all_cp = torch.cat([dm_u_all_cp, dm_u_all_cp_tmp])
            remaining_depth -= min(input_dim, remaining_depth)

        new_dm_l_all = dm_l_all_cp.reshape(-1, *self.net.input_shape[1:])
        new_dm_u_all = dm_u_all_cp.reshape(-1, *self.net.input_shape[1:])

        return new_dm_l_all, new_dm_u_all


    @torch.no_grad()
    def compute_abstraction(self, dm_l_all, dm_u_all):
        (lb, ub), _ = self.abstractor(dm_l_all, dm_u_all, reset_param=True, stop_criterion_func=stop_criterion_batch_any(self.rhs))
        return lb, ub


    def attack(self, topk=10):
        worst_indices = self.domains.get_topk_indices(k=topk)
        best_indices = self.domains.get_topk_indices(k=1, largest=True)
        indices = worst_indices.numpy().tolist() + best_indices.numpy().tolist()

        dm_l, dm_u = [], []
        for idx in indices:
            val = self.domains[idx]
            dm_l.append(val[0][None].detach().cpu())
            dm_u.append(val[1][None].detach().cpu())

        adv_example = torch.cat([torch.cat([dm_l[i], dm_u[i]]) for i in range(len(indices))])
        adv_example = adv_example.unsqueeze(0).to(self.device, non_blocking=True)

        prop_mat = self.c.repeat_interleave(len(indices) * 2, dim=0).unsqueeze(1)
        prop_rhs = self.rhs.repeat_interleave(len(indices) * 2, dim=0)

        cond_mat = [[prop_mat.shape[1] for i in range(prop_mat.shape[0])]]
        # [1, num_or, input_shape]
        prop_mat = prop_mat.view(1, -1, prop_mat.shape[-1]).to(self.device, non_blocking=True)
        # [1, num_spec, output_dim]
        prop_rhs = prop_rhs.view(1, -1).to(self.device, non_blocking=True).to(self.device, non_blocking=True)
        # [1, num_spec]

        data_max = torch.cat([torch.cat([dm_u[i], dm_u[i]]) for i in range(len(indices))])
        data_max = data_max.unsqueeze(0).to(self.device, non_blocking=True)

        data_min = torch.cat([torch.cat([dm_l[i], dm_l[i]]) for i in range(len(indices))])
        data_min = data_min.unsqueeze(0).to(self.device, non_blocking=True)

        alpha = (data_max - data_min).max() / 4

        best_deltas, _ = attack_with_general_specs(model=self.net, 
                                                   X=adv_example, 
                                                   data_min=data_min, 
                                                   data_max=data_max, 
                                                   C_mat=prop_mat, 
                                                   rhs_mat=prop_rhs, 
                                                   cond_mat=cond_mat, 
                                                   same_number_const=True, 
                                                   alpha=alpha, 
                                                   attack_iters=5, 
                                                   num_restarts=30, 
                                                   only_replicate_restarts=True)

        attack_image = best_deltas + adv_example.squeeze(1)
        attack_image = torch.min(torch.max(attack_image, data_min), data_max)

        attack_output = self.net(attack_image.view(-1, *attack_image.shape[2:])).view(*attack_image.shape[:2], -1)                                                  
        
        if test_conditions(attack_image.unsqueeze(1), 
                           attack_output.unsqueeze(1), 
                           prop_mat.unsqueeze(1), 
                           prop_rhs, 
                           cond_mat, 
                           True, 
                           data_max, 
                           data_min).all():
            attack_image = attack_image.squeeze(0)
            for i in range(len(attack_image)):
                adv = attack_image[i]
                if (adv <= self.input_ub).all() and (adv >= self.input_lb).all():
                    if self.spec.check_solution(self.net(adv)):
                        return True, adv

        return False, None
