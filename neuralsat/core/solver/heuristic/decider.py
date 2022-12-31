import torch

from core.abstraction.third_party.abcrown.branching_heuristics import choose_node_parallel_kFSB
from util.misc.logger import logger
import arguments

class Decider:

    def __init__(self, net):
        self.net = net
        self.device = net.device

        self.current_domain = None
        self.all_domains = None
        self.abstractor = None

        self.batch = arguments.Config['batch']

    def get(self, unassigned_variables):
        # print(unassigned_variables)
        assert self.current_domain is not None
        assert self.abstractor is not None

        # print('assignment:', self.current_domain.get_assignment())
        # print('cached decision:', self.current_domain.next_decision)
        if self.current_domain.next_decision is not None:
            logger.debug(f'\t\tdecider_get_one {self.current_domain.next_decision}')
            return self.current_domain.next_decision

        # print('batch:', self.batch)
        # print(self.current_domain.valid)

        mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = self.get_domain_params(self.current_domain)
        history = [sd.history for sd in selected_domains]

        branching_decision = choose_node_parallel_kFSB(orig_lbs, 
                                                       orig_ubs, 
                                                       mask, 
                                                       self.abstractor.core.lirpa, 
                                                       self.abstractor.core.pre_relu_indices, 
                                                       lAs, 
                                                       branching_reduceop=arguments.Config['bab']['branching']['reduceop'], 
                                                       slopes=slopes, 
                                                       betas=betas,
                                                       history=history)

        logger.debug(f'\t\tdecider_get_one {branching_decision}')
        node = self.convert_crown_decision(branching_decision[0])
        self.current_domain.next_decision = node
        return node

    def convert_crown_decision(self, branching_decision):
        # decision_layer, decision_index = branching_decision
        return self.abstractor.core.assignment_mapping[tuple(branching_decision)]


    def get_batch_decisions(self, current_domain):
        # print(len(self.all_domains))
        extra_params = self.get_domain_params(current_domain)


        mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = extra_params
        if len(selected_domains) == 0:
            return None, None, None
            
        history = [sd.history for sd in selected_domains]
        # TODO: Fixme: use cached decision
        branching_decision = choose_node_parallel_kFSB(orig_lbs, 
                                                       orig_ubs, 
                                                       mask, 
                                                       self.abstractor.core.lirpa, 
                                                       self.abstractor.core.pre_relu_indices, 
                                                       lAs, 
                                                       branching_reduceop=arguments.Config['bab']['branching']['reduceop'], 
                                                       slopes=slopes, 
                                                       betas=betas,
                                                       history=history)
                                                       
        logger.debug(f'\t\tdecider_get_batch (first 5) {branching_decision[:5]}')
        for idx, d in enumerate(selected_domains):
            d.valid = False # mark as processed
            if d.next_decision is None:
                d.next_decision = self.convert_crown_decision(branching_decision[idx])
            else:
                assert d.next_decision == self.convert_crown_decision(branching_decision[idx])
        # else:
        #     branching_decision = crown_decision
        extra_params = extra_params + (branching_decision, )
        return [self.convert_crown_decision(bd) for bd in branching_decision], selected_domains, extra_params

    # copy abcrown
    def get_domain_params(self, current_domain):
        lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []

        selected_domain_hashes = []

        idx = 0
        if current_domain is not None:
            current_domain.to_device(self.device, partial=True)
            lAs.append(current_domain.lA)
            lower_all.append(current_domain.lower_all)
            upper_all.append(current_domain.upper_all)
            slopes_all.append(current_domain.slope)
            betas_all.append(current_domain.beta)
            intermediate_betas_all.append(current_domain.intermediate_betas)
            selected_candidate_domains.append(current_domain)
            selected_domain_hashes.append(current_domain.get_assignment())
            idx += 1

        if self.batch > idx:
            for k, selected_candidate_domain in self.all_domains.items():
                if (not selected_candidate_domain.unsat) and selected_candidate_domain.valid and (selected_candidate_domain.get_assignment() not in selected_domain_hashes):
                    # print('--------> select:', selected_candidate_domain.get_assignment())
                    selected_candidate_domain.to_device(self.device, partial=True)
                    # selected_candidate_domain.valid = False  # set False to avoid another pop
                    lAs.append(selected_candidate_domain.lA)
                    lower_all.append(selected_candidate_domain.lower_all)
                    upper_all.append(selected_candidate_domain.upper_all)
                    slopes_all.append(selected_candidate_domain.slope)
                    betas_all.append(selected_candidate_domain.beta)
                    intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
                    selected_candidate_domains.append(selected_candidate_domain)
                    selected_domain_hashes.append(selected_candidate_domain.get_assignment())
                    idx += 1
                    if idx == self.batch:
                        break
                # selected_candidate_domain.valid = False   

        batch = len(selected_candidate_domains)
        if batch == 0:
            return None, None, None, None, None, None, None, []

        # set False to avoid another pop
        # for domain in selected_candidate_domains:
        #     domain.valid = False

        lower_bounds = []
        for j in range(len(lower_all[0])):
            lower_bounds.append(torch.cat([lower_all[i][j] for i in range(batch)]))
        lower_bounds = [t.to(device=self.device, non_blocking=True) for t in lower_bounds]

        upper_bounds = []
        for j in range(len(upper_all[0])):
            upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
        upper_bounds = [t.to(device=self.device, non_blocking=True) for t in upper_bounds]

        # Reshape to batch first in each list.
        new_lAs = []
        for j in range(len(lAs[0])):
            new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
        # Transfer to GPU.
        new_lAs = [t.to(device=self.device, non_blocking=True) for t in new_lAs]

        slopes = []
        if slopes_all[0] is not None:
            if isinstance(slopes_all[0], dict):
                # Per-neuron slope, each slope is a dictionary.
                slopes = slopes_all
            else:
                for j in range(len(slopes_all[0])):
                    slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

        # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
        lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
        upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]
        
        
        # Recompute the mask on GPU.
        new_masks = []
        for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
            new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())
        return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains
