from collections import defaultdict
from torch import nn
import numpy as np
import torch

from .util import _compute_babsr_scores, _get_branching_op, _get_bias_term

class DecisionHeuristic:
    
    def __init__(self, branching_candidates, input_split, branching_reduceop='max'):
        self.branching_candidates = branching_candidates
        self.input_split = input_split
        self.branching_reduceop = branching_reduceop
        

    @torch.no_grad()
    def __call__(self, abstractor, domain_params):
        if self.input_split:
            return self.input_branching(domain_params=domain_params)
         
        return self.filtered_smart_branching(
            abstractor=abstractor, 
            domain_params=domain_params,
        )

    # hidden branching
    def filtered_smart_branching(self, abstractor, domain_params):
        batch = len(domain_params.masks[0])
        reduce_op = _get_branching_op(self.branching_reduceop)
        topk = min(self.branching_candidates, int(sum([i.sum() for i in domain_params.masks]).item()))
        
        score, intercept_tb = _compute_babsr_scores(
            abstractor=abstractor, 
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            lAs=domain_params.lAs, 
            batch=batch, 
            masks=domain_params.masks, 
            reduce_op=reduce_op, 
            number_bounds=domain_params.cs.shape[1]
        )
        
        # batch * 2
        # hidden
        double_lower_bounds = [torch.cat([i, i]) for i in domain_params.lower_bounds]
        double_upper_bounds = [torch.cat([i, i]) for i in domain_params.upper_bounds]
        
        # slope
        if isinstance(domain_params.slopes, dict):
            sps = defaultdict(dict)
            for k, vv in domain_params.slopes.items():
                sps[k] = {}
                for kk, v in vv.items():
                    sps[k][kk] = torch.cat([v, v], dim=2)
        else:
            sps = [torch.cat([i, i]) for i in domain_params.slopes]
        
        # spec
        double_cs = torch.cat([domain_params.cs, domain_params.cs])
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs])
        
        # input
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers])
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers])
        
        assert torch.all(double_input_lowers <= double_input_uppers)

        # use to convert an index to its layer and offset.
        score_length = np.cumsum([len(score[i][0]) for i in range(len(score))])
        score_length = np.insert(score_length, 0, 0)

        # top-k candidates among all layers for two kinds of scores.
        all_score = torch.cat(score, dim=1)
        score_idx = torch.topk(all_score, topk)
        score_idx_indices = score_idx.indices.cpu()
        
        all_itb = torch.cat(intercept_tb, dim=1)
        itb_idx = torch.topk(all_itb, topk, largest=False)  # k-smallest elements.
        itb_idx_indices = itb_idx.indices.cpu()

        k_decision = []
        k_ret = torch.empty(size=(topk, batch * 2), device=domain_params.lower_bounds[0].device, requires_grad=False)
        # only set slope once
        set_slope = True 
        for k in range(topk):
            # top-k candidates from the slope scores.
            decision_index = score_idx_indices[:, k]
            decision_max_ = [] # higher is better
            for l in decision_index:
                l = l.item()
                layer = np.searchsorted(score_length, l, side='right') - 1
                idx = l - score_length[layer]
                decision_max_.append([layer, idx])

            # top-k candidates from the intercept (backup) scores.
            decision_index = itb_idx_indices[:, k]
            decision_min_ = [] # lower is better
            for l in decision_index:
                l = l.item()
                layer = np.searchsorted(score_length, l, side='right') - 1
                idx = l - score_length[layer]
                decision_min_.append([layer, idx])
            
            # top-k decisions
            k_decision.append(decision_max_ + decision_min_)

            # lower bounds of the temporal splits
            abs_ret = abstractor._forward_hidden(
                input_lowers=double_input_lowers,
                input_uppers=double_input_uppers,
                lower_bounds=double_lower_bounds, # hidden bounds
                upper_bounds=double_upper_bounds, # hidden bounds
                branching_decisions=k_decision[-1], 
                slopes=sps if set_slope else [], 
                cs=double_cs,
                rhs=double_rhs,
                use_beta=False, 
                shortcut=True, 
            )
            k_ret_lbs = abs_ret.output_lbs

            # consider the max improvement among multi bounds in one C matrix
            k_ret_lbs = (k_ret_lbs - torch.cat([double_rhs, double_rhs])).max(-1).values
            # set slope once
            set_slope = False 

            # build masks indicates invalid scores (1) for stable neurons
            mask_score = (score_idx.values[:, k] <= 1e-4).float()  
            mask_itb = (itb_idx.values[:, k] >= -1e-4).float()
            k_ret[k] = reduce_op((k_ret_lbs.view(-1) - torch.cat([mask_score, mask_itb]).repeat(2) * 1e6).reshape(2, -1), dim=0).values # (top-k, batch*2)

        # find corresponding decision
        i_idx = k_ret.topk(1, 0)
        rets = i_idx.values.cpu().numpy()[0]
        rets_indices = i_idx.indices.cpu().numpy()[0]
        decision_tmp = [k_decision[rets_indices[ii]][ii] for ii in range(batch * 2)]
        final_decision = [[] for b in range(batch)]

        for b in range(batch):
            mask_item = [m[b] for m in domain_params.masks]
            thres = max(rets[b], rets[b + batch])
            if thres > -1e4:
                if rets[b] > rets[b + batch]:
                    decision = decision_tmp[b]
                else:
                    decision = decision_tmp[b + batch]
                    
                if mask_item[decision[0]][decision[1]] != 0:
                    final_decision[b].append(decision)
                    assert mask_item[decision[0]][decision[1]] == 1, "selected decision node should be unstable!"
                    mask_item[decision[0]][decision[1]] = 0

        # no valid scores, choose randomly
        for b in range(batch):
            mask_item = [m[b] for m in domain_params.masks]
            if len(final_decision[b]) == 0:
                for preferred_layer in np.random.choice(len(abstractor.pre_relu_indices), len(abstractor.pre_relu_indices), replace=False):
                    if len(mask_item[preferred_layer].nonzero(as_tuple=False)) != 0:
                        final_decision[b].append([preferred_layer, mask_item[preferred_layer].nonzero(as_tuple=False)[0].item()])
                        mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                        break
        
        final_decision = sum(final_decision, [])
        return final_decision


    def input_branching(self, domain_params, topk=1):
        final_decision = torch.topk(domain_params.input_uppers.flatten(1) - domain_params.input_lowers.flatten(1), topk, -1).indices
        return final_decision
        
