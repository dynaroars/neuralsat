from collections import defaultdict
from torch import nn
import numpy as np
import random
import torch

from .util import _compute_babsr_scores
from setting import Settings

LARGE = 1e6
SMALL = 1.0 / LARGE

class DecisionHeuristic:
    
    def __init__(self, decision_topk, input_split, decision_reduceop=torch.max, random_selection=False, seed=0):
        self.decision_topk = decision_topk
        self.input_split = input_split
        self.decision_reduceop = decision_reduceop
        self.random_selection = random_selection
        
        # if Settings.test:
        #     random.seed(seed)

    @torch.no_grad()
    def __call__(self, abstractor, domain_params):
        if Settings.test:
            return self.naive_randomized_branching(domain_params=domain_params, mode=random.choice(['scale', 'distance', 'polarity']))
        
        if self.input_split:
            return self.input_branching(domain_params=domain_params)
        
        if self.random_selection:
            if random.uniform(0, 1) > 0.7:
                return self.naive_randomized_branching(domain_params=domain_params, mode=random.choice(['scale', 'distance', 'polarity']))
            
        return self.filtered_smart_branching(
            abstractor=abstractor, 
            domain_params=domain_params,
        )


    def get_topk_scores(self, abstractor, domain_params, topk_scores, topk_backup_scores, score_length, topk):
        class TMP:
            pass
        
        topk_decisions = []
        batch = len(domain_params.masks[0])
        topk_output_lbs = torch.empty(
            size=(topk, batch * 2), 
            device=domain_params.lower_bounds[0].device, 
            requires_grad=False,
        )
        
        # hidden
        double_lower_bounds = [torch.cat([i, i]) for i in domain_params.lower_bounds]
        double_upper_bounds = [torch.cat([i, i]) for i in domain_params.upper_bounds]
        
        # slope
        double_slopes = defaultdict(dict)
        for k, v in domain_params.slopes.items():
            double_slopes[k] = {kk: torch.cat([vv, vv], dim=2) for (kk, vv) in v.items()}
        
        # spec
        double_cs = torch.cat([domain_params.cs, domain_params.cs])
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs])
        
        # input
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers])
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers])
        
        assert torch.all(double_input_lowers <= double_input_uppers)
        
        topk_scores_indices = topk_scores.indices.cpu()
        topk_backup_scores_indices = topk_backup_scores.indices.cpu()
        
        for k in range(topk):
            # top-k candidates from scores
            decision_max = [] # higher is better
            for idx in topk_scores_indices[:, k]:
                idx = idx.item()
                layer_idx = np.searchsorted(score_length, idx, side='right') - 1
                neuron_idx = idx - score_length[layer_idx]
                decision_max.append([layer_idx, neuron_idx])

            # top-k candidates from backup scores.
            decision_min = [] # lower is better
            for idx in topk_backup_scores_indices[:, k]:
                idx = idx.item()
                layer_idx = np.searchsorted(score_length, idx, side='right') - 1
                neuron_idx = idx - score_length[layer_idx]
                decision_min.append([layer_idx, neuron_idx])
            
            # top-k candidates
            topk_decisions.append(decision_max + decision_min)

            k_domain_params = TMP()
            k_domain_params.input_lowers = double_input_lowers # input bounds
            k_domain_params.input_uppers = double_input_uppers # input bounds
            k_domain_params.lower_bounds = double_lower_bounds # hidden bounds
            k_domain_params.upper_bounds = double_upper_bounds # hidden bounds
            k_domain_params.slopes = double_slopes if k == 0 else []
            k_domain_params.cs = double_cs
            k_domain_params.rhs = double_rhs
            
            abs_ret = abstractor._naive_forward_hidden(
                domain_params=k_domain_params,
                decisions=topk_decisions[-1], 
            )
            # improvements over specification
            k_output_lbs = (abs_ret.output_lbs - torch.cat([double_rhs, double_rhs])).max(-1).values

            # invalid scores for stable neurons
            invalid_mask_scores = (topk_scores.values[:, k] <= SMALL).to(torch.get_default_dtype())  
            invalid_mask_backup_scores = (topk_backup_scores.values[:, k] >= -SMALL).to(torch.get_default_dtype())  
            invalid_mask = torch.cat([invalid_mask_scores, invalid_mask_backup_scores]).repeat(2) * LARGE
            topk_output_lbs[k] = self.decision_reduceop((k_output_lbs.view(-1) - invalid_mask).reshape(2, -1), dim=0).values

        return topk_output_lbs, topk_decisions
    
    
    # hidden branching
    def filtered_smart_branching(self, abstractor, domain_params):
        batch = len(domain_params.masks[0])
        topk = min(self.decision_topk, int(sum([i.sum() for i in domain_params.masks]).item()))
        
        # babsr scores
        scores, backup_scores = _compute_babsr_scores(
            abstractor=abstractor, 
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            lAs=domain_params.lAs, 
            batch=batch, 
            masks=domain_params.masks, 
            reduce_op=self.decision_reduceop, 
            number_bounds=domain_params.cs.shape[1]
        )
        
        # convert an index to its layer and offset
        score_length = np.insert(np.cumsum([len(scores[i][0]) for i in range(len(scores))]), 0, 0)

        # top-k candidates
        topk_scores = torch.topk(torch.cat(scores, dim=1), topk)
        topk_backup_scores = torch.topk(torch.cat(backup_scores, dim=1), topk, largest=False)  

        topk_output_lbs, topk_decisions = self.get_topk_scores(
            abstractor=abstractor, 
            domain_params=domain_params, 
            topk_scores=topk_scores, 
            topk_backup_scores=topk_backup_scores, 
            score_length=score_length, 
            topk=topk,
        )
        
        # best improvements
        if self.random_selection:
            best_output_lbs_indices = np.random.random_integers(low=0, high=len(topk_output_lbs)-1, size=topk_output_lbs.shape[1])
            topk_output_lbs_np = topk_output_lbs.detach().cpu().numpy()
            best_output_lbs = np.array([topk_output_lbs_np[best_output_lbs_indices[ii]][ii] for ii in range(batch * 2)])
        else:
            best = topk_output_lbs.topk(1, 0)
            best_output_lbs = best.values.cpu().numpy()[0]
            best_output_lbs_indices = best.indices.cpu().numpy()[0]
            
        # find corresponding decisions
        all_topk_decisions = [topk_decisions[best_output_lbs_indices[ii]][ii] for ii in range(batch * 2)]
        final_decision = [[] for b in range(batch)]
        for b in range(batch):
            mask_item = [m[b] for m in domain_params.masks]
            # valid scores
            if max(best_output_lbs[b], best_output_lbs[b + batch]) > -LARGE:
                decision = all_topk_decisions[b] if best_output_lbs[b] > best_output_lbs[b + batch] else all_topk_decisions[b + batch]
                if mask_item[decision[0]][decision[1]] != 0:
                    final_decision[b].append(decision)
                    mask_item[decision[0]][decision[1]] = 0
            # if invalid scores
            if len(final_decision[b]) == 0: 
                # use random decisions 
                for layer in np.random.choice(len(abstractor.pre_relu_indices), len(abstractor.pre_relu_indices), replace=False):
                    if len(mask_item[layer].nonzero(as_tuple=False)) != 0:
                        final_decision[b].append([layer, mask_item[layer].nonzero(as_tuple=False)[0].item()])
                        mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                        break
        
        final_decision = sum(final_decision, [])
            
        return final_decision


    def input_branching(self, domain_params, topk=1):
        final_decision = torch.topk(domain_params.input_uppers.flatten(1) - domain_params.input_lowers.flatten(1), topk, -1).indices
        return final_decision
        

    def naive_randomized_branching(self, domain_params, mode):
        batch = len(domain_params.masks[0])
        # print('shape', [l.shape for l in domain_params.lower_bounds[:-1]])
        # print('lower', [l for l in domain_params.lower_bounds[:-1]])
        # print('upper', [u for u in domain_params.upper_bounds[:-1]])
        if mode == 'distance':
            scores = [torch.min(u, -l) for (u, l) in zip(domain_params.upper_bounds[:-1], domain_params.lower_bounds[:-1])]
        elif mode == 'polarity':
            scores = [(u * l) / (l - u) for (u, l) in zip(domain_params.upper_bounds[:-1], domain_params.lower_bounds[:-1])]
        elif mode == 'scale':
            scores = [torch.min(u, -l) / (u + l).abs() for (u, l) in zip(domain_params.upper_bounds[:-1], domain_params.lower_bounds[:-1])]
        else:
            raise NotImplementedError()
            
        # print('mask', sum([s.sum() for s in domain_params.masks]))
        # print('mask', [s.bool().shape for s in domain_params.masks])
        # print('scores', [s for s in scores])
        masked_scores = [torch.where(m.bool(), s, 0.0) for (s, m) in zip(scores, domain_params.masks)]
        # print('masked_scores', [s for s in masked_scores])
        # print('masked_scores', [s.shape for s in masked_scores])
        # exit()
        # best_scores = [s.topk(1, 1) for s in scores]
        best_scores = [s.topk(1, 1) for s in masked_scores]
        
        # print('scores', scores)
        
        # print('best_scores', [s.values for s in best_scores])
        best_scores_all_layers = torch.cat([s.values for s in best_scores], dim=1)
        best_scores_all_layers_indices = torch.cat([s.indices for s in best_scores], dim=1).detach().cpu().numpy()
        # print('best_scores', best_scores_all_layers)
        # print('best_scores_indices', best_scores_all_layers_indices)
        # print('best_scores', [s.indices for s in best_scores])
        best_scores_all = best_scores_all_layers.topk(1, 1)
        # print('best_scores_all', best_scores_all.values.flatten())
        assert (best_scores_all.values > 0.0).all()
        layer_ids = best_scores_all.indices[:, 0].detach().cpu().numpy()
        assert len(layer_ids) == batch
        # print(layer_ids)
        decisions = [[layer_ids[b], best_scores_all_layers_indices[b, layer_ids[b]]] for b in range(batch)]
        # print('decisions:', mode, decisions)
        
        return decisions

        # print()
        