from __future__ import annotations
from collections import defaultdict
from beartype import beartype
import numpy as np
import typing
import random
import torch

if typing.TYPE_CHECKING:
    import abstractor
    
from heuristic.util import _compute_babsr_scores
from util.misc.result import AbstractResults
from setting import Settings

LARGE = 1e6
SMALL = 1.0 / LARGE

class DecisionHeuristic:
    
    @beartype
    def __init__(self, input_split: bool, decision_topk: int, decision_method: str) -> None:
        self.input_split = input_split
        self.decision_topk = decision_topk
        self.decision_method = decision_method
        self.decision_reduceop = torch.max
        

    @beartype
    @torch.no_grad()
    def __call__(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', domain_params: AbstractResults) -> torch.Tensor | list[list]:
        if Settings.test:
            # hidden split
            return self.naive_hidden_branching(
                domain_params=domain_params, 
                abstractor=abstractor, 
                mode=random.choice(['scale', 'distance', 'polarity']),
            )
        
        if self.input_split:
            if (self.decision_method == 'smart') and (domain_params.lAs is not None):
                return self.smart_input_branching(
                    domain_params=domain_params,
                    abstractor=abstractor, 
                )
                
            return self.naive_input_branching(
                domain_params=domain_params,
                abstractor=abstractor, 
            )
        
        # hidden split
        if self.decision_method != 'smart':
            if random.uniform(0, 1) > 0.7:
                return self.naive_hidden_branching(
                    domain_params=domain_params, 
                    abstractor=abstractor, 
                    mode=random.choice(['scale', 'distance', 'polarity'])
                )
            
        return self.smart_hidden_branching(
            abstractor=abstractor, 
            domain_params=domain_params,
        )


    @beartype
    def get_topk_scores(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', domain_params: AbstractResults, 
                        topk_scores: torch.return_types.topk, topk_backup_scores: torch.return_types.topk, score_length: np.ndarray, 
                        topk: int) -> tuple[torch.Tensor, list]:
        
        topk_decisions = []
        batch = len(domain_params.input_lowers)
        topk_output_lbs = torch.empty(
            size=(topk, batch * 2), 
            device=domain_params.input_lowers.device, 
            requires_grad=False,
        )
        
        # hidden
        double_lower_bounds = {k: torch.cat([v, v]) for k, v in domain_params.lower_bounds.items()}
        double_upper_bounds = {k: torch.cat([v, v]) for k, v in domain_params.upper_bounds.items()}
        
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
                layer_name = abstractor.net.split_nodes[layer_idx].name
                layer_split_point = abstractor.net.split_activations[layer_name][0][0].get_split_point()
                neuron_idx = idx - score_length[layer_idx]
                if layer_split_point is not None: # relu
                    decision_max.append([layer_name, neuron_idx, layer_split_point])
                else: # general activation
                    raise NotImplementedError

            # top-k candidates from backup scores.
            decision_min = [] # lower is better
            for idx in topk_backup_scores_indices[:, k]:
                idx = idx.item()
                layer_idx = np.searchsorted(score_length, idx, side='right') - 1
                layer_name = abstractor.net.split_nodes[layer_idx].name
                layer_split_point = abstractor.net.split_activations[layer_name][0][0].get_split_point()
                neuron_idx = idx - score_length[layer_idx]
                if layer_split_point is not None: # relu
                    decision_min.append([layer_name, neuron_idx, layer_split_point])
                else: # general activation
                    raise NotImplementedError
            
            # top-k candidates
            topk_decisions.append(decision_max + decision_min)

            k_domain_params = AbstractResults(**{
                'input_lowers': double_input_lowers,
                'input_uppers': double_input_uppers,
                'lower_bounds': double_lower_bounds,
                'upper_bounds': double_upper_bounds,
                'slopes': double_slopes if k == 0 else [],
                'cs': double_cs,
                'rhs': double_rhs,
            })
            
            abs_ret = abstractor._forward_hidden(
                domain_params=k_domain_params,
                decisions=topk_decisions[-1], 
                simplify=True
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
    @beartype
    def smart_hidden_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', 
                                 domain_params: AbstractResults) -> list[list]:
        batch = len(domain_params.input_lowers)
        topk = min(self.decision_topk, int(sum([i.sum() for (_, i) in domain_params.masks.items()]).item()))
        split_node_names = [_.name for _ in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}
        
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
        if self.decision_method != 'smart':
            best_output_lbs_indices = np.random.random_integers(low=0, high=len(topk_output_lbs)-1, size=topk_output_lbs.shape[1])
            topk_output_lbs_np = topk_output_lbs.detach().cpu().numpy()
            best_output_lbs = np.array([topk_output_lbs_np[best_output_lbs_indices[ii]][ii] for ii in range(batch * 2)])
        else:
            best = topk_output_lbs.topk(1, 0)
            best_output_lbs = best.values.cpu().numpy()[0]
            best_output_lbs_indices = best.indices.cpu().numpy()[0]
            
        # align decisions
        all_topk_decisions = [topk_decisions[best_output_lbs_indices[ii]][ii] for ii in range(batch * 2)]
        final_decision = [[] for b in range(batch)]

        for b in range(batch):
            mask_item = {k: domain_params.masks[k][b].clone() for k in split_node_names}
            # valid scores
            if max(best_output_lbs[b], best_output_lbs[b + batch]) > -LARGE:
                n_name, n_id, n_point = all_topk_decisions[b] if best_output_lbs[b] > best_output_lbs[b + batch] else all_topk_decisions[b + batch]
                if n_point is not None: # relu
                    if mask_item[n_name][n_id] != 0: # unstable relu
                        final_decision[b].append([n_name, n_id, n_point])
                        mask_item[n_name][n_id] = 0
                else:
                    assert n_point is None
                    # TODO: general activation
                    raise NotImplementedError
            # invalid scores
            if len(final_decision[b]) == 0: 
                # use random decisions 
                selected = False
                for layer in np.random.choice(split_node_names, len(split_node_names), replace=False):
                    if (len(mask_item[layer].nonzero(as_tuple=False)) != 0) or (split_node_points[layer] is None):
                        if split_node_points[layer] is not None: # relu
                            final_decision[b].append([layer, mask_item[layer].nonzero(as_tuple=False)[0].item(), split_node_points[layer]])
                            mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                        else:
                            # TODO: general activation
                            raise NotImplementedError
                        selected = True
                        break
                assert selected
        
        final_decision = sum(final_decision, [])
        return final_decision


    @beartype
    def naive_input_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', 
                        domain_params: AbstractResults) -> torch.Tensor:
        n_inputs = domain_params.input_uppers.flatten(1).shape[1]
        topk = min(self.decision_topk, n_inputs)
        topk_decisions = torch.topk(domain_params.input_uppers.flatten(1) - domain_params.input_lowers.flatten(1), topk, -1).indices
        if topk == 1:
            return topk_decisions
    
        batch = len(domain_params.input_lowers)
        topk_output_lbs = torch.empty(
            size=(topk, batch), 
            device=domain_params.input_lowers.device, 
            requires_grad=False,
        )
        
        for i in range(topk):
            tmp_decision = topk_decisions[:, i:i+1]
            abs_ret = abstractor._forward_input(
                domain_params=domain_params,
                decisions=tmp_decision,
                simplify=True,
            )
            output_lbs_tmp = (abs_ret.output_lbs - torch.cat([domain_params.rhs, domain_params.rhs])).max(-1).values
            # output_lbs_tmp[torch.logical_and(output_lbs_tmp >= -5, output_lbs_tmp > -10)] = LARGE
            topk_output_lbs[i] = torch.max(output_lbs_tmp[:batch], output_lbs_tmp[batch:])
        
        # print(topk_output_lbs.min(), topk_output_lbs.max())
        # topk_output_lbs[topk_output_lbs >= 0] = -LARGE
        best_output_lbs = torch.topk(topk_output_lbs, 1, 0, largest=True)
        best_indices = best_output_lbs.indices
        # best_values = best_output_lbs.values
        # invalid_indices = torch.where(best_values == -LARGE)
        
        final_decision = torch.tensor([[topk_decisions[i, best_indices[0, i]]] for i in range(batch)])

        return final_decision
    
    
    @beartype
    def smart_input_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', 
                        domain_params: AbstractResults) -> torch.Tensor:
        lAs_factors = domain_params.lAs[abstractor.net.input_name[0]].flatten(2).abs().clamp(min=0.1)
        diff = (domain_params.input_uppers - domain_params.input_lowers).flatten(1).unsqueeze(1)
        objective = (domain_params.output_lbs - domain_params.rhs).unsqueeze(-1)
        score = lAs_factors * diff + objective
        score = score.amax(dim=-2)
        decisions = torch.topk(score, 1, -1).indices
        return decisions


    @beartype
    def naive_hidden_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', domain_params: AbstractResults, mode: str) -> list[list]:
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}
        
        if mode == 'distance':
            scores = {
                k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k]) 
                    for k in domain_params.upper_bounds
            }
        elif mode == 'polarity':
            scores = {
                k: (domain_params.upper_bounds[k] * domain_params.lower_bounds[k]) / (domain_params.lower_bounds[k] - domain_params.upper_bounds[k]) 
                    for k in domain_params.upper_bounds
            }
        elif mode == 'scale':
            scores = {
                k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k]) / torch.abs(domain_params.upper_bounds[k] + domain_params.lower_bounds[k]) 
                    for k in domain_params.upper_bounds
            }
        else:
            raise NotImplementedError()
            
        masked_scores = {k: torch.where(domain_params.masks[k].bool(), scores[k], 0.0) for k in scores}
        
        assert len(abstractor.net.split_nodes) == len(masked_scores)
        best_scores = [masked_scores[k.name].topk(1, 1) for k in abstractor.net.split_nodes]
        best_scores_all_layers = torch.cat([s.values for s in best_scores], dim=1)
        best_scores_all_layers_indices = torch.cat([s.indices for s in best_scores], dim=1).detach().cpu().numpy()
        best_scores_all = best_scores_all_layers.topk(1, 1)
        assert (best_scores_all.values > 0.0).all()
        
        layer_ids = best_scores_all.indices[:, 0].detach().cpu().numpy()
        assert len(layer_ids) == batch
        decisions = [[
            split_node_names[layer_ids[b]], 
            best_scores_all_layers_indices[b, layer_ids[b]], 
            split_node_points[split_node_names[layer_ids[b]]]
        ] for b in range(batch)]
        
        return decisions

        