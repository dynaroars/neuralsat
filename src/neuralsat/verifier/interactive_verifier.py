from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import torch
import copy

from ..heuristic.decision_heuristics import DecisionHeuristic
from ..auto_LiRPA.utils import stop_criterion_batch_any
from ..heuristic.domains_list import DomainsList
from ..util.misc.result import AbstractResults
from ..abstractor.utils import new_slopes


class InteractiveVerifier:

    "Branch-and-Bound Interactive Verifier"

    # @beartype
    def __init__(self: 'InteractiveVerifier', net: torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device

        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)

        self.scorer = DecisionHeuristic(
            input_split=self.input_split,
            decision_topk=-1,
            decision_method='greedy'
        )


    # @beartype
    def _initialize(self: 'InteractiveVerifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        # initialization params
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)

        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []

        # full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)

        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
            objective_ids=ret.objective_ids,
            output_lbs=ret.output_lbs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds,
            upper_bounds=ret.upper_bounds,
            lAs=ret.lAs,
            slopes=slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )


    def decide(self, observation, domain_params):
        decisions = self.decision(self.abstractor, domain_params)
        return (decisions, domain_params), None


    def init(self: 'InteractiveVerifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        self._setup_restart(0, objective)
        self.domains_list = self._initialize(
            objective=objective,
            preconditions=preconditions,
            reference_bounds=reference_bounds,
        )
        return len(self.domains_list) == 0


    def get_observation(self, batch):
        pick_ret = self.domains_list.pick_out(batch, self.device)

        obs = self.scorer.get_branching_scores(
            abstractor=self.abstractor,
            domain_params=pick_ret,
        )
        # topk_output_lbs: (topk, batch)
        # topk_decisions: (topk, batch)
        return obs, pick_ret

    def get_rewards(self, pick_ret, reduce_op=torch.max):
        all_output_lbs, all_decisions = self.scorer.get_all_branching_rewards(
            abstractor=self.abstractor,
            domain_params=pick_ret,
            reduce_op=reduce_op,
        )
        return all_output_lbs, all_decisions

    def step(self, action):
        decisions, pick_ret = action
        abstraction_ret = self.abstractor.forward(decisions, pick_ret)
        self.domains_list.add(abstraction_ret, decisions)
        done = len(self.domains_list) == 0

        # reward = self.domains_list.minimum_lowers
        # given action (selected neuron)
        # return minimum lowerbound on two branches of that neuron
        reward = abstraction_ret.output_lbs.min(dim=-1).values
        # print(reward)
        r1, r2 = torch.chunk(reward, 2)
        reward, reward_indices = torch.min(torch.stack((r1, r2)), dim=0)
        assert not reward.isnan().any()
        # DEBUG
        # reward, reward_indices = torch.max(torch.stack((r1, r2)), dim=0)
        #
        split_observation = self.scorer.get_branching_scores(abstractor=self.abstractor,
                                                             domain_params=abstraction_ret)

        assert all([not _.isnan().any() for _ in split_observation[0]])
        
        info = {
            'worst_bound': self.domains_list.minimum_lowers,
            'visited': self.domains_list.visited,
            'remaining': len(self.domains_list),
            'pick_ret': pick_ret, # subproblem before
            'abstraction_ret': abstraction_ret, # subproblem after
            'split_observation': split_observation,
            'reward_indices': reward_indices,
        }
        return reward, done, info
    
    def topk_action(self, scores, domain_params, topk=1):
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in self.abstractor.net.split_nodes]
        split_node_points = {k: self.abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}
        
        # print(f'{batch=} {topk=}')
        # print([(k, v.shape) for k, v in scores.items()])
        # print(f'{split_node_points=}')
        
        score_length = np.insert(np.cumsum([len(scores[i][0]) for i in range(len(scores))]), 0, 0)
        # print(f'{score_length=}')
        
        topk_scores = torch.topk(torch.cat(scores, dim=1), topk)
        
        topk_output_lbs, topk_decisions = self.get_topk_scores(
            domain_params=domain_params,
            topk_scores=topk_scores,
            score_length=score_length,
            topk=topk,
        )
        
        # print(f'{topk_output_lbs=}')
        # print(f'{topk_decisions=}')
        
        best = topk_output_lbs.topk(1, 0)
        best_output_lbs = best.values.cpu().numpy()[0]
        best_output_lbs_indices = best.indices.cpu().numpy()[0]
        
        all_topk_decisions = [topk_decisions[best_output_lbs_indices[ii]][ii] for ii in range(batch)]
        # print(f'{all_topk_decisions=}')
        final_decision = [[] for b in range(batch)]
        
        for b in range(batch):
            mask_item = {k: domain_params.masks[k][b].clone() for k in split_node_names}
            assert best_output_lbs[b] > -1e6, f'{best_output_lbs[b]=}'
            # valid scores
            n_name, n_id, n_point = all_topk_decisions[b]
            if n_point is not None: # relu
                if mask_item[n_name][n_id]: # unstable relu
                    final_decision[b].append([n_name, n_id, n_point])
                    mask_item[n_name][n_id] = 0
            else:
                raise NotImplementedError
            # invalid scores
            if len(final_decision[b]) == 0:
                selected = False
                for layer in np.random.choice(split_node_names, len(split_node_names), replace=False):
                    if (len(mask_item[layer].nonzero(as_tuple=False)) != 0) or (split_node_points[layer] is None):
                        if split_node_points[layer] is not None: # relu
                            final_decision[b].append([layer, mask_item[layer].nonzero(as_tuple=False)[0].item().int(), split_node_points[layer]])
                            mask_item[final_decision[b][-1][0]][final_decision[b][-1][1]] = 0
                        else:
                            # TODO: general activation
                            raise NotImplementedError
                        selected = True
                        break
                assert selected
                
        final_decision = sum(final_decision, [])
        # print(f'{final_decision=}')
        return final_decision
        
        
    def get_topk_scores(self: 'DecisionHeuristic', domain_params: AbstractResults,
                        topk_scores: torch.return_types.topk, score_length: np.ndarray,
                        topk: int, reduce_op=torch.max) -> tuple[torch.Tensor, list]:

        topk_decisions = []
        batch = len(domain_params.input_lowers)
        topk_output_lbs = torch.empty(
            size=(topk, batch),
            device=domain_params.input_lowers.device,
            requires_grad=False,
        )
        topk_scores_indices = topk_scores.indices.cpu()

        for k in range(topk):
            # top-k candidates from scores
            decision_max = [] # higher is better
            for idx in topk_scores_indices[:, k]:
                idx = idx.item()
                layer_idx = np.searchsorted(score_length, idx, side='right') - 1
                layer_name = self.abstractor.net.split_nodes[layer_idx].name
                layer_split_point = self.abstractor.net.split_activations[layer_name][0][0].get_split_point()
                neuron_idx = int(idx - score_length[layer_idx])
                if layer_split_point is not None: # relu
                    decision_max.append([layer_name, neuron_idx, layer_split_point])
                else: # general activation
                    raise NotImplementedError

            # top-k candidates
            topk_decisions.append(decision_max)

            k_domain_params = AbstractResults(**{
                'input_lowers': domain_params.input_lowers,
                'input_uppers': domain_params.input_uppers,
                'lower_bounds': domain_params.lower_bounds,
                'upper_bounds': domain_params.upper_bounds,
                'slopes': domain_params.slopes if k == 0 else [],
                'cs': domain_params.cs,
                'rhs': domain_params.rhs,
            })

            abs_ret = self.abstractor._forward_hidden(
                domain_params=k_domain_params,
                decisions=topk_decisions[-1],
                simplify=True
            )
            # improvements over specification
            k_output_lbs = (abs_ret.output_lbs - torch.cat([domain_params.rhs, domain_params.rhs])).max(-1).values

            # invalid scores for stable neurons
            topk_output_lbs[k] = reduce_op(k_output_lbs.flatten().reshape(2, -1), dim=0).values
            # print(f'{k=} {topk_output_lbs.shape=} {k_output_lbs.shape=}')
            # print(f'{k=} {k_output_lbs=} {topk_output_lbs[k]=}')

        return topk_output_lbs, topk_decisions

    from .utils import _preprocess, _init_abstractor, _setup_restart
