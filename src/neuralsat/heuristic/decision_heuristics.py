from __future__ import annotations

from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from beartype import beartype
import numpy as np
import typing
import random
import torch
import os

if typing.TYPE_CHECKING:
    from .. import abstractor

from ..heuristic.util import _compute_babsr_scores, compute_masks
from ..util.misc.result import AbstractResults
from ..setting import Settings

LARGE = 1e6
SMALL = 1.0 / LARGE

class DecisionHeuristic:

    # @beartype
    def __init__(self, input_split: bool, decision_topk: int, decision_method: str) -> None:
        self.input_split = input_split
        self.decision_topk = decision_topk
        self.decision_method = decision_method
        self.decision_reduceop = torch.max


    # @beartype
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

        if None in abstractor.split_points:
            # FIXME: generalize it with relu
            # handle general activation
            return self.naive_hidden_branching(
                domain_params=domain_params,
                abstractor=abstractor,
                mode=random.choice(['width', 'distance'][1:]),
            )

        # hidden split
        if self.decision_method == 'greedy':
            return self.brute_force_hidden_branching(
                abstractor=abstractor,
                domain_params=domain_params,
            )

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


    # @beartype
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

        if os.environ.get('NEURALSAT_ASSERT'):
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

            # print(f'{decision_max=} {len(decision_max)=}')

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


    # @beartype
    def get_topk_scores_greedy(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', domain_params: AbstractResults,
                        topk_scores: torch.return_types.topk, score_length: np.ndarray, topk: int, reduce_op=torch.max) -> tuple[torch.Tensor, list]:
        topk_decisions = []
        batch = len(domain_params.input_lowers)
        topk_output_lbs = torch.empty(
            size=(topk, batch),
            device=domain_params.input_lowers.device,
            requires_grad=False,
        )

        # # hidden
        # double_lower_bounds = {k: torch.cat([v, v]) for k, v in domain_params.lower_bounds.items()}
        # double_upper_bounds = {k: torch.cat([v, v]) for k, v in domain_params.upper_bounds.items()}

        # # slope
        # double_slopes = defaultdict(dict)
        # for k, v in domain_params.slopes.items():
        #     double_slopes[k] = {kk: torch.cat([vv, vv], dim=2) for (kk, vv) in v.items()}

        # # spec
        # double_cs = torch.cat([domain_params.cs, domain_params.cs])
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs])

        # # input
        # double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers])
        # double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers])

        # if os.environ.get('NEURALSAT_ASSERT'):
        #     assert torch.all(double_input_lowers <= double_input_uppers)

        topk_scores_indices = topk_scores.indices.cpu()
        topk_scores_values = topk_scores.values.cpu()

        for k in range(topk):
            # top-k candidates from scores
            decision_candidates = [] # higher is better
            for idx, value in zip(topk_scores_indices[:, k], topk_scores_values[:, k]):
                if value == 0.0:
                    decision_candidates.append(topk_decisions[0][0])
                    continue
                idx = idx.item()
                layer_idx = np.searchsorted(score_length, idx, side='right') - 1
                layer_name = abstractor.net.split_nodes[layer_idx].name
                layer_split_point = abstractor.net.split_activations[layer_name][0][0].get_split_point()
                neuron_idx = idx - score_length[layer_idx]
                if layer_split_point is not None: # relu
                    decision_candidates.append([layer_name, neuron_idx, layer_split_point])
                else: # general activation
                    raise NotImplementedError

            # print(f'{decision_candidates=} {len(decision_candidates)=}')

            # top-k candidates
            topk_decisions.append(decision_candidates)

            k_domain_params = AbstractResults(**{
                'input_lowers': domain_params.input_lowers,
                'input_uppers': domain_params.input_uppers,
                'lower_bounds': domain_params.lower_bounds,
                'upper_bounds': domain_params.upper_bounds,
                'slopes': domain_params.slopes if k == 0 else [],
                'cs': domain_params.cs,
                'rhs': domain_params.rhs,
            })

            abs_ret = abstractor._forward_hidden(
                domain_params=k_domain_params,
                decisions=topk_decisions[-1],
                simplify=True
            )
            # improvements over specification
            k_output_lbs = (abs_ret.output_lbs - double_rhs).max(-1).values

            # invalid scores for stable neurons
            # invalid_mask_scores = (topk_scores.values[:, k] <= SMALL).to(torch.get_default_dtype())
            # print(f'{invalid_mask_scores.sum()=}')
            topk_output_lbs[k] = reduce_op((k_output_lbs.flatten()).reshape(2, -1), dim=0).values

        return topk_output_lbs, topk_decisions


    # hidden branching
    # @beartype
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


    # @beartype
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


    # @beartype
    def smart_input_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor',
                        domain_params: AbstractResults) -> torch.Tensor:
        lAs_factors = domain_params.lAs[abstractor.net.input_name[0]].flatten(2).abs().clamp(min=0.1)
        diff = (domain_params.input_uppers - domain_params.input_lowers).flatten(1).unsqueeze(1)
        objective = (domain_params.output_lbs - domain_params.rhs).unsqueeze(-1)
        score = lAs_factors * diff + objective
        score = score.amax(dim=-2)
        decisions = torch.topk(score, 1, -1).indices
        return decisions


    # @beartype
    def naive_hidden_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor', domain_params: AbstractResults, mode: str) -> list[list]:
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}

        if mode == 'distance':
            scores = {
                k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k])
                    for k in split_node_points
            }
        elif mode == 'polarity':
            scores = {
                k: (domain_params.upper_bounds[k] * domain_params.lower_bounds[k]) / (domain_params.lower_bounds[k] - domain_params.upper_bounds[k])
                    for k in split_node_points
            }
        elif mode == 'scale':
            scores = {
                k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k]) / torch.abs(domain_params.upper_bounds[k] + domain_params.lower_bounds[k])
                    for k in split_node_points
            }
        elif mode == 'width':
            scores = {
                k: torch.abs(domain_params.upper_bounds[k] - domain_params.lower_bounds[k])
                    for k in split_node_points
            }
        else:
            raise NotImplementedError()

        masks = {
            k: domain_params.masks[k] if (split_node_points[k] is not None) else torch.ones_like(domain_params.masks[k])
                for k in split_node_points
        }

        masked_scores = {
            k: torch.where(masks[k].bool(), scores[k].flatten(1), 0.0)
                for k in split_node_points
        }

        # TODO: not always required to compute
        decision_points = {
            k: (domain_params.upper_bounds[k] + domain_params.lower_bounds[k]) / 2.0
                for k in split_node_points
        }
        assert len(abstractor.net.split_nodes) == len(masked_scores)
        best_scores = [masked_scores[k].topk(1, 1) for k in split_node_points]
        best_scores_all_layers = torch.cat([s.values for s in best_scores], dim=1)
        best_scores_all_layers_indices = torch.cat([s.indices for s in best_scores], dim=1).detach().cpu().numpy()
        best_scores_all = best_scores_all_layers.topk(1, 1)
        assert (best_scores_all.values > 0.0).all()

        layer_ids = best_scores_all.indices[:, 0].detach().cpu().numpy()
        assert len(layer_ids) == batch
        decisions = []
        for b in range(batch):
            l_name = split_node_names[layer_ids[b]]
            n_id = best_scores_all_layers_indices[b, layer_ids[b]]
            if split_node_points[l_name] is not None:
                point = split_node_points[l_name]
            else:
                point = decision_points[l_name][b].flatten()[n_id].item()
            decisions.append([l_name, n_id, point])

        return decisions


    def get_all_branching_rewards(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor',
                                 domain_params: AbstractResults, reduce_op=torch.max) -> list[list]:
        # print('get_all_branching_rewards')
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}

        masks = {
            k: domain_params.masks[k] if (split_node_points[k] is not None) else torch.ones_like(domain_params.masks[k])
                for k in split_node_points
        }

        # print(f'{masks=}')
        n_unstables = sum([v.flatten(1).sum(1) for v in masks.values()])
        assert len(n_unstables) == batch
        topk = int(n_unstables.amax().item())
        topk = topk if self.decision_topk == -1 else max(topk, self.decision_topk)
        # print(f'{topk=} {self.decision_topk=}')
        # print(f'{split_node_names=}')
        # topk = 10

        raw_scores = {
            k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k])
                for k in split_node_points
        }

        masked_scores = {
            k: masks[k] * raw_scores[k].flatten(1)
                for k in split_node_points
        }

        # print([s.shape for s in masked_scores.values()])
        scores = [masked_scores[name] for name in split_node_names]

        # convert an index to its layer and offset
        score_length = np.insert(np.cumsum([len(scores[i][0]) for i in range(len(scores))]), 0, 0)

        # top-k candidates
        topk_scores = torch.topk(torch.cat(scores, dim=1), topk)

        topk_output_lbs, topk_decisions = self.get_topk_scores_greedy(
            abstractor=abstractor,
            domain_params=domain_params,
            topk_scores=topk_scores,
            score_length=score_length,
            topk=topk,
            reduce_op=reduce_op,
        )
        # print(f'{batch=}')
        # print(len(topk_output_lbs))
        # print(topk_output_lbs)
        # print(len(topk_decisions))
        # print(topk_decisions)
        return topk_output_lbs, topk_decisions

    # hidden branching
    # @beartype
    def brute_force_hidden_branching(self: 'DecisionHeuristic', abstractor: 'abstractor.abstractor.NetworkAbstractor',
                                 domain_params: AbstractResults) -> list[list]:
        print('brute_force_hidden_branching')
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in abstractor.net.split_nodes]
        split_node_points = {k: abstractor.net.split_activations[k][0][0].get_split_point() for k in split_node_names}

        masks = {
            k: domain_params.masks[k] if (split_node_points[k] is not None) else torch.ones_like(domain_params.masks[k])
                for k in split_node_points
        }

        # print(f'{masks=}')
        n_unstables = sum([v.flatten(1).sum(1) for v in masks.values()])
        assert len(n_unstables) == batch
        topk = int(n_unstables.amin().item())
        topk = topk if self.decision_topk == -1 else max(topk, self.decision_topk)
        # print(f'{topk=} {self.decision_topk=}')
        # print(f'{split_node_names=}')
        topk = 50

        if 1:
            raw_scores = {
                k: torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k])
                    for k in split_node_points
            }

            masked_scores = {
                k: torch.where(masks[k].bool(), raw_scores[k].flatten(1), 0.0)
                    for k in split_node_points
            }

            # print([s.shape for s in masked_scores.values()])
            scores = [masked_scores[name] for name in split_node_names]
        else:

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
        # topk_backup_scores = torch.topk(torch.cat(backup_scores, dim=1), topk, largest=False)
        # topk_scores_indices = topk_scores.indices.cpu()
        # print(topk_scores_indices.shape)


        topk_output_lbs, topk_decisions = self.get_topk_scores_greedy(
            abstractor=abstractor,
            domain_params=domain_params,
            topk_scores=topk_scores,
            score_length=score_length,
            topk=topk,
        )

        # print(f'{topk_decisions=}')
        # print(f'{topk_output_lbs=}')

        best = topk_output_lbs.topk(1, 0)
        best_output_lbs_indices = best.indices.cpu().numpy()[0]
        all_topk_decisions = [topk_decisions[best_output_lbs_indices[b]][b] for b in range(batch)]
        final_decisions = [[] for b in range(batch)]

        for b in range(batch):
            mask_item = {k: domain_params.masks[k][b].clone() for k in split_node_names}
            n_name, n_id, n_point = all_topk_decisions[b]
            if n_point is not None: # relu
                assert mask_item[n_name][n_id] != 0 # unstable relu
                final_decisions[b].append([n_name, n_id, n_point])
                mask_item[n_name][n_id] = 0
            else:
                # TODO: general activation
                raise NotImplementedError


        final_decisions = sum(final_decisions, [])
        return final_decisions



    def get_branching_scores(self, abstractor, domain_params) -> list[list]:
        device = abstractor.device
        batch = len(domain_params.input_lowers)
        split_node_names = [_.name for _ in abstractor.net.split_nodes]

        masks = compute_masks(
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            device=device,
            non_blocking=False,
        )

        # features
        scores_1, scores_2 = _compute_babsr_scores(
            abstractor=abstractor,
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            lAs=domain_params.lAs,
            batch=batch,
            masks=masks,
            reduce_op=self.decision_reduceop,
            number_bounds=domain_params.cs.shape[1]
        )
        scores_1 = {split_node_names[i]: scores_1[i] for i in range(len(scores_1))}
        assert all([not _.isnan().any() for _ in scores_1.values()])
        
        scores_2 = {split_node_names[i]: scores_2[i] for i in range(len(scores_2))}
        assert all([not _.isnan().any() for _ in scores_2.values()])
        
        scores_3 = {k: (torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k])) for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_3.values()])
        
        scores_4 = {k: ((domain_params.upper_bounds[k] * domain_params.lower_bounds[k]) / (domain_params.lower_bounds[k] - domain_params.upper_bounds[k])) for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_4.values()])
        
        scores_5 = {k: (torch.min(domain_params.upper_bounds[k], -domain_params.lower_bounds[k]) / torch.abs(domain_params.upper_bounds[k] + domain_params.lower_bounds[k])) for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_5.values()])
        
        scores_6 = {k: (torch.abs(domain_params.upper_bounds[k] - domain_params.lower_bounds[k])) for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_6.values()])
        
        scores_7 = {k: domain_params.upper_bounds[k] for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_7.values()])

        scores_8 = {k: domain_params.lower_bounds[k] for k in split_node_names}
        assert all([not _.isnan().any() for _ in scores_8.values()])
        
        # scores_1 = {k: scores_1[k].flatten(1)[masks[k].bool()] for k in split_node_names}
        # print('score:', [scores_1[k].shape for k in split_node_names])
        
        scores_1, nonzero_masks = masked_scores(scores_1, masks, return_nonzero_masks=True)
        scores_2                = masked_scores(scores_2, masks)
        scores_3                = masked_scores(scores_3, masks)
        scores_4                = masked_scores(scores_4, masks)
        scores_5                = masked_scores(scores_5, masks)
        scores_6                = masked_scores(scores_6, masks)
        scores_7                = masked_scores(scores_7, masks)
        scores_8                = masked_scores(scores_8, masks)
        # print('score:', [scores_1[k].shape for k in split_node_names])
        
        scores_all_dict = {
            k: torch.stack([
                scores_1[k],
                scores_2[k],
                scores_3[k],
                scores_4[k],
                scores_5[k],
                scores_6[k],
                scores_7[k],
                scores_8[k],
            ], dim=-1)
            for k in split_node_names
        }

        scores_all_list = [scores_all_dict[k] for k in split_node_names]
        masks_all_list  = [nonzero_masks[k]   for k in split_node_names]
        
        print('bound:', [domain_params.lower_bounds[k].shape for k in split_node_names])
        print('mask :', [masks[k].shape for k in split_node_names])
        print('score:', [scores_1[k].shape for k in split_node_names])
        
        print('return masks: ', [_.shape for _ in masks_all_list])
        print('return scores:', [_.shape for _ in scores_all_list])
        print(masks_all_list[-1])
        
        return scores_all_list, masks_all_list

def masked_scores(scores_dict, masks_dict, return_nonzero_masks=False):
    returned_scores = {}
    returned_masks = {}
    for name in scores_dict:
        flat_scores = scores_dict[name].flatten(1)
        batch = flat_scores.size(0)
        
        masked_batch_scores = [flat_scores[i][masks_dict[name][i].bool()] for i in range(batch)]
        # print([_.shape for _ in masked_batch_scores])
        masked_batch_scores = pad_sequence(masked_batch_scores, batch_first=True)
        returned_scores[name] = masked_batch_scores
        if return_nonzero_masks:
            nonzero_batch_masks = [masks_dict[name][i].nonzero().squeeze(-1) for i in range(batch)]
            # print([_.shape for _ in nonzero_batch_masks])
            nonzero_batch_masks = pad_sequence(nonzero_batch_masks, batch_first=True)
            returned_masks[name] = nonzero_batch_masks
    if return_nonzero_masks:
        return returned_scores, returned_masks
    return returned_scores
    