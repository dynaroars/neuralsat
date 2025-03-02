from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
from beartype import beartype
import torch
import copy

from ..heuristic.decision_heuristics import DecisionHeuristic
from ..auto_LiRPA.utils import stop_criterion_batch_any
from ..onnx2pytorch.convert.model import ConvertModel
from ..heuristic.domains_list import DomainsList
from ..abstractor.utils import new_slopes


class InteractiveVerifier:

    "Branch-and-Bound Interactive Verifier"

    # @beartype
    def __init__(self: 'InteractiveVerifier', net: ConvertModel | torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
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
        topk_output_lbs, topk_decisions = self.scorer.get_all_branching_rewards(
            abstractor=self.abstractor,
            domain_params=pick_ret,
            reduce_op=reduce_op,
        )
        return topk_output_lbs, topk_decisions

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

    from .utils import _preprocess, _init_abstractor, _setup_restart




    '''
    - [ ] Input: n subproblem step t
    - [ ] Output: 2n subproblem at most

    '''











    # comment
