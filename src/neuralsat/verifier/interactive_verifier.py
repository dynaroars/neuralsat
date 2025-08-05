from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
from beartype import beartype
import numpy as np
import torch
import copy

from ..util.network.onnx2networkx import prepare_graph, get_edge_weight, get_edge_index
from ..heuristic.util import compute_masks, _compute_babsr_scores
from ..heuristic.decision_heuristics import DecisionHeuristic
from ..auto_LiRPA.utils import stop_criterion_batch_any
from ..heuristic.domains_list import DomainsList
from ..util.misc.result import AbstractResults
from ..abstractor.utils import new_slopes
from ..setting import Settings

class InteractiveVerifier:

    "Interactive Branch-and-Bound Verifier"

    @beartype
    def __init__(self, net: torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device

        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)
        
        self.fsb_heuristic = DecisionHeuristic(
            input_split=False, 
            decision_method='smart',
            decision_topk=10, 
        )
        
        
    @beartype
    def _compute_neuron_index_mapping(self, domain_params: AbstractResults) -> None:
        self.neuron_index_mapping = {}
        ordered_names = [_.name for _ in self.abstractor.net.split_nodes]
        count = np.prod(self.input_shape)
        for name in ordered_names:
            assert domain_params.lower_bounds[name].shape[0] == 1
            n_hiddens = domain_params.lower_bounds[name].numel()
            self.neuron_index_mapping.update(
                {int(count + i): (name, i, 0.0) for i in range(n_hiddens)}
            )
            count += n_hiddens
        
        self.reverse_neuron_index_mapping = {v: k for k, v in self.neuron_index_mapping.items()}

    @beartype
    def get_edge_data(self, objective) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(objective.lower_bounds) == 1
        nx_graph = prepare_graph(self.net, self.input_shape, objective)
        edge_weight = get_edge_weight(nx_graph)
        edge_index = get_edge_index(nx_graph)
        return edge_weight, edge_index
        
    @beartype
    def get_node_data(self, domain_params: AbstractResults) -> list[torch.Tensor]:
        assert len(domain_params.input_lowers) == len(domain_params.input_uppers)
        # assert all([len(v) == 1 for v in domain_params.lower_bounds.values()])
        # assert all([len(v) == 1 for v in domain_params.upper_bounds.values()])
        
        ret = []
        # 1. input features
        lb = domain_params.input_lowers.flatten(1).cpu()
        ub = domain_params.input_uppers.flatten(1).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.ones_like(lb).cpu() # convert to heursitic form: 0 - not selected, 1 - selected
        x = torch.stack([lb, ub, bias, mask], dim=-1)
        # print(f'input: {x.shape=}')
        ret.append(x)
        
        # 2. hidden features
        ordered_names = [_.name for _ in self.abstractor.net.split_nodes]
        # extract mask
        masks = compute_masks(domain_params.lower_bounds, domain_params.upper_bounds, device='cpu')
        # extract hidden
        for name in ordered_names:
            lb = domain_params.lower_bounds[name].cpu()
            ub = domain_params.upper_bounds[name].cpu()
            bias = self.abstractor.net[name].inputs[-1].param.detach().cpu()
            if lb.ndim == 4: # conv layer
                bs, _, h, w = lb.shape
                bias = bias[None, : , None, None]
                bias = bias.repeat(bs, 1, h, w)
            elif lb.ndim == 2: # fc layer
                bs, _ = lb.shape
                bias = bias[None].repeat(bs, 1)
            else:
                raise NotImplementedError
            mask = masks[name].view(lb.shape) 
            assert (lb[torch.where(mask == 1)] < 0).all()
            assert (ub[torch.where(mask == 1)] > 0).all()
            x = torch.stack([
                lb.flatten(1),
                ub.flatten(1),
                bias.flatten(1),
                1 - mask.flatten(1), # convert to heursitic form: 0 - not selected, 1 - selected
            ], dim=-1)
            ret.append(x)
            # print(f'hidden: {name=} {x.shape=}')
        
        # 3. output features
        lb = domain_params.output_lbs.flatten(1).cpu()
        ub = torch.zeros_like(lb).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.ones_like(lb).cpu() # convert to heursitic form: 0 - not selected, 1 - selected
        x = torch.stack([lb, ub, bias, mask], dim=-1)
        # print(f'output: {name=} {x.shape=}')
        ret.append(x)
        return ret
    
    def get_fsb_score(self, domain_params: AbstractResults) -> torch.Tensor:
    
        masks = compute_masks(
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            device=self.device,
            non_blocking=False,
        )

        # features
        batch = len(domain_params.input_lowers)
        scores_1, scores_2 = _compute_babsr_scores(
            abstractor=self.abstractor,
            lower_bounds=domain_params.lower_bounds,
            upper_bounds=domain_params.upper_bounds,
            lAs=domain_params.lAs,
            batch=batch,
            masks=masks,
            reduce_op=self.fsb_heuristic.decision_reduceop,
            number_bounds=domain_params.cs.shape[1]
        )
        # print([_.shape for _ in scores_1])
        # print([_.shape for _ in scores_2])
        dummy_input = torch.zeros_like(domain_params.input_lowers, device=self.device).flatten(1)
        dummy_output = torch.zeros_like(domain_params.output_lbs, device=self.device).flatten(1)
        score_1 = torch.cat([dummy_input] + scores_1 + [dummy_output], dim=-1)
        score_2 = torch.cat([dummy_input] + scores_2 + [dummy_output], dim=-1)
        score = torch.stack([score_1, score_2], dim=-1)
        return score
    
    # @beartype
    def get_initial_node_data(self, objective, return_fsb_score=False):
        assert len(objective.lower_bounds) == 1, f'{len(objective.lower_bounds)=}'
        self._setup_restart(0, objective)
        sample = self.abstractor.initialize(objective, reference_bounds=None)
        # print(f'{sample.output_lbs=}')
        if sample.input_lowers is None:
            return None
        if return_fsb_score:
            score = self.get_fsb_score(sample)
            return self.get_node_data(sample), score
        return self.get_node_data(sample)

    @beartype
    def _initialize(self, objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)
        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []
        
        # compute neuron index mapping
        self._compute_neuron_index_mapping(ret)
        
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
            slopes=ret.slopes if Settings.update_interm_bounds else slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )

    @beartype
    def setup(self, objective) -> None | AbstractResults:
        self._setup_restart(0, objective)
        self.domains_list = self._initialize(
            objective=objective,
            preconditions={},
            reference_bounds=None,
        )
        if not len(self.domains_list):
            return None
        
        subproblems = self.domains_list.pick_out(len(self.domains_list), device=self.device)
        return subproblems

    @beartype
    def step(self, domains_params: AbstractResults, action: list[int]):
        batch = len(domains_params.input_lowers)
        assert len(action) == batch, f'{len(action)=} {batch=}'
        decisions = [self.neuron_index_mapping[_] for _ in action]
        abstraction_ret = self.abstractor.forward(decisions, domains_params)
        remaining_index = self.domains_list.add(abstraction_ret, decisions)
        if len(remaining_index) == 0:
            return None, torch.tensor([0.0]), [], []
        subproblems = self.domains_list.pick_out(len(self.domains_list), device=self.device)
        rewards = subproblems.output_lbs
        next_features = self.get_node_data(subproblems)
        subproblems = subproblems._replace(last_decisions=decisions * 2)
        assert len(subproblems.input_lowers) == len(remaining_index), f'{len(subproblems.input_lowers)=} {len(remaining_index)=}'
        return subproblems, rewards, next_features, remaining_index
    
    def get_last_actions(self, domains_params: AbstractResults) -> list[int]:
        last_actions = [
            self.reverse_neuron_index_mapping[(_[0], _[1], _[2])] for _ in domains_params.last_decisions
        ]
        return last_actions
    
    def get_fsb_action(self, domains_params: AbstractResults) -> list[int]:
        decisions = self.fsb_heuristic(self.abstractor, domains_params)
        actions = [
            self.reverse_neuron_index_mapping[(_[0], _[1], _[2])] for _ in decisions
        ]
        return actions
    
    from .utils import _preprocess, _setup_restart, _init_abstractor
