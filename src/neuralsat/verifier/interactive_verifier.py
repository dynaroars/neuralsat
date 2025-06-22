from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
from beartype import beartype
import torch
import copy

from ..util.network.onnx2networkx import prepare_graph, get_edge_weight, get_edge_index
from ..auto_LiRPA.utils import stop_criterion_batch_any
from ..heuristic.domains_list import DomainsList
from ..util.misc.result import AbstractResults
from ..heuristic.util import compute_masks
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

    @beartype
    def get_edge_data(self, objective) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(objective.lower_bounds) == 1
        nx_graph = prepare_graph(self.net, self.input_shape, objective)
        edge_weight = get_edge_weight(nx_graph)
        edge_index = get_edge_index(nx_graph)
        return edge_weight, edge_index
        
    @beartype
    def gather_feature(self, domain_params: AbstractResults) -> list[torch.Tensor]:
        assert len(domain_params.input_lowers) == len(domain_params.input_uppers) == 1
        assert all([len(v) == 1 for v in domain_params.lower_bounds.values()])
        assert all([len(v) == 1 for v in domain_params.upper_bounds.values()])
        
        ret = []
        # 1. input features
        lb = domain_params.input_lowers.flatten(1).cpu()
        ub = domain_params.input_uppers.flatten(1).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu()
        x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
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
                bs, c, h, w = lb.shape
                bias = bias[None, : , None, None]
                bias = bias.repeat(bs, 1, h, w)
            elif lb.ndim == 2: # fc layer
                bias = bias[None]
            else:
                raise NotImplementedError
            mask = masks[name].view(lb.shape)
            assert (lb[torch.where(mask == 1)] < 0).all()
            assert (ub[torch.where(mask == 1)] > 0).all()
            x = torch.stack([
                lb.flatten(1),
                ub.flatten(1),
                bias.flatten(1),
                mask.flatten(1),
            ], dim=-1)[0]
            ret.append(x)
            # print(f'hidden: {name=} {x.shape=}')
        
        # 3. output features
        lb = domain_params.output_lbs.flatten(1).cpu()
        ub = torch.zeros_like(lb).cpu()
        bias = torch.zeros_like(lb).cpu()
        mask = torch.zeros_like(lb).cpu()
        x = torch.stack([lb, ub, bias, mask], dim=-1)[0]
        # print(f'output: {name=} {x.shape=}')
        ret.append(x)
        return ret
    
    @beartype
    def get_node_data(self, objective) -> list[torch.Tensor]:
        assert len(objective.lower_bounds) == 1
        self._setup_restart(0, objective)
        sample = self.abstractor.initialize(objective, reference_bounds=None)
        return self.gather_feature(sample)

    @beartype
    def _initialize(self, objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
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
            slopes=ret.slopes if Settings.update_interm_bounds else slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )

    @beartype
    def setup(self, objective) -> bool:
        self._setup_restart(0, objective)
        self.domains_list = self._initialize(
            objective=objective,
            preconditions={},
            reference_bounds=None,
        )
        return len(self.domains_list) == 0

    @beartype
    def step(self, action):
        raise NotImplementedError
    
    
    from .utils import _preprocess, _setup_restart, _init_abstractor
