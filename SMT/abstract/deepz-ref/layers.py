import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)


class Linear(nn.Module):
    """
    Modified Linear layer such that bias is added only to the first sample in batch.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.weight = layer.weight.data.T
        self.bias = layer.bias.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.mm(x, self.weight)
        y[0] += self.bias
        return y

class ReLU(nn.Module):
    """
    Modified ReLU that tries to approximate the Zonotope region.
    Adds a new sample to batch sample, if new eps term is added to system.
    Idea:
        a) `slope` is an optional attribute. In case no crossing takes place, given ReLU is irrelevant & thus it's
           `slope` is not optimized.
        b) During the forward pass, if any crossing takes place, `slope` is intialized as `Variable`, such that
           gradients are calculated with respect to it. See `leastArea()` for implementation.
        c) `lower_bound` and `upper_bound` define the state of ReLU and hence used as attributes.
        d) `intercept` for ReLU is a function of `slope`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_magnitude = torch.abs(x[1:]).sum(dim=0)
        self.lower_bound, self.upper_bound = (x[0] - noise_magnitude, x[0] + noise_magnitude)

        neg_mask, pos_mask = self.upper_bound <= 0, self.lower_bound >= 0
        y = x * (~neg_mask).float()  # Setting all values 0 if upper_bound is non-positive.
        mask = ~(neg_mask + pos_mask)  # Mask is 1 where crossing takes place.

        return self.convexApprox(y, mask.float()) if torch.any(mask) else y

    def convexApprox(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "slope"):
            self.leastArea()
        return self.addNewEpsilon(x, mask)

    def addNewEpsilon(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.slope.data.clamp_(min=0, max=1)  # Requirement!

        x = self.slope * x * mask + (1 - mask) * x
        new_eps_term = (torch.ones_like(x[0]) * self.intercept * 0.5) * mask
        x[0] += new_eps_term

        indexes = torch.where(new_eps_term != 0)
        output_dim, num_activations = [1] * len(indexes), len(indexes[0])
        new_eps_terms = torch.zeros_like(new_eps_term).repeat(num_activations, *output_dim)
        indexes = (torch.arange(num_activations), *indexes)
        new_eps_terms[indexes] = 1
        new_eps_terms *= new_eps_term
        return torch.cat([x, new_eps_terms], dim=0)

    def leastArea(self):
        self.slope = Variable(torch.clamp(self.slope_threshold, 0, 1), requires_grad=True)
        self.slope.retain_grad()

    @property
    def slope_threshold(self) -> torch.Tensor:
        return self.upper_bound / (self.upper_bound - self.lower_bound)

    @property
    def intercept(self) -> torch.Tensor:
        mask = (self.slope > self.slope_threshold).float()
        return (-(self.slope * self.lower_bound) * mask) + ((1 - self.slope) * self.upper_bound * (1 - mask))


def modLayer(layer: nn.Module) -> nn.Module:
    layer_name = layer.__class__.__name__
    modified_layers = {"Linear": Linear, "ReLU": ReLU}

    if layer_name not in modified_layers:
        return copy.deepcopy(layer)

    return modified_layers[layer_name](layer)