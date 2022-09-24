from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch.nn as nn
import numpy as np
import torch

from utils.timer import Timers

class CrownWrapper:

    def __init__(self, net):
        x = torch.zeros(net.input_shape)

        from_layer = {k: [] for k in net.layers_mapping.keys()}
        input_from_layer = {}

        self.layers = {}

        idx = 0
        for layer in net.layers:
            if isinstance(layer, nn.ReLU):
                input_from_layer[idx] = x
                idx += 1
            x = layer(x)

        for k in from_layer:
            idx = 0
            for layer in net.layers:
                if isinstance(layer, nn.ReLU):
                    idx += 1
                if k < idx:
                    from_layer[k] += [layer]

        for k in from_layer:
            self.layers[k] = BoundedModule(nn.Sequential(*from_layer[k]), input_from_layer[k])

        self.device = net.device
        # exit()


    @torch.no_grad()
    def forward_layer(self, lower, upper, layer_id):
        lower = lower.unsqueeze(0)
        upper = upper.unsqueeze(0)

        module = self.layers[layer_id]
        ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=lower, x_U=upper)
        data = (lower + upper) / 2
        x = BoundedTensor(data, ptb).to(self.device)
        (lb, ub), unstable_neurons = module.compute_bounds(x=(x,), method='backward')

        lb = lb.squeeze(0)
        ub = ub.squeeze(0)

        return (lb, ub), unstable_neurons