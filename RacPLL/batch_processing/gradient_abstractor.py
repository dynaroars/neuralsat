import torch.nn as nn
import random
import torch
import time

from batch_processing.deeppoly import BatchDeepPoly, BatchReLUTransformer
from utils.timer import Timers


class BetaClipper:

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'beta'):
            # print('\tclip:', module)
            module.beta.data = module.beta.data.clamp(0.0, 1.0)
    


class BetaZeroGrader:

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'zero_grad_indices'):
            # print('\tzero grad:', module)
            for ind in module.zero_grad_indices:
                module.beta.grad[ind] = torch.zeros_like(module.beta.grad[ind], device=module.beta.device)


class DeepPolyLayerWrapper(nn.Module):

    def __init__(self, abstractor):
        super(DeepPolyLayerWrapper, self).__init__()
        self.abstractor = abstractor

    def forward(self, lower, upper, assignment=None, return_hidden_bounds=False, reset_param=False):
        bounds = (lower, upper)
        hidden_bounds = []
        if reset_param:
            self.reset_parameter()

        for layer in self.abstractor:
            if isinstance(layer, BatchReLUTransformer):
                hidden_bounds.append(bounds.permute(1, 2, 0)) # B x 2 x H
            bounds, _ = layer(bounds, assignment)
            assert torch.all(bounds[..., 0] <= bounds[..., 1])
        if return_hidden_bounds:
            return (bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)), hidden_bounds
        return bounds[..., 0].transpose(0, 1), bounds[..., 1].transpose(0, 1)


    def reset_parameter(self):
        for layer in self.abstractor:
            if isinstance(layer, BatchReLUTransformer):
                layer.reset_beta()


class GradientAbstractor:

    def __init__(self, net, spec, n_iters=20, lr=0.2):

        self.abstractor = BatchDeepPoly(net)
        self.spec = spec
        self.net = net

        self.clipper = BetaClipper()
        self.zero_grader = BetaZeroGrader()
        self.lr = lr
        self.n_iters = n_iters

        self.layer_abstractors = {k: DeepPolyLayerWrapper(v) for k, v in self.abstractor.forward_from_layer.items()}


    def get_optimized_bounds_from_layer(self, lower, upper, layer_id):
        Timers.tic('Init Optimize')
        abstractor = self.layer_abstractors[layer_id]
        # print('--------run init')
        abstractor(lower, upper, reset_param=True)

        optimizer = torch.optim.Adam(abstractor.parameters(), lr=self.lr)
        Timers.toc('Init Optimize')

        for it in range(self.n_iters):
            # print('optimize iter:', it)
            optimizer.zero_grad()
            Timers.tic('Abstraction')
            (lb, ub), hb = abstractor(lower, upper, return_hidden_bounds=True, reset_param=False)
            Timers.toc('Abstraction')
            # loss = (uo - lo).sum()
            sat_mask = self.get_sat_mask(lb, ub)
            # print(unsat_mask)

            if sat_mask.sum() == 0:
                # print('all unsat at iter', it)
                break
            # print(it, unsat_mask.sum(), len(lb))
            # idx_sat = torch.where(unsat_mask == True)
            # print(idx_sat)
            loss = self.get_loss(lb[sat_mask], ub[sat_mask])
            
            # print(it, loss)
            Timers.tic('Backward Loss')
            loss.backward()
            Timers.toc('Backward Loss')

            # Timers.tic('Zero Grad')
            # self.abstractor.apply(self.zero_grader)
            # Timers.toc('Zero Grad')

            Timers.tic('Update Beta')
            optimizer.step()
            Timers.toc('Update Beta')

            Timers.tic('Clip Beta')
            abstractor.apply(self.clipper)
            Timers.toc('Clip Beta')

        return (lb, ub), [], hb

    def get_sat_mask(self, lower, upper):
        # print(lower.shape)
        mask = torch.ones(lower.shape[0], device=lower.device, dtype=torch.bool)
        for i in range(len(lower)):
            stat = self.spec.check_output_reachability(lower[i], upper[i])[0]
            if not stat:
                mask[i] = 0
        return mask

    def get_loss(self, lower, upper):
        # return (upper*mask)[..., 0].mean()
        return (upper - lower).mean()
    
    def get_optimized_bounds_from_input(self, lower, upper, assignment=None):
        Timers.tic('Init Optimize')
        self.abstractor(lower, upper, reset_param=True)

        optimizer = torch.optim.Adam(self.abstractor.parameters(), lr=self.lr)
        Timers.toc('Init Optimize')

        for it in range(self.n_iters):
            # print('optimize iter:', it)
            optimizer.zero_grad()
            Timers.tic('Abstraction')
            (lb, ub), invalid_batch, hb = self.abstractor(lower, upper, assignment=assignment, return_hidden_bounds=True, reset_param=False)
            Timers.toc('Abstraction')
            # loss = (uo - lo).sum()
            sat_mask = self.get_sat_mask(lb, ub)
            # print(unsat_mask)

            if sat_mask.sum() == 0:
                # print('all unsat at iter', it)
                break
            # print(it, unsat_mask.sum(), len(lb))
            # idx_sat = torch.where(unsat_mask == True)
            # print(idx_sat)
            loss = self.get_loss(lb[sat_mask], ub[sat_mask])
            
            # print(it, loss)
            Timers.tic('Backward Loss')
            loss.backward()
            Timers.toc('Backward Loss')

            # Timers.tic('Zero Grad')
            # self.abstractor.apply(self.zero_grader)
            # Timers.toc('Zero Grad')

            Timers.tic('Update Beta')
            optimizer.step()
            Timers.toc('Update Beta')

            Timers.tic('Clip Beta')
            self.abstractor.apply(self.clipper)
            Timers.toc('Clip Beta')

        return (lb, ub), invalid_batch, hb