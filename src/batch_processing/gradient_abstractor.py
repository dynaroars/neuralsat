import torch.nn as nn
import random
import torch
import time
import copy

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


def tensor_delete_indices(x, indices):
    mask = torch.ones(x.numel(), dtype=torch.bool, device=x.device)
    mask[indices] = False
    return x[mask]


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
        # tic = time.time()
        for i in range(len(lower)):
            stat = self.spec.check_output_reachability(lower[i], upper[i])[0]
            if not stat:
                mask[i] = 0
        # print(len(mask), mask.sum(), time.time() - tic)
        return mask

    def get_loss(self, lower, upper):
        # return (upper*mask)[..., 0].mean()
        return (upper - lower).mean()
    
    def get_optimized_bounds_from_input(self, lower, upper, assignment=None):
        # print(lower.shape)
        (lb, ub), invalid_batch, hidden_bounds = self.abstractor(lower, upper, assignment=assignment, return_hidden_bounds=True, reset_param=True)
        # print(invalid_batch, len(list(set(invalid_batch))))
        sat_mask = self.get_sat_mask(lb, ub)
        invalid_batch += torch.where(sat_mask==False)[0].detach().cpu().numpy().tolist()
        if sat_mask.sum() == 0:
            # print('all unsat')
            return (lb, ub), list(set(invalid_batch)), hidden_bounds

        # print(invalid_batch, len(list(set(invalid_batch))))

        # Timers.tic('Init Optimize')
        # Timers.toc('Init Optimize')

        # active_batch = torch.arange(len(lower), dtype=torch.int16, device=lower.device)
        # active_batch = tensor_delete_indices(active_batch.clone(), invalid_batch)
        # print(active_batch.numpy().tolist(), len(active_batch))
        valid_batch = torch.ones(len(lower), dtype=torch.bool, device=lower.device)
        valid_batch[invalid_batch] = False
        if valid_batch.sum() == 0:
            return (lb, ub), list(set(invalid_batch)), hidden_bounds
        # print(valid_batch.sum())

        if valid_batch.sum() >= 10:
            return (lb, ub), list(set(invalid_batch)), hidden_bounds

        # exit()
        # invalid_batch = []

        new_lower = lower.clone()[valid_batch]
        new_upper = upper.clone()[valid_batch]
        # new_assignment = assignment[valid_batch.numpy()]
        active_batch = torch.where(valid_batch==True)[0].detach().cpu()#.numpy()
        new_assignment = [assignment[i] for i in active_batch]

        self.abstractor(new_lower, new_upper, assignment=new_assignment, return_hidden_bounds=True, reset_param=True)
        optimizer = torch.optim.Adam(self.abstractor.parameters(), lr=self.lr)

        new_invalid_batch = []
        for it in range(self.n_iters):
            # print('optimize iter:', it)
            optimizer.zero_grad(set_to_none=True)
            Timers.tic('Abstraction')
            (l, u), iv, hb = self.abstractor(new_lower, new_upper, assignment=new_assignment, return_hidden_bounds=True, reset_param=False)
            Timers.toc('Abstraction')
            # loss = (uo - lo).sum()
            sat_mask = self.get_sat_mask(l, u)
            # print(unsat_mask)
            invalid_batch += [int(active_batch[i]) for i in iv]
            invalid_batch += [int(active_batch[i]) for i in torch.where(sat_mask==False)[0]]
            new_invalid_batch += iv
            new_invalid_batch += torch.where(sat_mask==False)[0].detach().cpu().numpy().tolist()
            # active_batch = tensor_delete_indices(active_batch.clone(), iv)

            if sat_mask.sum() == 0:
                assert len(list(set(invalid_batch))) == len(lower)
                # print('all unsat at iter', it)
                break
            # idx_sat = torch.where(unsat_mask == True)
            # print(idx_sat)
            loss = self.get_loss(l[sat_mask], u[sat_mask])
            
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

        optimizer.zero_grad(set_to_none=True)
        del optimizer
        # print(lb[valid_batch].shape, l.shape)
        # old_lb = lb.clone()
        new_invalid_batch = list(set(new_invalid_batch))
        new_valid_batch = torch.ones(valid_batch.sum(), dtype=torch.bool, device=lower.device)
        new_valid_batch[new_invalid_batch] = False
        # print(new_valid_batch.sum())
        lb[valid_batch][new_valid_batch] = l[new_valid_batch]
        ub[valid_batch][new_valid_batch] = u[new_valid_batch]
        
        # print(lb[valid_batch][new_valid_batch])
        # print(ub[valid_batch][new_valid_batch])
        # old_hidden = copy.deepcopy(hidden_bounds)
        for idx in range(len(hidden_bounds)):
            for i1, i2 in enumerate(active_batch):
                # print(f'[{idx}] hb[{i2}] = h[{i1}]')
                hidden_bounds[idx][i2] = hb[idx][i1]
        # print(torch.equal(old_lb, lb))
        return (lb, ub), list(set(invalid_batch)), hidden_bounds