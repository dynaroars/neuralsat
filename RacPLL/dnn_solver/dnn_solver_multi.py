import numpy as np
import itertools
import torch
import time
import copy

from dnn_solver.dnn_solver import DNNSolver
import settings

def naive_split_bounds(bounds, steps=3):
    lower = bounds['lbs']
    upper = bounds['ubs']

    bs = [(l, u) for l, u in zip(lower, upper)]
    bs = [torch.linspace(b[0], b[1], steps=steps) for b in bs]
    bs = [[torch.Tensor([b[i], b[i+1]]) for i in range(b.shape[0] - 1)] for b in bs]
    bs = itertools.product(*bs)
    splits = [{'lbs': torch.Tensor([_[0] for _ in b]),
               'ubs': torch.Tensor([_[1] for _ in b])} for b in bs]
    random.shuffle(splits)
    return splits



class DNNSolverMulti:

    def __init__(self, net, spec, initial_splits=10):

        self.net = net
        self.new_specs = []

        bounds = spec.get_input_property()
        lower = torch.tensor(bounds['lbs'], dtype=settings.DTYPE)
        upper = torch.tensor(bounds['ubs'], dtype=settings.DTYPE)

        # print(lower)
        # print(upper)
        # print()

        self.multi_bounds = self.split_multi_bounds(lower.clone(), upper.clone(), initial_splits)

        for l, u in self.multi_bounds:
            s = copy.deepcopy(spec)
            s.bounds = [(li.item(), ui.item()) for li, ui in zip(l, u)]
            self.new_specs.append(s)


    # def split_spec(self, spec, steps=2):
    #     bs = spec.bounds
    #     idx = 5
    #     bs = [np.linspace(b[0], b[1], steps+1) for b in bs[:idx]] + [np.array(b) for b in bs[idx:]]
    #     bs = [[[b[i], b[i+1]] for i in range(len(b) - 1)] for b in bs]
    #     bs = list(itertools.product(*bs))
    #     return bs


    def solve(self):
        for idx, spec in enumerate(self.new_specs):
            solver = DNNSolver(self.net, spec)
            print('lower:', self.multi_bounds[idx][0])
            print('upper:', self.multi_bounds[idx][1])
            tic = time.time()
            status = solver.solve()
            print(f'{idx}/{len(self.new_specs)}', status, time.time() - tic)



    def split_multi_bounds(self, lower, upper, initial_splits=10):
        if initial_splits <= 1:
            return ([(lower, upper)])
        grads = self.estimate_grads(lower, upper, steps=3)
        smears = np.multiply(torch.abs(grads) + 1e-5, [u - l for u, l in zip(upper, lower)]) + 1e-5
        print(smears.argmax())
        split_multiple = initial_splits / smears.sum()
        num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
        print('num_splits:', num_splits)
        assert all([x>0 for x in num_splits])
        return self.split_multi_bound([(lower, upper)], d=num_splits)




    def estimate_grads(self, lower, upper, steps=3):
        inputs = [(((steps - i) * lower + i * upper) / steps) for i in range(steps + 1)]
        diffs = torch.zeros(len(lower), dtype=settings.DTYPE)

        for sample in range(steps + 1):
            pred = self.net(inputs[sample].unsqueeze(0))
            for index in range(len(lower)):
                if sample < steps:
                    l_input = [m if i != index else u for i, m, u in zip(range(len(lower)), inputs[sample], inputs[sample+1])]
                    l_input = torch.tensor(l_input, dtype=settings.DTYPE).unsqueeze(0)
                    l_i_pred = self.net(l_input)
                else:
                    l_i_pred = pred
                if sample > 0:
                    u_input = [m if i != index else l for i, m, l in zip(range(len(lower)), inputs[sample], inputs[sample-1])]
                    u_input = torch.tensor(u_input, dtype=settings.DTYPE).unsqueeze(0)
                    u_i_pred = self.net(u_input)
                else:
                    u_i_pred = pred
                diff = [abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)][0]
                diffs[index] += diff.sum()
        return diffs / steps


    def split_multi_bound(self, multi_bound, dim=0, d=2):
        if isinstance(d, int):
            di = d
        else:
            di = d[dim]
        new_multi_bound = []
        for idx, (lower, upper) in enumerate(multi_bound):
            d_lb = lower[dim].clone()
            d_ub = upper[dim].clone()

            d_range = d_ub-d_lb
            d_step = d_range/di
            for i in range(di):
                # print(idx, dim, len(multi_bound), d_step, d_lb, d_ub)
                lower[dim] = d_lb + i*d_step
                upper[dim] = d_lb + (i+1)*d_step
                new_multi_bound.append((lower.clone(), upper.clone()))
                # print('new lower:', new_multi_bound[-1][0])
                # print('new upper:', new_multi_bound[-1][1])
            # print()
        # print('--')
        if dim + 1 < len(upper):
            return self.split_multi_bound(new_multi_bound, dim=dim+1, d=d)
        else:
            return new_multi_bound
