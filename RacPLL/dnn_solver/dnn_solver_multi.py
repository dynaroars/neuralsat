import numpy as np
import itertools
import torch
import time
import copy

from dnn_solver.dnn_solver import DNNSolver
import settings

def split_bounds(bounds, steps=3):
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

    def __init__(self, net, spec):

        self.net = net
        self.new_specs = []

        bounds = self.split_spec(spec)
        for bound in bounds:
            s = copy.deepcopy(spec)
            s.bounds = bound
            self.new_specs.append(s)

        print(len(self.new_specs))

        # for s in new_specs:
            # print(s.bounds, s.mat)



    def split_spec(self, spec, steps=2):
        bs = spec.bounds
        idx = 5
        bs = [np.linspace(b[0], b[1], steps+1) for b in bs[:idx]] + [np.array(b) for b in bs[idx:]]
        bs = [[[b[i], b[i+1]] for i in range(len(b) - 1)] for b in bs]
        bs = list(itertools.product(*bs))
        return bs


    def solve(self):
        for idx, spec in enumerate(self.new_specs):
            solver = DNNSolver(self.net, spec)
            tic = time.time()
            status = solver.solve()
            print(idx, status, time.time() - tic)

