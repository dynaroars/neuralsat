import random
import torch
import copy
import time

import math

from tqdm import trange


def rand(total_samples, num_samples, device='cpu'):
    if num_samples >= total_samples:
        return torch.Tensor(range(total_samples), device=device)
    return torch.Tensor(random.sample(range(total_samples), num_samples), device=device)



# print(rand(10, 5))
# print(rand(10, 15))

# X = torch.zeros(10)
# print(torch.empty_like(X).uniform_())


def test_solver():
    from solver.sat_solver import SATSolver
    import time
    
    init_alls = []
    
    if 0:
        for i in range(5000):
            init_alls.append(SATSolver([[_ for _ in range(i)]]))
            # init_alls.append(SATSolver([frozenset({i})]))
            
        tic = time.time()
        alls = init_alls + copy.deepcopy(init_alls)
        print('done', time.time() - tic)
        return
       
    init_alls = []
    for i in range(3):
        init_alls.append(SATSolver([[i+1], [i+1, i + 2], [i+1, -(i+3)]]))
        
    tic = time.time()
    alls = init_alls + copy.deepcopy(init_alls)
    # alls = init_alls + init_alls
    print('done', time.time() - tic)
    
    
    print('-----------all------------')
    [_.print_stats() for _ in alls]
    
    batch = 2
    picks = alls[-batch:]
    alls = alls[:-batch]
    
    
    print('-----------pick------------')
    [_.print_stats() for _ in picks]
    
    
    print('-----------remain------------')
    [_.print_stats() for _ in alls]
    
    
    # for i in range(5, 8):
    #     alls.append(SATSolver([frozenset({i})]))
    
    print('-----------bcp------------')
    [_.bcp() for _ in picks]
        
    print('-----------add------------')
    [alls.append(_) for _ in picks]
    
    
    print('-----------all------------')
    [_.print_stats() for _ in alls]
    
    
    
    
def test_small():
    size = 3
    a = {}
    for i in range(size):
        a[i] = [list(range(_)) for _ in range(i+2)]
        a[i] = list(range(i+2))
    # b = {}
    b = {k: v.copy() for k, v in a.items()}
    b[1].remove(1)
    print(a)
    print(b)
    
    tic = time.time()
    b = copy.deepcopy(a)
    print('copy dict in:', time.time() - tic)
    
    # a = [[1, 2, 3], [3, 4, 5]]
    # b = a[:]
    # b[0].remove(1)
    
    a = []
    for i in range(size):
        a.append(list(range(i+2)))
    
    tic = time.time()
    b = [_[:] for _ in a]
    b[1].remove(1)
    # print('copy list in:', time.time() - tic)
    print(a)
    print(b)
    
    
def test():
    
    a = torch.tensor([1, 2, 3, 4, 5])  
    b = torch.tensor([2, 3])  
    print(a)
    print(torch.where(a == b))
    a = a[a != b]
    print(a)



# @torch.jit.script
def haioc_old(data: torch.Tensor, xs: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    for i in range(xs.size(0)):
        remove_idx = torch.where(data == xs[i])[0]
        remain_mask = torch.ones(len(data), dtype=torch.bool)
        remain_mask[remove_idx] = False
        data = data[remain_mask]
        if not inplace:
            data = torch.where(data == -xs[i], 0, data)
        else:
            data[data == -xs[i]] = 0
    return data


# @torch.jit.script
def haioc(data: torch.Tensor, xs: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    remain_mask = torch.ones(data.size(0), dtype=torch.bool, device=data.device)
    for i in range(xs.size(0)):
        remain_mask &= data.ne(xs[i]).all(1)
    data = data[remain_mask]
    zero_mask = torch.zeros(data.size(), dtype=torch.bool, device=data.device)
    for i in range(xs.size(0)):
        zero_mask |= data.eq(-xs[i])
    if not inplace:
        data = torch.where(zero_mask, 0, data)
    else:
        data[zero_mask] = 0
    return data


class TimeMeter:
    def __init__(self):
        self.avg = self.count = 0
        self.start = self.end = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        elapsed_time = self.end - self.start
        self.avg = (self.avg * self.count + elapsed_time) / (self.count + 1)
        self.count += 1

    def reset(self):
        self.avg = self.count = 0
        self.start = self.end = None

    @property
    def fps(self):
        return 1. / self.avg if self.avg else math.nan


def main(n_trials=100, inplace=False):
    size = (5000, 50)
    n = 3000
    data = torch.randperm(size[0] * size[1]).sub_(size[0] * size[1] // 2).view(size[0], size[1]).int()
    xs = torch.arange(0, n).int()

    time_meter = TimeMeter()

    # warmup if using jit scripting:
    if isinstance(haioc, torch.jit.ScriptFunction):
        haioc(data, xs, inplace=inplace)

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        torch.manual_seed(i)
        with time_meter:
            haioc_old(data, xs, inplace=inplace)
        pbar.set_description(f'[old] average_fps={time_meter.fps:.05f}')

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        torch.manual_seed(i)
        with time_meter:
            haioc(data, xs, inplace=inplace)
        pbar.set_description(f'[new] average_fps={time_meter.fps:.05f}')

def extract_log():
    import os
    import re
    
    root = './mnistfc_hard_results/'
    with open('valid_instances.csv', 'w') as fp:
        for file in os.listdir(root):
            file_path = os.path.join(root, file)
            content = open(file_path).read().strip()
            if ('unsat' not in content) and ('timeout' not in content):
                continue
            
            net, spec = file.split('_[spec]_')
            net = net.split('[net]_')[1]
            spec = spec.split('.log')[0]
            # print(net, spec)
            # if 'unsat' in content:
            if 'timeout,' in content:
                last_iter_log = content.split('\n')[-3]
                last_bound = float(re.findall(r'Bound: -([0-9\.]+)', last_iter_log)[0])
                # print(last_bound)
                if last_bound >= 5.0:
                    continue
                print(last_iter_log)
            print(file_path)
                
            print(f'onnx/{net}.onnx,vnnlib/{spec}.vnnlib,3600', file=fp)

if __name__ == '__main__':
    # main(100, True)
    extract_log()

