import math
import time

import torch
from torch import Tensor
from tqdm import trange

import haioc


@torch.jit.script
def preprocess_old(data: Tensor, xs: Tensor, inplace: bool = False) -> Tensor:
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


@torch.jit.script
def preprocess_python(data: Tensor, xs: Tensor, inplace: bool = False) -> Tensor:
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


@torch.jit.script
def preprocess_c(data: Tensor, xs: Tensor, inplace: bool = False) -> Tensor:
    data = data[haioc.any_eq_any(data, xs).logical_not_()]
    data = haioc.fill_if_eq_any(data, -xs, 0., inplace)
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


def _tensors_match(a: Tensor, b: Tensor, strict: bool = True):
    if strict:
        if a.size() == b.size():
            return torch.allclose(a, b)
    elif a.numel() == b.numel():
        return torch.allclose(a.flatten(), b.flatten())
    return False


def main(n_trials=100, device='cpu'):
    data_size = (7000, 100)
    xs_size = 5000
    data = torch.randperm(data_size[0] * data_size[1]).sub_(data_size[0] * data_size[1] // 2).view(data_size[0], data_size[1]).int().to(device)
    xs = torch.arange(0, xs_size).int().to(device)

    # warmup if using jit scripting:
    if isinstance(preprocess_python, torch.jit.ScriptFunction):
        preprocess_python(data, xs)

    print('old and python outputs match:',
          _tensors_match(preprocess_old(data, xs), preprocess_python(data, xs)))
    print('python and C outputs match:',
          _tensors_match(preprocess_python(data, xs), preprocess_c(data, xs)))

    time_meter = TimeMeter()

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            preprocess_old(data, xs)
        pbar.set_description(f'[old] average_fps={time_meter.fps:.05f}')

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            preprocess_python(data, xs)
        pbar.set_description(f'[python] average_fps={time_meter.fps:.05f}')

    time_meter.reset()
    pbar = trange(n_trials)
    for i in pbar:
        with time_meter:
            preprocess_c(data, xs)
        pbar.set_description(f'[C] average_fps={time_meter.fps:.05f}')


if __name__ == '__main__':
    main(50, 'cpu')
    main(50, 'cuda')
