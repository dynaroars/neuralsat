import torch

@torch.compile(mode='reduce-overhead')
def triton_matmul(x, y):
    raise
    return x.to(y).matmul(y)
