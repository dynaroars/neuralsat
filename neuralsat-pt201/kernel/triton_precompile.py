import torch.nn as nn
import torch
import time

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 3)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


def kernels_precompile(device):
    print('Pre-compiling ...')
    start_time = time.time()
    pytorch_model = SimpleModel()
        
    num_outputs = 3
    data = torch.randn(10, 3, device=device)
    # y = 0
    # labels = torch.tensor([y]).long()
    # c = torch.eye(num_outputs).type_as(data)[labels].unsqueeze(1) - torch.eye(num_outputs).type_as(data).unsqueeze(0)
    # I = (~(labels.data.unsqueeze(1) == torch.arange(num_outputs).type_as(labels.data).unsqueeze(0)))
    # c = (c[I].view(data.size(0), num_outputs - 1, num_outputs))
    c = None
    
    net = BoundedModule(
        model=pytorch_model, 
        global_input=data,
        bound_opts={'conv_mode': 'patch', 'verbosity': 0},
        device=device,
        verbose=False,
    )
        
    x_L = torch.randn(10, 3, device=device)
    x_U = x_L + 0.5
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)

    net.init_alpha((x,), share_alphas=True, c=c, bound_upper=False)
    net.set_bound_opts({'optimize_bound_args': {'iteration': 2, 'use_float64_in_last_iteration': False}})
    net.compute_bounds(x=(x,), method='CROWN-Optimized', C=c, bound_upper=False)

    del data, c, x_L, x_U, x
    del pytorch_model, net
    # del labels, I

    if device == "cuda":
        torch.cuda.empty_cache()
    print(f'Kernels compiled in {time.time() - start_time:.4f}s.')


if __name__ == "__main__":
    kernels_precompile('cuda')