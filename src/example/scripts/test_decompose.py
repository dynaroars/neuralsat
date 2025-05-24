import traceback
import random
import torch
import tqdm
import math
import os

from abstractor.auto_LiRPA.perturbations import PerturbationLpNorm
from abstractor.auto_LiRPA import BoundedModule, BoundedTensor
from helper.misc.torch_cuda_memory import gc_cuda, is_cuda_out_of_memory
from decomposer.utils import PytorchWrapper
from helper.misc.logger import logger
from setting import Settings

from helper.network.read_onnx import parse_onnx, parse_pth
from helper.spec.objective import parse_vnnlib

from train.models.vit.vit import *

def get_hidden_bounds(self, device):
    lower_bounds, upper_bounds = {}, {}
    # print(list(set(self.layers_requiring_bounds + self.split_nodes)))
    for layer in list(set(self.layers_requiring_bounds + self.split_nodes)):
        lower_bounds[layer.name] = layer.lower.detach().to(device)
        upper_bounds[layer.name] = layer.upper.detach().to(device)

    return lower_bounds, upper_bounds
   
def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params = }')
    return total_params
        
def execute(net, lower, upper, device, method='backward', cs=None, verbose=True):
    x_new = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros((1,) + lower.shape[1:], device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0, 'sparse_intermediate_bounds': False},
        device=device,
        verbose=False,
    )
    abstract.eval()
    abstract(x_new)
    abstract.get_split_nodes()

    # lb, ub = abstract.compute_bounds(x=(x_new,), method=method, C=cs, bound_upper=cs is None)
    print(lower.shape, upper.shape, x_new.shape, cs.shape if cs is not None else None)
    bound_upper = cs is None
    lb, ub, aux = abstract.init_alpha(x=(x_new,), c=cs, bound_upper=bound_upper, method='backward')

    if verbose:
        print(f'[{method}] {bound_upper=} {lb = }')
        print(f'[{method}] {bound_upper=} {ub = }')
    else:
        print(f'[{method}] {bound_upper=} {lb.shape = }')
        print(f'[{method}] {bound_upper=} {ub.shape = }')
        
        
    if method == 'crown-optimized' and cs is not None:
        lb, _ = abstract.compute_bounds(x=(x_new,), method=method, bound_upper=False, aux_reference_bounds=aux, C=cs) 

        if verbose:
            print(f'[{method}] {bound_upper=} {lb = }')
            print(f'[{method}] {bound_upper=} {ub = }')
        else:
            print(f'[{method}] {bound_upper=} {lb.shape = }')
            print(f'[{method}] {bound_upper=} {ub.shape = }')

    return lb.clone(), ub.clone() if ub is not None else None


def split_vit(model, split):
    nlayer = math.ceil(len(model.layers) / split)
    subnets = [PytorchWrapper(model.layers[i:i + nlayer]) for i in range(0, len(model.layers), nlayer)]
    print(split, [get_model_params(s) for s in subnets], sum([get_model_params(s) for s in subnets]), get_model_params(model))
    assert len(subnets) == split
    assert sum([get_model_params(s) for s in subnets]) == get_model_params(model)
    return subnets
    
    
def test_vit_2(model, input_lower, input_upper, device, method, cs):
    print(f'[input] lower={input_lower.sum().item()} upper={input_upper.sum().item()}')

    subnet0, subnet1 = split_vit(model, 2)
    input_shape = input_lower.shape

    for i in tqdm.tqdm(range(10)):
        x = torch.randn(i+1, *input_shape[1:]).to(input_lower)
        y1 = model(x)
        y2 = subnet1(subnet0(x))
        assert torch.equal(y1, y2)
    print('Matched')

    
    print('abstract subnet0', len(subnet0.layers))
    lb_pre1, ub_pre1 = execute(subnet0, input_lower, input_upper, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet1', len(subnet1.layers))
    lb_pre2, ub_pre2 = execute(subnet1, lb_pre1, ub_pre1, device, method=method, verbose=True, cs=cs)
    gc_cuda()
    

    
def test_vit_3(model, input_lower, input_upper, device, method, cs):
    print(f'[input] lower={input_lower.sum().item()} upper={input_upper.sum().item()}')
    gc_cuda()

    subnet0, subnet1, subnet2 = split_vit(model, 3)
    input_shape = input_lower.shape

    for i in tqdm.tqdm(range(10)):
        x = torch.randn(i+1, *input_shape[1:]).to(input_lower)
        y1 = model(x)
        y2 = subnet2(subnet1(subnet0(x)))
        assert torch.equal(y1, y2)
    print('Matched')

    print('abstract subnet0', len(subnet0.layers))
    lb_pre1, ub_pre1 = execute(subnet0, input_lower, input_upper, device, method=method, verbose=False)
    gc_cuda()

    print('abstract subnet1', len(subnet1.layers))
    lb_pre2, ub_pre2 = execute(subnet1, lb_pre1, ub_pre1, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet2', len(subnet2.layers))
    lb_pre3, ub_pre3 = execute(subnet2, lb_pre2, ub_pre2, device, method=method, verbose=True, cs=cs)
    gc_cuda()
    

def test():
    logger.setLevel(2)
    # torch.manual_seed(36)
    Settings.setup(None)
    n_outputs = 10
    device = 'cuda'
    batch = 1
    indices = torch.arange(0, batch)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(device).transpose(0, 1)

    input_shape = (batch, 3, 32, 32)
    # method = 'forward+backward'
    # method = 'forward'
    model = vit_3_32()
    state_dict = torch.load('train/weights/vit/vit_3_32/model_best.pth.tar', weights_only=False)
    print(state_dict['state_dict'].keys())
    model.load_state_dict(state_dict['state_dict'])
    model.to(device)
    
    # model = 
    model.eval()
    input_lower = torch.randn(input_shape, device=device)
    input_upper = input_lower + 0.0001

    
    if 0:
        method = 'backward'        
        print('abstract full')
        lb_full, ub_full = execute(model, input_lower, input_upper, device, method=method, cs=cs)
        exit()
        
        
    if 1:
        try:
            method = 'backward'        
            test_vit_2(
                model=model, 
                input_lower=input_lower[0:1], 
                input_upper=input_upper[0:1], 
                device=device, 
                method=method,
                cs=cs[0:1],
            )
            print('[+] Split 2 success\n\n')
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                print('[+] Split 2 OOM\n\n')
            else:
                traceback.print_exc()
        except:
            traceback.print_exc()
    
    # exit()
        
    print('\n\n==========\n\n')
    method = 'crown-optimized'        
    method = 'backward'
    gc_cuda()
    test_vit_3(
        model=model, 
        input_lower=input_lower, 
        input_upper=input_upper, 
        device=device, 
        method=method,
        cs=cs,
    )
    exit()
    
    
def test2():
    logger.setLevel(2)
    # torch.manual_seed(36)
    Settings.setup(None)
    n_outputs = 10
    device = 'cuda'
    batch = 1
    indices = torch.arange(0, batch)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(device).transpose(0, 1)
    
    pth_path = 'example/generated_benchmark/vit/eps_0.030000_vit_3_32/net/vit_3_32.pth'
    vnnlib_path = 'example/generated_benchmark/vit/eps_0.030000_vit_3_32/spec/'
    vnnlib_name = os.path.join(vnnlib_path, random.choice(os.listdir(vnnlib_path)))
    
    model, input_shape, output_shape = parse_pth(pth_path)
    model = model.to(device)
    
    dnf_objectives = parse_vnnlib(vnnlib_name, input_shape)
    objective = dnf_objectives.pop(batch)
    
    input_lower = objective.lower_bounds.view(batch, *input_shape[1:]).to(device)
    input_upper = objective.upper_bounds.view(batch, *input_shape[1:]).to(device)

    # input_upper = input_lower.clone()
    # perturb_range = 0.03
    # num_perturb = 50
    # indices = torch.randperm(input_lower.numel())[:num_perturb]
    # input_upper.flatten()[indices] += perturb_range
       
       
    diff = input_upper - input_lower
    eps = diff.max().item()
    perturbed = (diff > 0).int().sum()
    print(f'[!] eps={eps:.06f}, perturbed={perturbed}')
    
    method = 'crown-optimized'        
    method = 'backward'
    gc_cuda()
    test_vit_3(
        model=model, 
        input_lower=input_lower, 
        input_upper=input_upper, 
        device=device, 
        method=method,
        cs=cs,
    )
    
if __name__ == "__main__":
    test2()