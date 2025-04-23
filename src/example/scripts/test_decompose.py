import torch
import tqdm
import os

from abstractor.auto_LiRPA.perturbations import PerturbationLpNorm
from abstractor.auto_LiRPA import BoundedModule, BoundedTensor
from helper.misc.torch_cuda_memory import gc_cuda
from helper.misc.logger import logger

from trainer.models.vit.vit import vit_medium_2, vit_medium_3
from test import extract_instance

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
        
def execute(net, shape, lower, upper, device, method='backward', cs=None, verbose=True):
    x_new = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros(shape, device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0, 'sparse_intermediate_bounds': False},
        device=device,
        verbose=False,
    )
    abstract.eval()
    abstract(x_new)
    abstract.get_split_nodes()

    # lb, ub = abstract.compute_bounds(x=(x_new,), method=method, C=cs, bound_upper=cs is None)
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

    return abstract, lb, ub


def split_vit(model, split):
    subnets = list(model.children())
    print([get_model_params(s) for s in subnets], sum([get_model_params(s) for s in subnets]), get_model_params(model))
    assert len(subnets) == split
    assert sum([get_model_params(s) for s in subnets]) == get_model_params(model)
    return subnets
    
    
def test_vit_2(model, input_shape, n_outputs, device, method):

    subnet0, subnet1 = split_vit(model, 2)

    for i in tqdm.tqdm(range(10)):
        x = torch.randn(i+1, *input_shape[1:])
        y1 = model(x)
        y2 = subnet1(subnet0(x))
        assert torch.equal(y1, y2)
    print('Matched')


    input_lower = torch.randn(input_shape, device=device)
    input_upper = input_lower + 0.001

    indices = torch.arange(0, 1)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lower)
    
    print('abstract subnet0')
    abstract_pre1, lb_pre1, ub_pre1 = execute(subnet0, input_shape, input_lower, input_upper, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet1')
    abstract_pre2, lb_pre2, ub_pre2 = execute(subnet1, lb_pre1.shape, lb_pre1, ub_pre1, device, method=method, verbose=True, cs=cs)
    gc_cuda()
    

    
def test_vit_3(model, input_shape, n_outputs, device, method):

    subnet0, subnet1, subnet2 = split_vit(model, 3)

    for i in tqdm.tqdm(range(10)):
        x = torch.randn(i+1, *input_shape[1:])
        y1 = model(x)
        y2 = subnet2(subnet1(subnet0(x)))
        assert torch.equal(y1, y2)
    print('Matched')


    input_lower = torch.randn(input_shape, device=device)
    input_upper = input_lower + 0.001

    indices = torch.arange(0, 1)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lower)
    
    print('abstract subnet0')
    abstract_pre1, lb_pre1, ub_pre1 = execute(subnet0, input_shape, input_lower, input_upper, device, method=method, verbose=False)
    gc_cuda()

    print('abstract subnet1')
    abstract_pre2, lb_pre2, ub_pre2 = execute(subnet1, lb_pre1.shape, lb_pre1, ub_pre1, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet2')
    abstract_pre3, lb_pre3, ub_pre3 = execute(subnet2, lb_pre2.shape, lb_pre2, ub_pre2, device, method=method, verbose=True, cs=cs)
    gc_cuda()
    

if __name__ == "__main__":
    logger.setLevel(2)
    # torch.manual_seed(4)
    
    input_shape = (1, 3, 32, 32)
    n_outputs = 10
    device = 'cuda'

    method = 'crown-optimized'        
    # method = 'backward'        
    # method = 'forward+backward'
    # method = 'forward'
    if 0:
        print('abstract full')
        model_2 = vit_medium_2()
        input_lower = torch.randn(input_shape, device=device)
        input_upper = input_lower + 0.001
        abstract_full, lb_full, ub_full = execute(model_2, input_shape, input_lower, input_upper, device, method=method)

    if 0:
        model_2 = vit_medium_2()
        model_2.eval()
        test_vit_2(
            model=model_2, 
            input_shape=input_shape, 
            n_outputs=n_outputs, 
            device=device, 
            method=method,
        )
    else:
        model_3 = vit_medium_3()
        model_3.eval()
        test_vit_3(
            model=model_3, 
            input_shape=input_shape, 
            n_outputs=n_outputs, 
            device=device, 
            method=method,
        )
    exit()
    
    if 1:
        model = vit_medium_2()
        model.eval()
        # print(model)
        get_model_params(model)
        layers = list(model.children())
        
        subnet0 = layers[0]
        subnet1 = layers[1]
        subnet2 = layers[2]
        
        print(f'{subnet0=}')
        print(f'{subnet1=}')
        print(f'{subnet2=}')
        
        for i in tqdm.tqdm(range(10)):
            x = torch.randn(i+1, *input_shape[1:])
            y1 = model(x)
            y2 = subnet2(subnet1(subnet0(x)))
            # y2 = subnet1(subnet0(x))
            assert torch.equal(y1, y2)
        print('Matched')
        # print(model)
        # output_name = f'example/onnx/vit_toy.onnx'    
        
        # torch.onnx.export(
        #     model,
        #     torch.zeros(input_shape),
        #     output_name,
        #     opset_version=12,
        #     input_names=["input"],
        #     output_names=["output"],
        #     dynamic_axes={
        #         'input': {0: 'batch_size'},
        #         'output': {0: 'batch_size'},
        #     }
        # )
        
        # os.system(f'onnxsim "{output_name}" "{output_name}"')
        # print(f'[+] Exporting ONNX: {output_name=}')
            
    else:
        net_path = 'example/onnx/pgd_2_3_16.onnx'
        vnnlib_path = 'example/vnnlib/pgd_2_3_16_4021.vnnlib'
        model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
        model.to(device)
        pass
    
        
    # method = 'crown-optimized'        
    method = 'backward'        
    # method = 'forward+backward'
    # method = 'forward'
    input_lower = torch.randn(input_shape, device=device)
    input_upper = input_lower + 0.001
    # input_upper[:, 0] = input_lower[:, 0] + .001
    # model = model.to(device)
    
    # print(model)
    
    # cs = None
    
    indices = torch.arange(0, 1)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lower)
    
    # print('abstract full')
    # abstract_full, lb_full, ub_full = execute(model, input_shape, input_lower, input_upper, device, method=method, cs=cs)
    # exit()
    
    print('abstract subnet0')
    abstract_pre1, lb_pre1, ub_pre1 = execute(subnet0, input_shape, input_lower, input_upper, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet1')
    abstract_pre2, lb_pre2, ub_pre2 = execute(subnet1, lb_pre1.shape, lb_pre1, ub_pre1, device, method=method, verbose=False)
    gc_cuda()
    
    print('abstract subnet2')
    abstract_pre3, lb_pre3, ub_pre3 = execute(subnet2, lb_pre2.shape, lb_pre2, ub_pre2, device, method=method, verbose=True, cs=cs)
    gc_cuda()
    
    
    exit()
    # print()
    