import torch
import tqdm
import os

from abstractor.auto_LiRPA.perturbations import PerturbationLpNorm
from abstractor.auto_LiRPA import BoundedModule, BoundedTensor
from helper.misc.logger import logger

from trainer.models.vit.vit import vit_medium
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
    print(f'{total_params = }')
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
    lb, ub, _ = abstract.init_alpha(x=(x_new,), c=cs, bound_upper=cs is None, method=method)

    if verbose:
        print(f'[{method}] {lb = }')
        print(f'[{method}] {ub = }')
    return abstract, lb, ub


if __name__ == "__main__":
    logger.setLevel(2)
    input_shape = (1, 3, 32, 32)
    n_outputs = 10
    device = 'cuda'
    torch.manual_seed(0)
    
    if 1:
        model = vit_medium()
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
    input_upper = input_lower + .01
    model = model.to(device)
    
    # print(model)
    
    # cs = None
    
    indices = torch.arange(0, 1)
    cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lower)
    
    print('abstract full')
    abstract_full, lb_full, ub_full = execute(model, input_shape, input_lower, input_upper, device, method=method, cs=cs)
    exit()
    
    print('abstract subnet0')
    print(subnet0)
    abstract_pre, lb_pre, ub_pre = execute(subnet0, input_shape, input_lower, input_upper, device, method=method, verbose=True)
    exit()
    # print()
    