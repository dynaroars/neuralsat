import torch
import time

from abstractor.auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from test import extract_instance

@torch.no_grad()
def test1():
    onnx_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    
    
    onnx_path = 'example/onnx/vit.onnx'
    vnnlib_path = 'example/vnnlib/spec_vit.vnnlib'
    
    device = 'cuda'
    device = 'cpu'
    # method = 'backward'
    extra_opts = {'sparse_intermediate_bounds': False, 'conv_mode': 'matrix'}
    # extra_opts = {}
    bound_opts = {'conv_mode': 'patches', 'verbosity': 0, **extra_opts}
    print(f'{bound_opts=}')
    model, input_shape, objectives = extract_instance(onnx_path, vnnlib_path)
    print(model)
    

    polytope = BoundedModule(
        model=model, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts=bound_opts,
        device=device,
        verbose=False,
    )
    polytope.eval()
    
    objective = objectives.pop(1)
    
    x_L = objective.lower_bounds[0].view(input_shape)
    x_U = objective.upper_bounds[0].view(input_shape)
    
    perturbation = PerturbationLpNorm(x_L=x_L, x_U=x_U)
    x = BoundedTensor(x_L, perturbation).to(device)
    
    # lb, _ = polytope.compute_bounds(x=(x,), method=method, bound_upper=False)
    
    lb, _, _ = polytope.init_alpha(x=(x,), bound_upper=False)
    print(f'{lb.tolist()=}')
    print(f"GPU Mem: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    
    # polytope.visualize('example/scripts/graph')
    

def test2():
    from train.models.gpt.gpt_fake import GPT
    device = 'cpu'
    # device = 'cuda'
    
    class Config:
        pass
    
    config = Config()
    config.vocab_size = 4
    config.n_layer = 5
    config.n_head = 3
    config.n_embd = 6
    
    model = GPT(config)
    print(model)
    
    x = torch.randint(0, config.vocab_size, (1, config.n_embd)).float()
    print(x.shape)
    logits = model(x)
    print(logits.shape)
    
    
    bound_opts = {'verbosity': 0, 'sparse_intermediate_bounds': True, 'conv_mode': 'matrix'}
    print(f'{bound_opts=}')

    polytope = BoundedModule(
        model=model, 
        global_input=torch.zeros(x.shape, device=device),
        bound_opts=bound_opts,
        device=device,
        verbose=False,
    )
    polytope.eval()
    
    x_L = torch.randn(x.shape, device=device)
    x_U = x_L + 0.01
    
    perturbation = PerturbationLpNorm(x_L=x_L, x_U=x_U)
    x = BoundedTensor(x_L, perturbation).to(device)
    
    # lb, _ = polytope.compute_bounds(x=(x,), method=method, bound_upper=False)
    
    lb, _, _ = polytope.init_alpha(x=(x,), bound_upper=False)
    print(lb.tolist())
    
if __name__ == "__main__":
    # test1()
    test2()