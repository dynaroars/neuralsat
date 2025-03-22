import torch

from abstractor.auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from test import extract_instance

def test1():
    onnx_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    device = 'cpu'
    bound_opts = {'conv_mode': 'patches', 'verbosity': 0}
    method = 'backward'
    
    model, input_shape, objectives = extract_instance(onnx_path, vnnlib_path)

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
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)
    
    lb, _ = polytope.compute_bounds(x=(x,), method=method, bound_upper=False)
    
    print(f'{lb=}')
    
    polytope.visualize('graph')
    

if __name__ == "__main__":
    test1()