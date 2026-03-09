import torch

from abstractor.auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from helper.network.read_onnx import _parse_onnx
from test import extract_instance

def test_bound(onnx_path, device):
    
    model, input_shape, output_shape = _parse_onnx(onnx_path)
    print(model, input_shape, output_shape)

    polytope = BoundedModule(
        model=model, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
        device=device,
        verbose=True,
    )
    polytope.eval()
    # polytope.visualize('example/scripts/graph')
    
    x_L = torch.randn(input_shape)
    x_U = x_L + 0.05
    x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)
    lb, ub = polytope.compute_bounds(x=(x,), method='backward', bound_upper=True)
    print(f'{lb=}')
    print(f'{ub=}')
    
def test1():
    # onnx_path = 'example/onnx/mnist-net_256x2.onnx'
    # vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    
    
    onnx_path = 'example/onnx/vit.onnx'
    vnnlib_path = 'example/vnnlib/spec_vit.vnnlib'
    
    device = 'cpu'
    method = 'backward'
    extra_opts = {'sparse_intermediate_bounds': False, 'conv_mode': 'matrix'}
    # extra_opts = {}
    bound_opts = {'conv_mode': 'patches', 'verbosity': 0, **extra_opts}
    print(f'{bound_opts=}')
    model, input_shape, objectives = extract_instance(onnx_path, vnnlib_path)
    # print(model)
    

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
    
    # lb, _ = polytope.compute_bounds(x=(x,), method=method, bound_upper=False)
    
    polytope(x)
    exit()
    lb, _, _ = polytope.init_alpha(x=(x,), bound_upper=False)
    
    print(f'{lb=}')
    
    polytope.visualize('example/scripts/graph')
    

def test2():
    # onnx_path = '/home/roars/submissions/verify_structural_robustness/data/rotate_layer.onnx'
    # onnx_path = '/home/roars/submissions/verify_structural_robustness/data/deform_layer.onnx'
    onnx_path = '/home/roars/submissions/verify_structural_robustness/data/lightness_layer.onnx'
    device = 'cuda'
    test_bound(onnx_path, device)
    
if __name__ == "__main__":
    test2()