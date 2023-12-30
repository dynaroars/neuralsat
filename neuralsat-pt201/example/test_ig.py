# import torchvision.transforms as transforms
# from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from test import extract_instance
from heuristic.ig import *

def normalize(img):
    A = img.clone()
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0]
    return A

def print_selected_indices(upper, lower, indices, name=None):
    print(f'[{name}] Selected indices:', indices)
    for ind in indices:
        print(f'\t + lower[{ind}]={lower.flatten()[ind]:.03f} \t upper[{ind}]={upper.flatten()[ind]:.03f} \t range[{ind}]={(upper.flatten()[ind]-lower.flatten()[ind]):.03f}')
    print()

if __name__ == "__main__":
    # path = 'example/doberman.png'

    # net_path = 'example/cifar10_2_255_simplified.onnx'
    # net_path = '../benchmark/vnncomp23/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx'
    net_path = '../benchmark/vnncomp23/yolo/onnx/TinyYOLO.onnx'
    
    # vnnlib_path = Path('example/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib')
    # vnnlib_path = Path('../benchmark/vnncomp23/acasxu/vnnlib/prop_3.vnnlib')
    vnnlib_path = Path('../benchmark/vnncomp23/yolo/vnnlib/TinyYOLO_prop_000246_eps_1_255.vnnlib')
    
    device = 'cpu'
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    objective = objectives.pop(1)
    
    input_lowers = objective.lower_bounds.view(input_shape)
    input_uppers = objective.upper_bounds.view(input_shape)
    print(input_lowers.shape)
    
    random_input = (input_uppers - input_lowers) * torch.rand(input_shape, device=device) + input_lowers
    assert torch.all(random_input <= input_uppers)
    assert torch.all(random_input >= input_lowers)
    
    ig = IntegratedGradient(model, device=device)
    attr, label = ig.attribute(random_input)
    
    
    indices = (input_uppers - input_lowers)[0].flatten().topk(5, 0).indices
    print_selected_indices(input_uppers, input_lowers, indices, name='diff')
    
    indices = attr[0].flatten().topk(5, 0).indices
    print_selected_indices(input_uppers, input_lowers, indices, name='ig')
    
    scaled_attr = scale_saliency_map_np(attr[0].detach().cpu().numpy(), dim=0)
    print(scaled_attr.shape)
    indices = torch.tensor(scaled_attr).flatten().topk(5, 0).indices
    print_selected_indices(input_uppers, input_lowers, indices, name='scaled ig')

    if 1:
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(random_input.permute(0, 2, 3, 1)[0].detach().cpu().numpy())
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Image')
        
        norm_attr = normalize(attr)
        axes[1].imshow(norm_attr.permute(0, 2, 3, 1)[0].detach().cpu().numpy(), cmap='RdBu_r')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title('IG')
        
        
        axes[2].imshow(scaled_attr, cmap='RdBu_r')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_title('Scaled IG')
        
        fig.tight_layout()
        plt.savefig('example/output.png')
    
    # img = Image.open(path).convert('RGB')
    # img_norm = transform(img)
    # print(img_norm.shape)
  
    # ig_attr, label = ig.attribute(img_norm)