import torch.nn as nn
import torch
import time
import copy
import math
import tqdm

from helper.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from abstractor.auto_LiRPA.perturbations import PerturbationLpNorm
from helper.spec.objective import DnfObjectives, Objective
from abstractor.params import get_initialize_opt_params
from abstractor.auto_LiRPA import BoundedTensor, BoundedModule
from helper.spec.read_vnnlib import read_vnnlib
from helper.network.read_pth import parse_pth
from setting import Settings

class PytorchWrapper(nn.Module):

    def __init__(self, module_lists):
        super(PytorchWrapper, self).__init__()
        self.layers = module_lists
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecomposeHeuristic:

    def __init__(self, input_shape) -> None:
        self.input_shape = input_shape
        
        
    def try_subnetwork(self, subnet, input_lb, input_ub, device, verbose=False):
        # print(f'{input_lb.shape=} {input_lb.device=} {device=}')
        new_x = BoundedTensor(input_lb, PerturbationLpNorm(x_L=input_lb, x_U=input_ub)).to(device)
            
        out_shape = subnet(input_lb).size()
        assert out_shape[0] == 1
        
        if len(out_shape) > 2:
            tmp_net = torch.nn.Sequential(subnet, torch.nn.Flatten(1))
            output_batch = Settings.sequential_batch 
        else:
            tmp_net = subnet
            output_batch = 1

        n_outputs = math.prod(out_shape)
        
        abstract = BoundedModule(
            model=copy.deepcopy(tmp_net).to(device), 
            global_input=torch.zeros(input_lb.shape, device=device),
            bound_opts={'conv_mode': 'patches', 'verbosity': 0, **Settings.verify_extra_opts},
            device=device,
            verbose=False,
        )
        
        abstract.set_bound_opts(get_initialize_opt_params(lambda x: False))
        
        
        output_lowers, output_uppers = [], []
        # print(f'\t- Backward bound {input_lb.shape=} {out_shape=}')
        if verbose:
            pbar = tqdm.tqdm(range(0, n_outputs, output_batch), desc=f'Compute bounds: {output_batch=}')
        else:
            pbar = range(0, n_outputs, output_batch)
        for i_, i in enumerate(pbar):
            if (i_ % 50 == 0) or i_ == len(pbar) - 1:
                
                indices = torch.arange(i, min(i+output_batch, n_outputs))
                cs = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(new_x)
                
                l, u, aux_reference_bounds = abstract.init_alpha(
                    x=(new_x,), 
                    share_alphas=Settings.share_alphas, 
                    c=cs, 
                    bound_lower=True,
                    bound_upper=True,
                )
                
                # print(f'\t\t- {l.shape=}')
                # assert not l.isnan().any()
                
                if Settings.init_abstraction_method == 'crown-optimized':
                    print('\t- Optimized bound')
                    l, _ = abstract.compute_bounds(
                        x=(new_x,), 
                        C=cs,
                        method='crown-optimized',
                        aux_reference_bounds=aux_reference_bounds, 
                        bound_lower=True, 
                        bound_upper=False, 
                    )
                    
                    _, u = abstract.compute_bounds(
                        x=(new_x,), 
                        C=cs,
                        method='crown-optimized',
                        aux_reference_bounds=aux_reference_bounds, 
                        bound_lower=False, 
                        bound_upper=True, 
                    )
                
            output_lowers.append(l.detach().cpu().clone())
            output_uppers.append(u.detach().cpu().clone())
            
        output_lowers = torch.cat(output_lowers, dim=-1)
        output_uppers = torch.cat(output_uppers, dim=-1)
        return output_lowers.view(out_shape).detach().cpu(), output_uppers.view(out_shape).detach().cpu()
            
    def decompose(self, original_network, objective):
        assert isinstance(original_network.layers, nn.ModuleList)

        n_layers = len(original_network.layers)

        if n_layers == 2:
            return [PytorchWrapper(original_network.layers[i:i+1]).cpu() for i in range(n_layers)]

        device = 'cuda'
        subnetworks = []
        start_idx = 0
        in_shape = self.input_shape

        input_lb = objective.lower_bounds.view(in_shape).cpu()
        input_ub = objective.upper_bounds.view(in_shape).cpu()
        
        while start_idx < n_layers:
            max_valid_size = 1
            last_valid_subnet = None
            gc_cuda()
            
            while start_idx + max_valid_size <= n_layers:
                print(f'- Trying {start_idx=} {max_valid_size=} ... ', end='')
                
                subnet = PytorchWrapper(original_network.layers[start_idx:start_idx+max_valid_size]).cpu()
                # print(subnet)
                
                try:
                    l, u = self.try_subnetwork(
                        subnet=subnet, 
                        input_lb=input_lb,
                        input_ub=input_ub, 
                        device=device,
                    )
                except RuntimeError as exception:
                    if is_cuda_out_of_memory(exception):
                        # print(f'[!] Got OOM for {start_idx=} {max_valid_size=}')
                        print('Failed')
                        gc_cuda()
                        break
                    else:
                        raise
                except:
                    raise
                else:
                    last_valid_subnet = subnet
                    last_valid_size = max_valid_size
                    max_valid_size += 1
                    last_lb = l.cpu().clone()
                    last_ub = u.cpu().clone()
                    gc_cuda()
                    print('Passed')

            assert last_valid_subnet is not None
            
            print(f'\n[+] Saved {start_idx=} {last_valid_size=}\n')
            last_valid_subnet.cpu()
            subnetworks.append(last_valid_subnet)
            in_shape = last_valid_subnet(torch.randn(in_shape)).size()
            input_lb = last_lb.clone().detach().cpu()
            input_ub = last_ub.clone().detach().cpu()

            start_idx += last_valid_size


        return subnetworks
    
    
if __name__ == "__main__":
    path = '../../../benchmarks/resnet/resnet12/net/resnet12.pth'
    spec = '../../../benchmarks/resnet/resnet12/spec/spec_idx_0_net_resnet12_eps_0.020000_seed_37.vnnlib'
    
    # path = '../../../benchmarks/resnet/resnet36/net/resnet36BN.pth'
    # spec = '../../../benchmarks/resnet/resnet36/spec/spec_idx_2_net_resnet36BN_eps_0.001000_seed_36.vnnlib'
    
    # path = '../../../benchmarks/resnet/resnet18/net/resnet18.pth'
    # path = '../../../benchmarks/vae/vae_base/net/vae_base.pth'
    model, input_shape, _, is_nhwc = parse_pth(path)
    vnnlibs = read_vnnlib(spec)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    dnf_objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=input_shape, 
        is_nhwc=is_nhwc,
    )
    
    objective = dnf_objectives.pop(1)
    
    Settings.setup_resnet_large(None)
    
    print(model)
    
    print(Settings)

    heuristic = DecomposeHeuristic(input_shape)
    
    
    start = time.time()
    subnets = heuristic.decompose(model, objective)
    print([len(s.layers) for s in subnets], time.time() - start, 'seconds')