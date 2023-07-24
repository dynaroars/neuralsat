import torch
import copy
import time

from setting import Settings

def print_tightened_bounds(name, olds, news):
    old_lowers, old_uppers = olds
    new_lowers, new_uppers = news
    
    old_lowers = old_lowers.flatten().detach().cpu()#.numpy()
    old_uppers = old_uppers.flatten().detach().cpu()#.numpy()
    
    new_lowers = new_lowers.flatten().detach().cpu()#.numpy()
    new_uppers = new_uppers.flatten().detach().cpu()#.numpy()
    
    print(f'[+] Layer: {name}')
    for i in range(len(old_lowers)):
        if (new_lowers[i] - old_lowers[i]).abs() > 1e-4 or (new_uppers[i] - old_uppers[i]).abs() > 1e-4:
            print(f'\t- neuron {i}: [{old_lowers[i]:.04f}, {old_uppers[i]:.04f}] => [{new_lowers[i]:.04f}, {new_uppers[i]:.04f}]')

class Tightener:
    
    def __init__(self, abstractor):
        self.abstractor = abstractor
        
        
    def __call__(self, domain_params):
        if self.abstractor.input_split:
            return domain_params
        
        if not Settings.use_mip_refine_domain_bounds:
            return domain_params
        
        assert len(self.abstractor.pre_relu_indices) == len(domain_params.lower_bounds) - 1, print('Support ReLU only')
        
        remaining_index = torch.where((domain_params.output_lbs.detach().cpu() <= domain_params.rhs.detach().cpu()).all(1))[0]
        for idx in remaining_index:
            cur_intermediate_layer_bounds = {
                self.abstractor.name_dict[d]: [
                    domain_params.lower_bounds[d][idx][None].clone(),
                    domain_params.upper_bounds[d][idx][None].clone(),
                ] for d in self.abstractor.pre_relu_indices # exclude output layer
            } 
            cur_input_lowers = domain_params.input_lowers[idx][None]
            cur_input_uppers = domain_params.input_uppers[idx][None]
                
            # tic = time.time()
            self.abstractor.build_lp_solver(
                model_type='mip', 
                input_lower=cur_input_lowers, 
                input_upper=cur_input_uppers, 
                c=None, 
                # intermediate_layer_bounds=copy.deepcopy(cur_intermediate_layer_bounds),
                intermediate_layer_bounds=cur_intermediate_layer_bounds,
            )
            # print(idx, 'refine in:', time.time() - tic)
                
            new_intermediate_layer_bounds = self.abstractor.net.get_refined_intermediate_bounds()
            
            # for k in new_intermediate_layer_bounds:
            #     old_bounds = cur_intermediate_layer_bounds[k]
            #     new_bounds = new_intermediate_layer_bounds[k]
            #     print_tightened_bounds(name=k, olds=old_bounds, news=new_bounds)
            
            for i_ in self.abstractor.pre_relu_indices:
                new_l, new_u = new_intermediate_layer_bounds[self.abstractor.name_dict[i_]]
                domain_params.lower_bounds[i_][idx] = new_l.clone()
                domain_params.upper_bounds[i_][idx] = new_u.clone()
            
        return domain_params