import random
import torch
import os

from util.misc.logger import logger

class MILPTightener:
    
    def __init__(self, abstractor, objectives):
        self.abstractor = abstractor
        self.input_lowers = objectives.lower_bounds[0].clone().view(self.abstractor.input_shape).to(self.abstractor.device)
        self.input_uppers = objectives.upper_bounds[0].clone().view(self.abstractor.input_shape).to(self.abstractor.device)
        self.c_to_use = objectives.cs.clone().transpose(0, 1).to(self.abstractor.device)
        
        assert len(abstractor.net.relus) == len(abstractor.net.perturbed_optimizable_activations), print('[!] Error: Support ReLU only', len(abstractor.net.relus), len(abstractor.net.perturbed_optimizable_activations))
        self.pre_relu_names = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.relu_names = {i: layer.name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.reset()
        
        
    def reset(self):
        self.black_list = []
        self.tightened_layers = []
        
        
    def select_layer(self):
        for l_id, l_name in self.pre_relu_names.items():
            if (l_id != 0) and (l_name not in self.tightened_layers):
                return l_name
        return random.choice(self.tightened_layers)
        
        
    # FIXME: only work with ReLU
    def __call__(self, domain_list, topk=64, largest=False, timeout=2.0):
        # step 1: select domains
        worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')        
        unified_lower_bounds = {k: v.min(dim=0).values.flatten() for k, v in worst_domains.lower_bounds.items()}
        unified_upper_bounds = {k: v.max(dim=0).values.flatten() for k, v in worst_domains.upper_bounds.items()}
        
        if os.environ.get('NEURALSAT_ASSERT'):
            assert all([(unified_lower_bounds[k] <= worst_domains.lower_bounds[k].flatten(1)).all() for k in worst_domains.lower_bounds])
            assert all([(unified_upper_bounds[k] >= worst_domains.upper_bounds[k].flatten(1)).all() for k in worst_domains.upper_bounds])
        assert len(self.pre_relu_names) == len(unified_lower_bounds)
        
        # step 2: select candidates
        unified_masks = {
            k: torch.where(unified_lower_bounds[k] * unified_upper_bounds[k] < 0)[0].numpy() 
                for k in unified_lower_bounds if k != self.pre_relu_names[0] # skip first layer
        } 
        unified_indices = [(k, vi) for k, v in unified_masks.items() for vi in v] 
        unified_scores = torch.concat([
            torch.min(unified_upper_bounds[k][v].abs(), unified_lower_bounds[k][v].abs()).flatten() 
                for k, v in unified_masks.items()
        ])
        assert unified_scores.numel() == len(unified_indices)
        
        if not len(unified_indices):
            return
        
        n_candidates = len(unified_indices)
        candidates = []
        selected_indices = unified_scores.topk(n_candidates, largest=largest).indices
        selected_layer = self.select_layer()
        # print('selected_layer:', selected_layer)
        
        for s_idx in selected_indices:
            l_id, n_id = unified_indices[s_idx]
            
            # 1st time, tighten only "selected_layer"
            if (l_id != selected_layer) and (selected_layer not in self.tightened_layers):
                continue
            
            var_name = f"lay{l_id}_{n_id}"
            if var_name in self.black_list:
                continue
            
            candidates.append((
                l_id, 
                n_id, 
                var_name, 
                self.pre_relu_names, 
                self.relu_names, 
                self.abstractor.net.final_name,
            ))
            
            # 1st time, tighten all neurons of "selected_layer"
            if selected_layer not in self.tightened_layers: 
                continue
            
            # 2nd time, tighten topk neurons
            if len(candidates) == topk: 
                break
        
        if selected_layer not in self.tightened_layers:
            self.tightened_layers.append(selected_layer)

        # step 3: rebuild mip model
        unified_bound_shapes = {k: v.size() for k, v in worst_domains.lower_bounds.items()}
        current_model = self.rebuild_mip_model(unified_lower_bounds, unified_upper_bounds, unified_bound_shapes)
        if current_model is None:
            return
        
        # step 4: stabilize
        unified_lower_bounds_refined, unified_upper_bounds_refined, num_neuron_refined, unstable_to_stable_neurons = self.abstractor.net.stabilize(
            mip_model=current_model,
            candidates=candidates,
            unified_lower_bounds=unified_lower_bounds,
            unified_upper_bounds=unified_upper_bounds,
            timeout=timeout,
        )
        logger.debug(f'[Stabilize ({timeout=})] layer="{selected_layer}", #candidates={len(candidates)}, #total={len(unified_indices)}, #tightened={num_neuron_refined}, #stabilized={len(unstable_to_stable_neurons)}, #blacklisted={len(self.black_list)}')
        
        # step 5: update domains bounds
        class TMP:
            pass
        
        if os.environ.get('NEURALSAT_ASSERT'):
            assert all([torch.all(unified_lower_bounds_refined[key] <= unified_upper_bounds_refined[key]) for key in unified_upper_bounds_refined])
                    
        refined_domain = TMP()
        refined_domain.lower_bounds = unified_lower_bounds_refined
        refined_domain.upper_bounds = unified_upper_bounds_refined
        domain_list.update_refined_bounds(refined_domain)
        return
        
        
    def rebuild_mip_model(self, refined_lower_bounds, refined_upper_bounds, bound_shapes):
        intermediate_layer_bounds = {}
        assert len(bound_shapes) == len(refined_lower_bounds) == len(refined_upper_bounds)
        
        for l_name, shape in bound_shapes.items():
            intermediate_layer_bounds[l_name] = [
                refined_lower_bounds[l_name].to(self.abstractor.device).view(1, *shape[1:]), 
                refined_upper_bounds[l_name].to(self.abstractor.device).view(1, *shape[1:])
            ]
        try:
            self.abstractor.build_lp_solver(
                model_type='mip', 
                input_lower=self.input_lowers,
                input_upper=self.input_uppers,
                c=self.c_to_use,
                refine=False,
                intermediate_layer_bounds=intermediate_layer_bounds,
            )
        except AttributeError:
            return None
        except:
            raise NotImplementedError()
        
        current_model = self.abstractor.net.model.copy()
        current_model.setParam('Threads', 1)
        current_model.setParam('MIPGap', 0.01)
        current_model.setParam('MIPGapAbs', 0.01)
        current_model.update()
        
        return current_model
    

