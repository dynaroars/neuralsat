import random
import torch
import tqdm
import copy
import os


from .utils import optimize_dnn, optimize_dnn_2, filter_dnf_pairs, verify_dnf_pairs
from util.network.read_onnx import parse_onnx, decompose_pytorch

from setting import Settings

def setup_gpu_tightener_settings():
    config = {
        'use_restart': Settings.use_restart,
        'use_mip_tightening': Settings.use_mip_tightening,
        'use_gpu_tightening': Settings.use_gpu_tightening,
    }
    Settings.use_restart = 0
    Settings.use_mip_tightening = 0
    Settings.use_gpu_tightening = 0
    return config

def reset_gpu_tightener_settings(config):
    for k, v in config.items():
        setattr(Settings, k, v)
    
def print_diff_bounds(lowers, uppers, name):
    diff = sum([(uppers[k] - lowers[k]).sum() for k in lowers])
    print(f'[{name}] {diff=}')

def print_under_bounds(best_interm_bounds):
    print('Best intermediate bounds so far:')
    for (layer_idx, bounds) in best_interm_bounds.items():
        print(f'\t- {layer_idx=} min={bounds[0].sum().item():.04f} max={bounds[1].sum().item():.04f}')
    print()
    
class GPUTightener:
    
    def __init__(self, verifier, abstractor, batch=100):
        assert not hasattr(verifier, "other")
        self.orig_verifier = copy.deepcopy(verifier)
        self.orig_verifier.batch = batch
        self.orig_net = copy.deepcopy(verifier.net).to(verifier.device)
        self.orig_abstractor = abstractor

        self.pre_activations = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.input_shape = verifier.input_shape
        self.device = verifier.device
        
        self._setup_sub_networks()
        self._setup_sub_verifiers()
        self.reset()
        
        
    def reset(self):
        self.best_interm_bounds = {}
        self.tightened_layers = []
        
        
    def _setup_sub_networks(self):
        split_idx = 1
        self.sub_networks = {}
        while True:
            prefix_onnx_byte, _ = decompose_pytorch(self.orig_net, self.orig_verifier.input_shape, split_idx + 1)
            if prefix_onnx_byte is None:
                return
            # parse subnet
            prefix, _, output_shape, _ = parse_onnx(prefix_onnx_byte)
            
            # flatten output
            if len(output_shape) > 2:
                prefix = torch.nn.Sequential(prefix, torch.nn.Flatten(1))
                
            self.sub_networks[split_idx] = prefix.to(self.device)
            print(f'{split_idx = }')
            print(f'{prefix = }')
            print(f'{output_shape = }')
            print()
            split_idx += 1
        
    def _setup_sub_verifiers(self):
        self.sub_verifiers = {}
        for layer_id, subnet in self.sub_networks.items():
            # print(layer_id, subnet)
            sub_verifier = copy.deepcopy(self.orig_verifier)
            sub_verifier.net = subnet
            sub_verifier._setup_restart_naive(0, None)
            self.sub_verifiers[layer_id] = sub_verifier

        
    @torch.no_grad()
    def _sampling_random(self, layer_idx, lower_bounds, upper_bounds, n_sample=1000):
        x = (upper_bounds - lower_bounds) * torch.rand(n_sample, *self.input_shape[1:], device=self.device) + lower_bounds
        net = self.orig_net.to(self.device)
        _, outputs = net(x, return_interm=True)
        bounds = (outputs[layer_idx].min(0).values.flatten(), outputs[layer_idx].max(0).values.flatten())
        return bounds
    
    
    def _sampling_gradient(self, layer_idx, lower_bounds, upper_bounds):
        min_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=True)
        max_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=False)
        bounds = (min_i.flatten(), max_i.flatten())
        return bounds
        
        
    def _sampling_gradient_2(self, layer_idx, lower_bounds, upper_bounds):
        min_i = optimize_dnn_2(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=True)
        max_i = optimize_dnn_2(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=False)
        bounds = (min_i.flatten(), max_i.flatten())
        return bounds
    
    
    def sampling(self, layer_idx, input_lower, input_upper):
        (interms_gradient_min, interms_gradient_max) = self._sampling_gradient(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        (interms_gradient_min_2, interms_gradient_max_2) = self._sampling_gradient_2(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        (interms_random_min, interms_random_max) = self._sampling_random(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        best_min = torch.min(torch.stack([interms_random_min, interms_gradient_min, interms_gradient_min_2]), dim=0).values
        best_max = torch.max(torch.stack([interms_random_max, interms_gradient_max, interms_gradient_max_2]), dim=0).values
        return [best_min, best_max]
        
        
    def select_layer(self):
        for l_id, l_name in self.pre_activations.items():
            if (l_id != 0) and ((l_id, l_name) not in self.tightened_layers):
                return l_id, l_name
        print('[!] Select layer randomly')
        return random.choice(self.tightened_layers)
        
    
    def __call__(self, domain_list, topk, iteration=1):
        # backup settings
        setting_config = setup_gpu_tightener_settings()
        
        # stabilize
        for _ in range(iteration):
            for layer_idx in range(1, len(self.pre_activations)):
                layer_name = self.pre_activations[layer_idx]
                print(f'[Stabilize] {layer_idx=}, {layer_name=}')
                
                # step 1: get domains to stabilize
                worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')     
                
                # step 2: stabilize
                refined_lower_bounds, refined_upper_bounds = self.stabilize(
                    domain_params=worst_domains, 
                    layer_idx=layer_idx,
                    topk=topk,
                ) 
                
                if refined_lower_bounds is None or refined_upper_bounds is None:
                    continue
                
                # step 2: update bounds
                class TMP:
                    pass        
                refined_domain = TMP()
                refined_domain.lower_bounds = refined_lower_bounds
                refined_domain.upper_bounds = refined_upper_bounds
                domain_list.update_refined_bounds(refined_domain)
        
        # restore settings
        reset_gpu_tightener_settings(setting_config)
        return
        
    
    @torch.no_grad()
    def _update_best_interm_bounds_by_samples(self, layer_idx, samples):
        if not len(samples):
            return
        assert layer_idx in self.best_interm_bounds    
        model = self.sub_networks[layer_idx]
        output = model(samples)
        min_output = output.amin(0)
        max_output = output.amax(0)
        
        self.best_interm_bounds[layer_idx][0] = torch.where(
            self.best_interm_bounds[layer_idx][0] < min_output, 
            self.best_interm_bounds[layer_idx][0], 
            min_output
        )
        
        self.best_interm_bounds[layer_idx][1] = torch.where(
            self.best_interm_bounds[layer_idx][1] > max_output, 
            self.best_interm_bounds[layer_idx][1], 
            max_output
        )
  
                
    def extract_tightening_neurons(self, layer_idx, lower_bounds, upper_bounds, topk=1000, eps=1e-6):
        # positive_neurons, negative_neurons = [], []
        assert eps >= 0
        candidates = []
        extended_candidates = []
        assert layer_idx in self.best_interm_bounds
        best_interm_min, best_interm_max = self.best_interm_bounds[layer_idx]
        # print(f'{best_interm_min.shape = }')
        for i in range(len(best_interm_min)):
            best_min = best_interm_min[i].item()
            best_max = best_interm_max[i].item()
            if (lower_bounds[i] * upper_bounds[i] < 0) and (best_min * best_max > 0): # possible stabilized neurons
                if best_min >= 0: # positive neuron
                    candidates.append([(i, 0.0 + eps, 'lt')])
                elif best_max <= 0: # negative neuron
                    candidates.append([(i, 0.0 - eps, 'gt')])
            else:
                threshold = 5e-2 * 2
                diff_min = abs(best_min - lower_bounds[i])
                if diff_min >= threshold:
                    extended_candidates.append([diff_min, (i, (best_min + lower_bounds[i]) / 2 + eps, 'lt')])
                    # print(f'\t- Added {i=} {best_min=} {lower_bounds[i]=} {(best_min + lower_bounds[i]) / 2 =}')
                diff_max = abs(best_max - upper_bounds[i])
                if diff_max >= threshold:
                    extended_candidates.append([diff_max, (i, (best_max + upper_bounds[i]) / 2 - eps, 'gt')])
                    # print(f'\t- Added {i=} {best_max=} {upper_bounds[i]=} {(best_max + upper_bounds[i]) / 2 =}')
                    
        if len(extended_candidates):
            if len(candidates) < topk:
                extended_candidates = sorted(extended_candidates, key=lambda x: x[0], reverse=True) # sort by scores
                extended_candidates = [[_[1]] for _ in extended_candidates] # remove scores
                extended_candidates = extended_candidates[:(topk-len(candidates))]
                candidates = candidates + extended_candidates

        # return candidates[:topk] # TODO: remove
        return candidates
        
        
    def falsify_layer(self, layer_idx, input_lower, input_upper, lower_bounds, upper_bounds, batch=20, iteration=5, topk=1000):
        for _ in range(iteration):
            # step 1: find possible stabilized neurons
            candidate_neurons = self.extract_tightening_neurons(
                layer_idx=layer_idx,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                topk=topk*2,
            )
            if not len(candidate_neurons):
                return
            
            # step 2: falsify intermediate properties
            attack_samples = []
            pbar = tqdm.tqdm(range(0, len(candidate_neurons), batch), desc=f'[{_+1}/{iteration}] Falsifying {layer_idx=} {len(candidate_neurons)=}')
            for batch_id in pbar:
                # print(f'{batch_id=} {candidate_neurons[batch_id:batch_id+batch]=}')
                _, adv = filter_dnf_pairs(
                    model=self.sub_networks[layer_idx],
                    input_lower=input_lower,
                    input_upper=input_upper,
                    n_outputs=len(lower_bounds),
                    candidate_neurons=candidate_neurons[batch_id:batch_id+batch],
                    n_iterations=10, # TODO: update
                    patient_limit=2,
                )
                if len(adv):
                    attack_samples.append(adv)

                pbar.set_postfix(attack_samples=len(attack_samples))
                
            if not len(attack_samples):
                return

            # step 3: update best sampling bounds to avoid select wrong neurons in the future
            attack_samples = torch.vstack(attack_samples)
            self._update_best_interm_bounds_by_samples(
                layer_idx=layer_idx, 
                samples=attack_samples,
            )
            
            if len(attack_samples) / len(candidate_neurons) < 0.1:
                return
        
        
    def stabilize(self, domain_params, layer_idx, topk):
        print(f'Layer {layer_idx}, {self.pre_activations[layer_idx]}')
        assert domain_params.input_lowers is not None and domain_params.input_uppers is not None
        assert domain_params.lower_bounds is not None and domain_params.upper_bounds is not None
        
        # step 1: extract unified bounds
        unified_lower_bounds = {k: v.min(dim=0).values.flatten() for k, v in domain_params.lower_bounds.items()}
        unified_upper_bounds = {k: v.max(dim=0).values.flatten() for k, v in domain_params.upper_bounds.items()}  
        hidden_shapes = {k: (1, *v.shape[1:]) for k, v in domain_params.upper_bounds.items()}
        # print_diff_bounds(worst_domains.lower_bounds, worst_domains.upper_bounds, f'Before Stabilize {layer_idx=}')

        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(domain_params.input_lowers <= domain_params.input_uppers)
            assert torch.allclose(domain_params.input_lowers.mean(0), domain_params.input_lowers[0]), \
                print(domain_params.input_lowers.mean(0).sum(), domain_params.input_lowers[0].sum())
            assert torch.allclose(domain_params.input_uppers.mean(0), domain_params.input_uppers[0]), \
                print(domain_params.input_uppers.mean(0).sum(), domain_params.input_uppers[0].sum())
        
        # TODO: generalize for different input intervals
        input_lower = domain_params.input_lowers[0:1].to(self.device)
        input_upper = domain_params.input_uppers[0:1].to(self.device)
        
        # step 2: build intermediate properties
        if layer_idx not in self.best_interm_bounds:
            self.best_interm_bounds[layer_idx] = self.sampling(
                layer_idx=layer_idx,
                input_lower=input_lower,
                input_upper=input_upper
            )
            
        # step 3: falsify intermediate properties
        self.falsify_layer(
            layer_idx=layer_idx,
            input_lower=input_lower,
            input_upper=input_upper,
            lower_bounds=unified_lower_bounds[self.pre_activations[layer_idx]],
            upper_bounds=unified_upper_bounds[self.pre_activations[layer_idx]],
            batch=32,
            iteration=2, # TODO: update
            topk=topk,
        )
        
        # step 4: find possible stabilized neurons
        candidate_neurons = self.extract_tightening_neurons(
            layer_idx=layer_idx,
            lower_bounds=unified_lower_bounds[self.pre_activations[layer_idx]],
            upper_bounds=unified_upper_bounds[self.pre_activations[layer_idx]],
            topk=topk,
        )

        print(f'{len(candidate_neurons) = }')
        if not len(candidate_neurons):
            return None, None
        
        # step 5: verify intermediate properties
        pre_act_names = [self.pre_activations[k] for k in range(layer_idx)]
        reference_bounds = {
            name: [
                unified_lower_bounds[name].view(hidden_shapes[name]).to(self.device), 
                unified_upper_bounds[name].view(hidden_shapes[name]).to(self.device)
            ] 
            for name in pre_act_names
        }

        verified_candidates, attack_samples = verify_dnf_pairs(
            verifier=self.sub_verifiers[layer_idx],
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=len(unified_lower_bounds[self.pre_activations[layer_idx]]),
            candidate_neurons=candidate_neurons,
            batch=10,
            timeout=Settings.gpu_tightening_timeout,
            reference_bounds=reference_bounds,
        )
        
        # update best bounds to avoid select wrong neurons in the future
        self._update_best_interm_bounds_by_samples(
            layer_idx=layer_idx, 
            samples=attack_samples,
        )
        
        if not len(verified_candidates):
            return None, None
        
        # step 6: update improved bounds
        unified_lower_bounds_refined = {k: v.clone() for k, v in unified_lower_bounds.items()}
        unified_upper_bounds_refined = {k: v.clone() for k, v in unified_upper_bounds.items()}
        improved_neuron_indices = []
        for (neuron_idx, neuron_bound, neuron_direction) in verified_candidates:
            assert neuron_direction in ['lt', 'gt']
            layer_name = self.pre_activations[layer_idx]
            if neuron_direction == 'lt': # lower bound
                unified_lower_bounds_refined[layer_name][neuron_idx] = max(unified_lower_bounds_refined[layer_name][neuron_idx], neuron_bound)
            else: # upper bound
                unified_upper_bounds_refined[layer_name][neuron_idx] = min(unified_upper_bounds_refined[layer_name][neuron_idx], neuron_bound)
            improved_neuron_indices.append(neuron_idx)
                
        if os.environ.get('NEURALSAT_DEBUG'):
            for neuron_idx in list(sorted(set(improved_neuron_indices))):
                layer_name = self.pre_activations[layer_idx]
                print(f'Tightened {layer_name=} ({neuron_idx=}):\t[{unified_lower_bounds[layer_name][neuron_idx]:.04f}, {unified_upper_bounds[layer_name][neuron_idx]:.04f}]\t=>\t[{unified_lower_bounds_refined[layer_name][neuron_idx]:.04f}, {unified_upper_bounds_refined[layer_name][neuron_idx]:.04f}]')

        if os.environ.get('NEURALSAT_ASSERT'):
            assert all([torch.all(unified_lower_bounds_refined[key] <= unified_upper_bounds_refined[key]) for key in unified_upper_bounds_refined])
        
        return unified_lower_bounds_refined, unified_upper_bounds_refined