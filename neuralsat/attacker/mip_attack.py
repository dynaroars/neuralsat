from sortedcontainers import SortedList
import gurobipy as grb
import numpy as np
import random
import torch
import time
import math

from util.misc.result import AbstractResults
from util.misc.check import check_solution
from heuristic.util import compute_masks

multiprocess_mip_attack_model = None
multiprocess_stop = False


class CounterExample:
    
    "A single example"

    def __init__(self, x, obj, pattern):
        self.x = x
        self.obj = obj  # smaller obj is better
        self.activation_pattern = pattern

    def __lt__(self, other):
        return self.obj < other.obj

    def __le__(self, other):
        return self.obj <= other.obj

    def __eq__(self, other):
        return self.obj == other.obj


class CounterExamplePool:
    
    def __init__(self, abstractor, objectives, capacity):
        self.net = abstractor.pytorch_model
        self.objectives = objectives
        self.input_shape = abstractor.input_shape
        self.device = abstractor.device
        
        self.cs = objectives.cs.transpose(0, 1).cpu().clone() # [1, #props, #outputs]
        
        # maximum number of cex in pool
        self.pool = SortedList()
        self.capacity = capacity
        self.abstractor = abstractor
        
        # init pool
        self.threshold = None
        self._initialize()


    def _initialize(self):
        input_lowers = self.objectives.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = self.objectives.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)

        repeat = self.capacity // len(input_lowers) + 1 if (self.capacity % len(input_lowers)) else self.capacity // len(input_lowers)
        repeat *= 3
        
        input_lowers_repeat = input_lowers.repeat(repeat, *[1]*(len(self.input_shape) - 1))
        input_uppers_repeat = input_uppers.repeat(repeat, *[1]*(len(self.input_shape) - 1))

        # seed = random.randint(1, 100000)
        # seed = 28998
        # print('seed:', seed)
        # torch.manual_seed(seed)
        adv_example = (input_uppers_repeat - input_lowers_repeat) * torch.rand(input_lowers_repeat.shape, device=self.device) + input_lowers_repeat
        # adv_example = torch.clamp(torch.clamp(torch.randn(input_lowers_repeat.shape, device=self.device), max=input_uppers_repeat), min=input_lowers_repeat)
        assert (adv_example >= input_lowers_repeat).all() and (adv_example <= input_uppers_repeat).all()
        # print(adv_example.sum().item())
        self.add(adv_example)
        # exit()
        
    @torch.no_grad()
    def add(self, inputs):
        assert len(inputs) > 0
        # reset_perturbed_nodes must be False otherwise the .perturbed property will be missing.
        pred = self.abstractor.net(inputs).cpu()
        # FIXME: add rhs
        pred = pred.matmul(self.cs[0].transpose(-1, -2)).min(-1).values
        
        activations = [None] * len(self.abstractor.net.relus)
        for layer_i, layer in enumerate(self.abstractor.net.relus):
            activations[layer_i] = (layer.inputs[0].forward_value.flatten(1) > 0).int().cpu()
        
        for batch_idx in range(len(inputs)):
            activation_i = [_[batch_idx] for _ in activations]
            self._add_one(CounterExample(inputs[batch_idx], pred[batch_idx].item(), activation_i))
            
        
    def _add_one(self, cex):
        if len(self.pool) >= self.capacity:
            if cex.obj >= self.pool[-1].obj:
                # skip if cex is worse than the worst example in pool
                print('skip', cex.obj, self.pool[-1].obj)
                return
            
            pop_idx = -1
            if 1:
                current_patterns = torch.stack([torch.cat([ii.flatten() for ii in i.activation_pattern]) for i in self.pool])
                this_pattern = torch.cat([ii.flatten() for ii in cex.activation_pattern]).view(1, -1)
                # print(current_patterns)
                # print(this_pattern)
                
                diff = torch.cdist(current_patterns.float(), this_pattern.float(), p=0).flatten()
                min_idx = diff.argmin()
                
                if self.threshold is None:
                    # we set the threshold as the lowest diff when the first time we need to filter
                    self.threshold = diff[min_idx] if diff[min_idx] > 1.0 else 1.0
                        
                if diff[min_idx] < self.threshold and cex.obj < self.pool[min_idx].obj:
                    pop_idx = min_idx.item()
                        
            # print('remove idx:', pop_idx)
            # assert pop_idx == -1
            self.pool.pop(pop_idx)
            
        self.pool.add(cex)
        
    
    def get_common_pattern(self, prob_threshold=0.5):
        selected_advs = self.pool
        pos_threshold = min(int(math.ceil(prob_threshold * len(selected_advs))), len(selected_advs))
        neg_threshold = len(selected_advs) - pos_threshold
        
        # collect all activation patterns
        all_patterns = [[] for i in range(len(self.abstractor.net.relus))]
        for adv in selected_advs:
            for layer_i in range(len(all_patterns)):
                all_patterns[layer_i].append(adv.activation_pattern[layer_i])
        # print(all_patterns)
        
        # concat activation patterns across examples
        ret = [None] * len(self.abstractor.net.relus)
        for layer_i in range(len(all_patterns)):
            layer_i_pattern = torch.stack(all_patterns[layer_i], dim=0)
            # print(layer_i, layer_i_pattern.shape)
            acc_pattern = layer_i_pattern.sum(dim=0)
            # print(layer_i, acc_pattern)
            # +1 for most neurons active, -1 for most neurons inactive, 0 for not selected
            ret[layer_i] = ((acc_pattern >= pos_threshold).int() - (acc_pattern < neg_threshold).int())
        
        return ret
        
    def get_uncommon_pattern(self, prob_threshold=0.5):
        pass

    def __len__(self):
        return len(self.pool)

def mip_solver_worker(candidate, n_inputs):
    global multiprocess_stop
    if multiprocess_stop:
        return None
    
    # print('[!] Attacking:', candidate)
    tmp_model = multiprocess_mip_attack_model.copy()
    
    v = tmp_model.getVarByName(candidate)
    vlb = out_lb = v.lb
    vub = out_ub = v.ub
    adv = None
    
    tmp_model.setObjective(v, grb.GRB.MINIMIZE)
    tmp_model.update()
    try:
        tmp_model.optimize()
    except grb.GurobiError as e:
        print(f'Gurobi error: {e.message}')
        return None

    vlb = max(tmp_model.objbound, out_lb)
    if tmp_model.solcount > 0:
        vub = min(tmp_model.objval, out_ub)
    if vub < 0:
        input_vars = [tmp_model.getVarByName(f'inp_{dim}') for dim in range(n_inputs)]
        adv = [var.X for var in input_vars]
        multiprocess_stop = True
    return adv    
    
    
class MIPAttacker:

    def __init__(self, abstractor, objectives):
        self.objectives = objectives
        assert self.objectives.cs.shape[1] == 1 # c shape: [#props, 1, #outputs]

        self.net = abstractor.pytorch_model
        self.input_shape = abstractor.input_shape
        self.device = abstractor.device
        self.output_names = [v.VarName for v in abstractor.net[abstractor.net.final_name].solver_vars]
                
        assert abstractor.net.model.ModelName == 'mip', print(f'Model error: "{abstractor.net.model.ModelName}" != "mip"')
        self.mip_model = abstractor.net.model.copy()
        self.mip_model.setParam('BestBdStop', 1e-5)  # Terminiate as long as we find a positive lower bound.
        self.mip_model.setParam('BestObjStop', -1e-5)  # Terminiate as long as we find a adversarial example.
        
        self.abstractor = abstractor
        
        
    def manual_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        
    def run(self, reference_bounds=None, timeout=1.0):
        global multiprocess_mip_attack_model, multiprocess_stop
        multiprocess_mip_attack_model = self.mip_model.copy()
        multiprocess_stop = False
        
        for var_name in self.output_names:
            adv = mip_solver_worker(var_name, np.prod(self.input_shape))
            if adv is not None:
                adv = torch.tensor(adv, device=self.device).view(self.input_shape)
                if check_solution(net=self.net, adv=adv, cs=self.objectives.cs, rhs=self.objectives.rhs, data_min=self.objectives.lower_bounds, data_max=self.objectives.upper_bounds):
                    return True, adv
                
        return False, None
    
    
    def attack_domains(self, domain_params):
        # exit()
        # print('mip attack', self.run()[0])
        print('attack_domains', len(domain_params.lower_bounds[0]))
        # print( self.objectives.cs)
        # print( self.objectives.cs.shape)
        if not hasattr(self, 'cex_pool'):
            self.cex_pool = CounterExamplePool(self.abstractor, self.objectives, capacity=200)
        
        select_domain_params = self.select_domains(domain_params, 100)
        print(f'Selected {len(select_domain_params.lower_bounds[0])} domains from {len(domain_params.lower_bounds[0])} domains for MIP hidden attack')
        # for h in domain_params.histories:
        #     print(h)
        # exit()
        
        
    def select_domains(self, domain_params, n_candidates=1):
        # print(len(self.cex_pool))
        # print(common_activation)
        # print([_.shape for _ in common_activation])
        
        # domain_activations = [
        #     ((domain_params.lower_bounds[j] > 0).int() - (domain_params.upper_bounds[j] < 0).int()).flatten(1).cpu()
        #         for j in range(len(domain_params.lower_bounds) - 1)
        # ]
        
        if n_candidates >= len(domain_params.lower_bounds[0]):
            return domain_params

        common_activation = self.cex_pool.get_common_pattern()
        domain_masks = compute_masks(domain_params.lower_bounds, domain_params.upper_bounds, 'cpu')
        domain_activations = [
            (((domain_params.lower_bounds[j] + domain_params.upper_bounds[j]) > 0).int() - ((domain_params.lower_bounds[j] + domain_params.upper_bounds[j]) <= 0).int()).flatten(1).cpu()
                for j in range(len(domain_masks))
        ]
        
        domain_scores = torch.zeros([len(domain_masks[0])])
        for ca, da, dm in zip(common_activation, domain_activations, domain_masks):
            domain_scores += (((ca == da) * dm).sum(dim=1) / dm.sum(dim=1))
        domain_scores = domain_scores / len(domain_masks)
        assert (domain_scores <= 1.0).all()
        
        select_ids = torch.where(domain_scores >= 0.985)[0]
        # print(select_ids)
        if len(select_ids) >= n_candidates:
            select_ids = torch.topk(domain_scores, n_candidates).indices
        else:
            remain_candidates = n_candidates - len(select_ids)
            output_lbs = domain_params.lower_bounds[-1].flatten().cpu()
            normalized_output_lbs = -output_lbs / output_lbs.neg().max()
            # print(normalized_output_lbs, remain_candidates)
            probs = torch.nn.functional.softmax(normalized_output_lbs / 0.1, dim=0)
            extra_ids = probs.multinomial(remain_candidates, replacement=False)
            # print('extra_ids', extra_ids)
            # print('select_ids', select_ids)
            # print('select_ids', )
            select_ids = torch.concat([select_ids, extra_ids]).unique()
            
            
            # assert len(extra_ids) + len(select_ids) == n_candidates
        # print('select_ids', select_ids)
        
        return AbstractResults(**{
            'lower_bounds': [lb[select_ids] for lb in domain_params.lower_bounds], 
            'upper_bounds': [ub[select_ids] for ub in domain_params.upper_bounds], 
        })
        