import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import random

from utils.timer import Timers
from abstract.crown import arguments, choose_node_parallel_kFSB
import settings

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound.clamp(max=0)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio
    return slope_ratio, intercept


class Decider:

    def __init__(self, net, dataset):

        self.net = net
        self.layers_mapping = net.layers_mapping
        self.reversed_layers_mapping = {n: k for k, v in self.layers_mapping.items() for n in v}
        self.bounds_mapping = {}
        self.target_direction_list = None
        self.device = net.device
        self.dataset = dataset
        self.next_var = None
        if settings.SEED is not None:
            random.seed(settings.SEED)

    def update(self, output_bounds=None, hidden_bounds=None, layer_bounds=None, crown_params=None, bounds_mapping=None):
        if hidden_bounds is not None:
            for idx, (lb, ub) in enumerate(hidden_bounds):
                b = [(l, u) for l, u in zip(lb.flatten(), ub.flatten())]
                assert len(b) == len(self.layers_mapping[idx])
                assert (lb <= ub).all()
                self.bounds_mapping.update(dict(zip(self.layers_mapping[idx], b)))

        if output_bounds is not None:
            self.output_lower, self.output_upper = output_bounds

        if layer_bounds is not None:
            for node, bound in layer_bounds.items():
                self.bounds_mapping[node] = torch.tensor(bound['lb']), torch.tensor(bound['ub'])

        if crown_params is not None:
            self.crown_params = crown_params
            if arguments.Config['general']['search_cex']:
                orig_lbs, orig_ubs = crown_params[0], crown_params[1]
                temp_bounds_mapping = {}
                count = 1
                for lbs, ubs in zip(orig_lbs[:-1], orig_ubs[:-1]):
                    for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):
                        temp_bounds_mapping[count + jj] = (l, u)
                    count += lbs.numel()

                self.bounds_mapping.update(temp_bounds_mapping)


        if bounds_mapping is not None:
            self.bounds_mapping.update(bounds_mapping)

        

    def get_score(self, node):
        l, u = self.bounds_mapping[node]
        # score = (u - l)
        score = min(u, -l)
        # score = (u + l) / (u - l)
        # print(node, score, u, l)
        # exit()
        return score#.abs()

    def get_impact(self, node):
        l, u = self.bounds_mapping[node]
        impact = - u * l / (u - l)
        return impact * u.abs()

    def estimate_grads_from_layer(self, lower, upper, layer_id, steps=3):
        inputs = [(((steps - i) * lower + i * upper) / steps) for i in range(steps + 1)]
        diffs = torch.zeros(len(lower), dtype=settings.DTYPE, device=self.device)

        for sample in range(steps + 1):
            pred = self.net.forward_from_layer(inputs[sample], layer_id)
            for index in range(len(lower)):
                if sample < steps:
                    l_input = [m if i != index else u for i, m, u in zip(range(len(lower)), inputs[sample], inputs[sample+1])]
                    l_input = torch.tensor(l_input, dtype=settings.DTYPE, device=self.device)
                    l_i_pred = self.net.forward_from_layer(l_input, layer_id)
                else:
                    l_i_pred = pred
                if sample > 0:
                    u_input = [m if i != index else l for i, m, l in zip(range(len(lower)), inputs[sample], inputs[sample-1])]
                    u_input = torch.tensor(u_input, dtype=settings.DTYPE, device=self.device)
                    u_i_pred = self.net.forward_from_layer(u_input, layer_id)
                else:
                    u_i_pred = pred
                diff = sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
                diffs[index] += diff
        return diffs / steps



    def get(self, unassigned_nodes):
        # if settings.DECISION == 'MAX_BOUND':
        #     scores = [(n, self.get_score(n)) for n in unassigned_nodes]
        #     scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
        #     node = scores[0][0]
        #     l, u = self.bounds_mapping[node]
        #     return node, u.abs() >= l.abs()
        if (self.next_var is not None) and (self.next_var in unassigned_nodes):
            node = self.next_var
            self.next_var = None
            return node, True

        if self.dataset in ['acasxu', 'test']:
        # if settings.DECISION == 'MIN_BOUND':
            # print('unassigned_nodes:', unassigned_nodes)
            try:
                scores = [(n, self.get_score(n)) for n in unassigned_nodes]
                reverse = True
                scores = sorted(scores, key=lambda tup: tup[1], reverse=reverse)
                node = scores[0][0]
                # l, u = self.bounds_mapping[node]
                # print('\t- Decide:', (node-1) % 50)
                return node, True 
            except KeyError:
                node = random.choice(unassigned_nodes)
                return node, random.choice([True, False])

        # if settings.DECISION == 'RANDOM':
        #     node = random.choice(unassigned_nodes)
        #     return node, random.choice([True, False])
        
        if arguments.Config['general']['search_cex']:
            assert self.dataset in ['mnist', 'cifar'], print(self.dataset)
            # print('get decision from here')
            scores = [(n, self.get_score(n)) for n in unassigned_nodes]
            reverse = False
            scores = sorted(scores, key=lambda tup: tup[1], reverse=reverse)
            node = scores[0][0]
            l, u = self.bounds_mapping[node]
            return node, u.abs() >= l.abs()


        # if settings.DECISION == 'BABSR':
        else:
            branching_reduceop = arguments.Config['bab']['branching']['reduceop']
            orig_lbs, orig_ubs, mask, lirpa_model, pre_relu_indices, lAs, slopes, betas, history = self.crown_params

            # branching_decision = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, lirpa_model, pre_relu_indices, lAs, batch=1, branching_reduceop=branching_reduceop)

            branching_decision = choose_node_parallel_kFSB(orig_lbs, orig_ubs, mask, lirpa_model,pre_relu_indices, lAs, branching_reduceop=branching_reduceop, slopes=slopes, betas=betas, history=history)

            decision_layer, decision_index = branching_decision[0]
            # print(mask)
            for m in range(decision_layer):
                decision_index += mask[m].numel()

            # print(branching_decision, history)

            # exit()
            node = decision_index + 1
            # if node not in unassigned_nodes:
            #     return random.choice(unassigned_nodes), random.choice([True, False])
                # print(branching_decision)
                # print(orig_lbs[0].flatten()[node - 1])
                # print(orig_ubs[0].flatten()[node - 1])
                # print('dit', node)
                # raise
            # if node in unassigned_nodes:
            return node, random.choice([True, False])







        




