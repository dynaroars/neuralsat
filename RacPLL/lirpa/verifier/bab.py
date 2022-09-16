from collections import defaultdict, Counter
from sortedcontainers import SortedList
import numpy as np
import torch
import time

from .branching_domain import ReLUDomain, pick_out_batch, add_domain_parallel
from .branching_heuristic import choose_node_parallel_crown
from auto_lirpa.utils import stop_criterion_sum
import config

Visited, Flag_first_split = 0, True
Use_optimized_split = False
all_node_split = False
DFS_enabled = False


def batch_verification(domains, net, batch, pre_relu_indices, growth_rate, layer_set_bound=True, single_node_split=True, adv_pool=None):
    global Visited, Flag_first_split, Use_optimized_split, DFS_enabled

    decision_thresh = config.Config["bab"]["decision_thresh"]
    branching_method = config.Config['bab']['branching']['method']
    branching_reduceop = config.Config['bab']['branching']['reduceop']
    get_upper_bound = config.Config["bab"]["get_upper_bound"]

    mask, lAs, orig_lbs, orig_ubs, slopes, betas, intermediate_betas, selected_domains = pick_out_batch(domains, decision_thresh, batch=batch, device=net.x.device)

    if mask is not None:
        history = [sd.history for sd in selected_domains]
        split_history = [sd.split_history for sd in selected_domains]

        # print(history, ' <=========== history')
        # print(split_history, ' <=========== split_history')

        if branching_method == 'babsr':
            branching_decision = choose_node_parallel_crown(orig_lbs, orig_ubs, mask, net, pre_relu_indices, lAs, batch=batch, branching_reduceop=branching_reduceop)
        else:
            raise NotImplementedError

        # print(branching_decision, ' <=========== branching_decision')
        # print(len(mask[0]), ' <=========== len(mask[0])')

        if len(branching_decision) < len(mask[0]):
            print('all nodes are split!!')
            global all_node_split
            all_node_split = True
            return selected_domains[0].lower_bound, np.inf

        print('splitting decisions: {}'.format(branching_decision[:10]))

        # if not Use_optimized_split:
        split = {}
        split["decision"] = [[bd] for bd in branching_decision]
        split["coeffs"] = [[1.] for i in range(len(branching_decision))]
        split["diving"] = 0

        # else:
        #     split = {}
        #     num_nodes = 3
        #     split["decision"] = [[[2, i] for i in range(num_nodes)] for bd in branching_decision]
        #     split["coeffs"] = [[random.random() * 0.001 - 0.0005 for j in range(num_nodes)] for i in
        #                        range(len(branching_decision))]

        
        dom_ub, dom_lb, dom_ub_point, lAs, dom_lb_all, dom_ub_all, slopes, split_history, betas, intermediate_betas, primals = net.get_lower_bound(orig_lbs, orig_ubs, split, slopes=slopes, history=history, split_history=split_history, layer_set_bound=layer_set_bound, betas=betas, single_node_split=single_node_split, intermediate_betas=intermediate_betas)

        if adv_pool is not None:
            raise

        batch, diving_batch = len(branching_decision), split["diving"]
        check_infeasibility = not (single_node_split and layer_set_bound)
        unsat_list = add_domain_parallel(lA=lAs[:2*batch], lb=dom_lb[:2*batch], ub=dom_ub[:2*batch], lb_all=dom_lb_all[:2*batch], up_all=dom_ub_all[:2*batch],
                                         domains=domains, selected_domains=selected_domains[:batch], slope=slopes[:2*batch], beta=betas[:2*batch],
                                         growth_rate=growth_rate, branching_decision=branching_decision, decision_thresh=decision_thresh,
                                         split_history=split_history[:2*batch], intermediate_betas=intermediate_betas[:2*batch],
                                         check_infeasibility=check_infeasibility, primals=primals[:2*batch] if primals is not None else None)

        Visited += (len(selected_domains) - diving_batch - len(unsat_list)) * 2  # one unstable neuron split to two nodes

    print('length of domains:', len(domains))


    if len(domains) > 0:
        global_lb = domains[0].lower_bound
    else:
        print("No domains left, verification finished!")
        return torch.tensor(config.Config["bab"]["decision_thresh"] + 1e-7), np.inf

    batch_ub = np.inf
    if get_upper_bound:
        batch_ub = min(dom_ub)

    print('{} neurons visited'.format(Visited))

    return global_lb, batch_ub





def relu_bab_parallel(net, domain, x, use_neuron_set_strategy=False, refined_lower_bounds=None, refined_upper_bounds=None, reference_slopes=None, attack_images=None):
    global Visited, Flag_first_split, all_node_split, DFS_enabled
    start = time.time()

    decision_thresh = config.Config["bab"]["decision_thresh"]
    max_domains = config.Config["bab"]["max_domains"]
    batch = config.Config["bab"]["batch_size"]
    get_upper_bound = config.Config["bab"]["get_upper_bound"]
    timeout = config.Config["bab"]["timeout"]


    global_ub, global_lb, _, _, primals, updated_mask, lA, lower_bounds, upper_bounds, pre_relu_indices, slope, history = net.build_the_model(
        domain, x, stop_criterion_func=stop_criterion_sum(decision_thresh))

    if isinstance(global_lb, torch.Tensor):
        global_lb = global_lb.item()

    print(global_lb)

    if global_lb > decision_thresh:
        return global_lb, global_ub, [[time.time()-start, global_lb]], 0


    if True:
        # If we are not optimizing intermediate layer bounds, we do not need to save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        new_slope = defaultdict(dict)
        output_layer_name = net.net.final_name
        for relu_layer, alphas in slope.items():
            new_slope[relu_layer][output_layer_name] = alphas[output_layer_name]
        slope = new_slope

    # This is the first (initial) domain.
    candidate_domain = ReLUDomain(lA, global_lb, global_ub, lower_bounds, upper_bounds, slope, history=history, depth=0, primals=primals).to_cpu()
    domains = SortedList()
    domains.add(candidate_domain)


    tot_ambi_nodes = 0
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable
    print(f'-----------------\n# of unstable neurons: {tot_ambi_nodes}\n-----------------\n')


    glb_record = [[time.time()-start, global_lb]]
    stop_condition = len(domains) > 0

    while stop_condition:

        global_lb, batch_ub = batch_verification(domains, net, batch, pre_relu_indices, 0)
        print(f"Global ub: {global_ub}, batch ub: {batch_ub}")
        global_ub = min(global_ub, batch_ub)

        stop_condition = len(domains) > 0

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.item()
        if isinstance(global_ub, torch.Tensor):
            global_ub = global_ub.item()


        if all_node_split:
            del domains
            all_node_split = False
            return global_lb, global_ub, glb_record, Visited

        if len(domains) > max_domains:
            print("No enough memory for the domain list!!!!!!!!")
            del domains
            return global_lb, global_ub, glb_record, Visited


        if get_upper_bound:
            if global_ub < decision_thresh:
                print("Attack success during bab!!!!!!!!")
                # Terminate MIP if it has been started.
                del domains
                return global_lb, global_ub, glb_record, Visited

        if time.time() - start > timeout:
            print('Time out!!!!!!!!')
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, global_ub, glb_record, Visited

        print(f'Cumulative time: {time.time() - start}\n')

    del domains
    return global_lb, global_ub, glb_record, Visited

