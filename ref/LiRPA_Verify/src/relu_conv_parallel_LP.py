import time
from copy import deepcopy
import numpy as np
import torch

from branch_and_bound import pick_out, pick_out_batch, add_domain, add_domain_parallel, ReLUDomain
from babsr_score_conv import choose_node_conv, choose_node_parallel

Visited = 0


def batch_verification(d, net, batch, pre_relu_indices, no_LP, growth_rate, decision_thresh=0, layer_set_bound=True, lr_alpha=0.1):
    global Visited

    def find_parent(cd):
        while True:
            if cd.parent is None or cd.parent.valid is False:  # or candidate_domain.parent not in d, already popped
                break
            cd = cd.parent
            mask = cd.mask
            orig_lbs = cd.lower_all
            orig_ubs = cd.upper_all
            mask_temp = [i.clone() for i in mask]
            d_ub, d_lb, d_ub_point, u_mask, d_lb_all, d_ub_all, s = net.get_lower_bound(mask_temp, orig_lbs, orig_ubs,
                                                                                        None, None, parallel=False,
                                                                                        no_LP=False, warm=True,
                                                                                        slopes=cd.slope, lr_alpha=lr_alpha
                                                                                        )
            print('dom_lb LP (find parent): ', d_lb, 'depth:', cd.depth)
            if d_lb < 0:
                break
            else:
                cd.del_node()

    unsat_list = []
    mask, orig_lbs, orig_ubs, slopes, selected_domains = pick_out_batch(d, decision_thresh, batch, device=net.x.device)
    relu_start = time.time()
    if mask is not None:
        branching_decision = choose_node_parallel(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices,
                                                  None, 0, batch=batch)
        print('splitting decisions: {}'.format(branching_decision))
        if not branching_decision or len(selected_domains)!=len(branching_decision):
            print("no unstable relus in some branches, call LP!")
            d[:] = selected_domains+d
            return None
        # print('history', selected_domains[0].history)
        history = [sd.history for sd in selected_domains]
        # print(len(history), history, len(branching_decision))

        mask_temp = [i.clone() for i in mask]
        ret = net.get_lower_bound(mask_temp, orig_lbs, orig_ubs, branching_decision, choice=None, parallel=True,
                                  no_LP=no_LP, slopes=slopes, history=history, decision_thresh=decision_thresh,
                                  layer_set_bound=layer_set_bound, lr_alpha=lr_alpha)
        dom_ub, dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all, slopes = ret
        print('dom_lb parallel: ', dom_lb[:10])
        bd_0, bd_1 = [], []
        for bd in branching_decision:
            bd_0.append([(bd, 1)])
            bd_1.append([(bd, 0)])

        unsat_list = add_domain_parallel(updated_mask, lb=dom_lb, ub=dom_ub, lb_all=dom_lb_all, up_all=dom_ub_all,
                                         domains=d, selected_domains=selected_domains, slope=slopes,
                                         growth_rate=growth_rate, branching_decision=bd_0+bd_1, decision_thresh=decision_thresh)
        Visited += (len(selected_domains) - len(unsat_list)) * 2  # split to two nodes
    else:
        print('All candidate in domain were split by CROWN')
        candidate_domain = pick_out(d, decision_thresh)
        # Generate new, smaller domains by splitting over a ReLU
        mask = candidate_domain.mask

        orig_lbs = candidate_domain.lower_all
        orig_ubs = candidate_domain.upper_all
        branching_decision, _ = choose_node_conv(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices,
                                                         0, [0], 0)
        Visited += 2
        for choice in [0, 1]:
            mask_temp = [i.clone() for i in mask]
            ret = net.get_lower_bound(mask_temp, orig_lbs, orig_ubs, branching_decision, choice, parallel=False,
                                      no_LP=False, warm=True, slopes=candidate_domain.slope, lr_alpha=lr_alpha)
            d_ub, d_lb, d_ub_point, u_mask, d_lb_all, d_ub_all, s = ret
            print('dom_lb again: ', d_lb)
            # print('dom_ub: ', d_ub)

            if d_lb < 0:  # or global_ub?
                dom_to_add = ReLUDomain(u_mask, lb=d_lb, ub=d_ub, lb_all=d_lb_all, up_all=d_ub_all, slope=s, depth=candidate_domain.depth+1)
                add_domain(dom_to_add, d)
            elif len(d) > 0:
                find_parent(candidate_domain)
                # continue

    if not no_LP and len(unsat_list) > 0:
        print('solving LP...', unsat_list[0], 'split: ', len(selected_domains), 'reject: ', len(unsat_list))
        candidate_domain = selected_domains[unsat_list[0]]
        d.remove(candidate_domain)
        Visited += 2
        for c in [0, 1]:
            b = c*len(selected_domains) + unsat_list[0]
            if dom_lb[b] > 0: continue

            mask = candidate_domain.mask
            orig_lbs = candidate_domain.lower_all
            orig_ubs = candidate_domain.upper_all

            pre_split = (dom_lb_all[b], dom_ub_all[b], updated_mask[b])
            mask_temp = [i.clone() for i in mask]
            # solving by LP
            d_ub, d_lb, d_ub_point, u_mask, d_lb_all, d_ub_all, s = net.get_lower_bound(mask_temp,
                                                                orig_lbs, orig_ubs, None, None, parallel=False,
                                                                no_LP=False, warm=True, slopes=slopes[b],
                                                                pre_split=pre_split, lr_alpha=lr_alpha)

            print('dom_lb LP: ', d_lb, 'depth:', candidate_domain.depth)

            if d_lb < 0:
                dom_to_add = ReLUDomain(u_mask, lb=d_lb, ub=d_ub, lb_all=d_lb_all, up_all=d_ub_all, slope=s, depth=candidate_domain.depth+1)
                add_domain(dom_to_add, d)

    relu_end = time.time()
    print('relu split requires: ', relu_end - relu_start)
    print('length of domains:', len(d))

    if len(d) > 0:
        global_lb = d[0].lower_bound
    else:
        return 999

    print(f"Current lb:{global_lb.item()}")

    print('{} neurons visited'.format(Visited))

    return global_lb


def relu_bab_parallel(net, domain, x, batch=64, no_LP=False, growth_rate=0, decision_thresh=0, lr_alpha=0.1,
                      use_neuron_set_strategy=False, max_subproblems_list=100000, timeout=3600):
    start = time.time()
    max_length = 0
    global Visited
    Visited = 0
    global_ub, global_lb, global_ub_point, duals, primals, updated_mask, lower_bounds, upper_bounds, pre_relu_indices, slope = net.build_the_model(
        domain, x, no_lp=no_LP, decision_thresh=decision_thresh)

    print(global_lb)
    if global_lb > decision_thresh:
        return global_lb, global_ub, global_ub_point, 0, max_length

    lower_boundsc, upper_boundsc, slopec = deepcopy(lower_bounds), deepcopy(upper_bounds), deepcopy(slope)
    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds, slope, depth=0).to_cpu()
    domains = [candidate_domain]
    tot_ambi_nodes = 0
    for layer_mask in updated_mask:
        tot_ambi_nodes += torch.sum(layer_mask == -1).item()

    print('# of unstable neurons:', tot_ambi_nodes)
    random_order = np.arange(tot_ambi_nodes)
    np.random.shuffle(random_order)

    batch_count, current_lb, batch_early_stop = 0, 99, False
    while len(domains) > 0:
        if len(domains) > max_length: max_length = len(domains)

        if no_LP:
            global_lb = batch_verification(domains, net, batch, pre_relu_indices, no_LP, 0,
                                           decision_thresh=decision_thresh, lr_alpha=lr_alpha)

            if len(domains) > max_subproblems_list:
                print("reach the maximum number for the domain list")
                del domains
                return global_lb, np.inf, None, Visited, max_length

            if time.time() - start > timeout:
                print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                del domains
                return global_lb, np.inf, None, Visited, max_length

        else:
            if len(domains) > max_subproblems_list or batch_early_stop:
                print(time.time() - start, "time for parallel")
                global_ub, global_lb, global_ub_point, duals, primals, updated_mask, lower_bounds, upper_bounds, pre_relu_indices, slope = net.build_the_model(
                    domain, x, no_lp=no_LP, decision_thresh=decision_thresh, continue_LP=[upper_boundsc, lower_boundsc, slopec])

                print(global_lb)
                if global_ub < decision_thresh:
                    print("unsafe checked by LP")
                    return global_ub, global_ub, global_ub_point, 0, max_length
                if global_lb > decision_thresh:
                    return global_lb, global_ub, global_ub_point, 0, max_length

                candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds, slope,
                                              depth=0).to_cpu()

                domains = [candidate_domain]
                # Visited = 0
                torch.cuda.empty_cache()
                while len(domains) > 0:
                    global_lb = batch_verification(domains, net, 1, pre_relu_indices, no_LP=False, growth_rate=0.2, lr_alpha=lr_alpha)
                    if global_lb is None:
                        del domain
                        print("unsafe checked by LP")
                        return current_lb, global_ub, None, Visited, max_length
                    if global_lb >= decision_thresh:
                        del domains
                        return global_lb, np.inf, None, Visited, max_length
                    if time.time() - start > timeout:
                        print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        del domains
                        return global_lb, np.inf, None, Visited, max_length
            else:
                batch_count += 1
                global_lb = batch_verification(domains, net, batch, pre_relu_indices, no_LP=True, growth_rate=0, lr_alpha=lr_alpha)

                if global_lb is None:
                    batch_early_stop = True
                    continue

                # track whether no improvement
                if batch_count % 20 == 0:
                    if current_lb == global_lb:
                        batch_early_stop = True
                    current_lb = global_lb

            if global_lb >= decision_thresh:
                break

    del domains
    return global_lb, np.inf, None, Visited, max_length
