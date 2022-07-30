import bisect

import torch


class ReLUDomain:
    """
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    """

    def __init__(self, mask, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, slope=None, depth=None,
                 history=None, gnn_decision=None):
        if history is None:
            history = []
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.history = history
        self.slope = slope
        self.left = None
        self.right = None
        self.parent = None
        self.valid = True
        self.split = False
        self.depth = depth
        self.gnn_decision = gnn_decision

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def del_node(self):
        if self.left is not None:
            self.left.del_node()
        if self.right is not None:
            self.right.del_node()

        self.valid = False

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.mask = [msk.cpu() for msk in self.mask]
        self.lower_bound = self.lower_bound.cpu()
        self.upper_bound = self.upper_bound.cpu()
        self.lower_all = [lbs.cpu() for lbs in self.lower_all]
        self.upper_all = [ubs.cpu() for ubs in self.upper_all]
        self.slope = [s.cpu() for s in self.slope]
        return self

    def to_device(self, device):
        self.mask = [msk.to(device) for msk in self.mask]
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        self.lower_all = [lbs.to(device) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device) for ubs in self.upper_all]
        self.slope = [s.to(device) for s in self.slope]
        return self



def add_domain(candidate, domains):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    """
    bisect.insort_left(domains, candidate.to_cpu())


def add_domain_parallel(updated_mask, lb, ub, lb_all, up_all, domains, selected_domains, slope, growth_rate=0,
                        branching_decision=None, save_tree=False, decision_thresh=0):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    add domains in two ways:
    1. add to a sorted list
    2. add to a binary tree
    """
    unsat_list = []
    batch = len(selected_domains)
    for i in range(batch):
        if selected_domains[i].valid is True:
            infeasible = False
            if lb[i] < decision_thresh:
                if growth_rate and (
                        selected_domains[i].lower_bound - lb[i] > selected_domains[i].lower_bound * growth_rate or
                        selected_domains[i].lower_bound - lb[i + batch] > selected_domains[
                            i].lower_bound * growth_rate):
                    selected_domains[i].split = True
                    unsat_list.append(i)  # not satisfy the growth_rate
                    # if len(unsat_list) == 1: # skip the first unsat node, cause we will solve it by LP
                    #     choice = unsat_list[0]
                    bisect.insort_left(domains, selected_domains[i].to_cpu())
                    continue
                else:
                    for ii, (l, u) in enumerate(zip(lb_all[i][3:-2], up_all[i][3:-2])):
                        if (l-u).max() > 1e-6:
                            infeasible = True
                            print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                            break

                    if not infeasible:
                        # only when two splits improved, we insert them to domains
                        left = ReLUDomain(updated_mask[i], lb[i], ub[i], lb_all[i], up_all[i], slope[i],
                                          selected_domains[i].depth+1, history=selected_domains[i].history+branching_decision[i])
                        if save_tree:
                            selected_domains[i].left = left
                            left.parent = selected_domains[i]

                            # assert (m[mp == 0] == 0).all(), m[mp == 0].abs().sum()
                            # assert (m[mp == 1] == 1).all(), m[mp == 1].abs().sum()
                        bisect.insort_left(domains, left.to_cpu())

            infeasible = False
            if lb[i+batch] < decision_thresh:
                # if growth_rate and (selected_domains[i].lower_bound - lb[i+batch]) > selected_domains[i].lower_bound * growth_rate and flag:
                #     selected_domains[i].split = True
                #     bisect.insort_left(domains, selected_domains[i])
                #     unsat_list.append(i+batch)
                #     # if len(unsat_list) == 1: choice = unsat_list[0]
                # else:

                for ii, (l, u) in enumerate(zip(lb_all[i+batch][3:-2], up_all[i+batch][3:-2])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

                if not infeasible:
                    right = ReLUDomain(updated_mask[i+batch], lb[i+batch], ub[i+batch], lb_all[i+batch], up_all[i+batch],
                                       slope[i+batch], selected_domains[i].depth+1, history=selected_domains[i].history+branching_decision[i+batch])
                    if save_tree:
                        selected_domains[i].right = right
                        right.parent = selected_domains[i]

                    # for ii, (m, mp) in enumerate(zip(updated_mask[i+batch], selected_domains[i].mask)):
                    #     if not ((m[mp == 0] == 0).all() and (m[mp == 1] == 1).all()):
                    #         infeasible = True
                    #         print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                    #         break

                        # assert (m[mp == 0] == 0).all(), m[mp == 0].abs().sum()
                        # assert (m[mp == 1] == 1).all(), m[mp == 1].abs().sum()

                    bisect.insort_left(domains, right.to_cpu())

    return unsat_list


def pick_out(domains, threshold):
    '''
    Pick the first domain in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    '''
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        assert len(domains) > 0, "No domain left to pick from."
        selected_candidate_domain = domains.pop(0)
        if selected_candidate_domain.lower_bound < threshold and selected_candidate_domain.valid is True:
            break
        else:
            print('select domain again', selected_candidate_domain.lower_bound, threshold)

    return selected_candidate_domain


def pick_out_batch(domains, threshold, batch, device='cuda'):
    """
    Pick the first batch of domains in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    """
    idx, idx2 = 0, 0
    batch = min(len(domains), batch)
    masks, lower_all, upper_all, slopes_all, selected_candidate_domains = [], [], [], [], []
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        if len(domains) == 0:
            print("No domain left to pick from. current batch: {}".format(idx))
            break
        # try:
        if idx2 == len(domains): break  # or len(domains)-1?
        if domains[idx2].split is True:
            idx2 += 1
            # print(idx2, len(domains))
            continue
        # except:
        #     import pdb; pdb.set_trace()
        selected_candidate_domain = domains.pop(idx2)
        # idx2 -= 1
        if selected_candidate_domain.lower_bound < threshold and selected_candidate_domain.valid is True:
            # unique = [x for i, x in enumerate(selected_candidate_domain.history) if i == selected_candidate_domain.history.index(x)]
            # assert len(unique) == len(selected_candidate_domain.history)
            idx += 1
            selected_candidate_domain.to_device(device)
            masks.append(selected_candidate_domain.mask)
            lower_all.append(selected_candidate_domain.lower_all)
            upper_all.append(selected_candidate_domain.upper_all)
            slopes_all.append(selected_candidate_domain.slope)
            selected_candidate_domains.append(selected_candidate_domain)
            if idx == batch: break
        # else:
        #     print('select domain again: {:.4f}, split: {}'.format(selected_candidate_domain.lower_bound, selected_candidate_domain.split))
    batch = idx

    if batch == 0:
        return None, None, None, None, None

    # reshape to batch first in each list
    new_mask = []
    for j in range(len(masks[0])):
        new_mask.append(torch.cat([masks[i][j].unsqueeze(0) for i in range(batch)]))

    lower_bounds = []
    for j in range(len(lower_all[0])):
        lower_bounds.append(torch.cat([lower_all[i][j].unsqueeze(0) for i in range(batch)]))

    upper_bounds = []
    for j in range(len(upper_all[0])):
        upper_bounds.append(torch.cat([upper_all[i][j].unsqueeze(0) for i in range(batch)]))

    slopes = []
    if slopes_all[0] is not None:
        for j in range(len(slopes_all[0])):
            slopes.append(torch.cat([slopes_all[i][j].unsqueeze(0) for i in range(batch)]))

    return new_mask, lower_bounds, upper_bounds, slopes, selected_candidate_domains


def prune_domains(domains, threshold):
    '''
    Remove domain from `domains`
    that have a lower_bound greater than `threshold`
    '''
    # TODO: Could do this with binary search rather than iterating.
    # TODO: If this is not sorted according to lower bounds, this
    # implementation is incorrect because we can not reason about the lower
    # bounds of the domain that come after
    for i in range(len(domains)):
        if domains[i].lower_bound >= threshold:
            domains = domains[0:i]
            break
    return domains


def print_remaining_domain(domains):
    '''
    Iterate over all the domains, measuring the part of the whole input space
    that they contain and print the total share it represents.
    '''
    remaining_area = 0
    for dom in domains:
        remaining_area += dom.area()
    print(f'Remaining portion of the input space: {remaining_area*100:.8f}%')
