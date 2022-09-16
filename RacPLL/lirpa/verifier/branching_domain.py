import torch
import copy


class ReLUDomain:

    def __init__(self, lA=None, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, slope=None, beta=None, depth=None, split_history=None, history=None, gnn_decision=None, intermediate_betas=None, primals=None, priority=0):
        if history is None:
            history = []
        if split_history is None:
            self.split_history = []

        self.lA = lA
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.intermediate_betas = intermediate_betas
        self.beta = beta
        self.slope = slope
        self.split = False
        self.valid = True

        self.split_history = split_history
        self.history = history
        self.depth = depth
        self.priority = priority  # Higher priority will be more likely to be selected.

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.lower_bound < other.lower_bound
        else:
            # higher priority should be in the front of the queue.
            return self.priority >= other.priority

    def __le__(self, other):
        if self.priority == other.priority:
            return self.lower_bound <= other.lower_bound
        else:
            return self.priority > other.priority

    def __eq__(self, other):
        if self.priority == other.priority:
            return self.lower_bound == other.lower_bound
        else:
            return self.priority == other.priority



    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.lA = [lA.to(device='cpu', non_blocking=True) for lA in self.lA]
        self.lower_all = [lbs.to(device='cpu', non_blocking=True) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device='cpu', non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].half().to(device='cpu', non_blocking=True)

        if self.split_history:
            if "beta" in self.split_history:
                for lidx in range(len(self.split_history["beta"])):
                    if self.split_history["single_beta"][lidx] is not None:
                        self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                        self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device='cpu', non_blocking=True)
                        self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device='cpu', non_blocking=True)
                    if self.split_history["beta"][lidx] is not None:
                        self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device='cpu', non_blocking=True)
                        self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device='cpu', non_blocking=True)
                        self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device='cpu', non_blocking=True)
                        self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device='cpu', non_blocking=True)
                    if self.split_history["bias"][lidx] is not None:
                        self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device='cpu', non_blocking=True)
            if "general_beta" in self.split_history:
                self.split_history["general_beta"] = self.split_history["general_beta"].to(device="cpu", non_blocking=True)

        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device='cpu', non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device='cpu', non_blocking=True)

        if self.beta is not None:
            self.beta = [b.to(device='cpu', non_blocking=True) for b in self.beta]
        
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device='cpu', non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device='cpu', non_blocking=True)
        return self

    def to_device(self, device, partial=False):
        if not partial:
            self.lA = [lA.to(device, non_blocking=True) for lA in self.lA]
            self.lower_all = [lbs.to(device, non_blocking=True) for lbs in self.lower_all]
            self.upper_all = [ubs.to(device, non_blocking=True) for ubs in self.upper_all]
        for layer in self.slope:
            for intermediate_layer in self.slope[layer]:
                self.slope[layer][intermediate_layer] = self.slope[layer][intermediate_layer].to(device, non_blocking=True, dtype=torch.get_default_dtype())
        if self.split_history:
            if "beta" in self.split_history:
                for lidx in range(len(self.split_history["beta"])):
                    if self.split_history["single_beta"][lidx] is not None:
                        self.split_history["single_beta"][lidx]["nonzero"] = self.split_history["single_beta"][lidx]["nonzero"].to(device=device, non_blocking=True)
                        self.split_history["single_beta"][lidx]["value"] = self.split_history["single_beta"][lidx]["value"].to(device=device, non_blocking=True)
                        self.split_history["single_beta"][lidx]["c"] = self.split_history["single_beta"][lidx]["c"].to(device=device, non_blocking=True)
                    if self.split_history["beta"][lidx] is not None:
                        self.split_history["beta"][lidx] = self.split_history["beta"][lidx].to(device=device, non_blocking=True)
                        self.split_history["c"][lidx] = self.split_history["c"][lidx].to(device=device, non_blocking=True)
                        self.split_history["coeffs"][lidx]["nonzero"] = self.split_history["coeffs"][lidx]["nonzero"].to(device=device, non_blocking=True)
                        self.split_history["coeffs"][lidx]["coeffs"] = self.split_history["coeffs"][lidx]["coeffs"].to(device=device, non_blocking=True)
                    if self.split_history["bias"][lidx] is not None:
                        self.split_history["bias"][lidx] = self.split_history["bias"][lidx].to(device=device, non_blocking=True)
            if "general_beta" in self.split_history:
                self.split_history["general_beta"] = self.split_history["general_beta"].to(device=device, non_blocking=True)
        if self.intermediate_betas is not None:
            for split_layer in self.intermediate_betas:
                for intermediate_layer in self.intermediate_betas[split_layer]:
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["lb"].to(device, non_blocking=True)
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"] = \
                    self.intermediate_betas[split_layer][intermediate_layer]["ub"].to(device, non_blocking=True)
        if self.beta is not None:
            self.beta = [b.to(device, non_blocking=True) for b in self.beta]
        # if self.primals is not None:
        #     for layer_idx, _ in enumerate(self.primals['p']):
        #         self.primals['p'][layer_idx] = self.primals['p'][layer_idx].to(device, non_blocking=True)
        #     for layer_idx, _ in enumerate(self.primals['z']):
        #         self.primals['z'][layer_idx] = self.primals['z'][layer_idx].to(device, non_blocking=True)
        return self





def pick_out_batch(domains, threshold, batch, device='cuda', DFS_percent=0, diving=False):
    if torch.cuda.is_available(): 
        torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

    batch = min(len(domains), batch)
    lAs, lower_all, upper_all, slopes_all, betas_all, intermediate_betas_all, selected_candidate_domains = [], [], [], [], [], [], []

    idx, idx2 = 0, 0
    while True:
        if len(domains) == 0:
            print(f"No domain left to pick from. Batch limit {batch} current batch: {idx}")
            break

        if idx2 == len(domains): 
            break  # or len(domains)-1?

        if domains[idx2].split is True:
            idx2 += 1
            continue

        selected_candidate_domain = domains.pop(idx2)

        if selected_candidate_domain.lower_bound < threshold and selected_candidate_domain.valid is True:
            selected_candidate_domain.to_device(device, partial=True)

            lAs.append(selected_candidate_domain.lA)
            lower_all.append(selected_candidate_domain.lower_all)
            upper_all.append(selected_candidate_domain.upper_all)
            slopes_all.append(selected_candidate_domain.slope)
            betas_all.append(selected_candidate_domain.beta)
            intermediate_betas_all.append(selected_candidate_domain.intermediate_betas)
            selected_candidate_domains.append(selected_candidate_domain)
            selected_candidate_domain.valid = False  # set False to avoid another pop

            idx += 1
            if idx == batch: 
                break

        selected_candidate_domain.valid = False   # set False to avoid another pop

    batch = idx

    if batch == 0:
        return None, None, None, None, None

    # Reshape to batch first in each list + Transfer to GPU.
    lower_bounds = []
    for j in range(len(lower_all[0])):
        lower_bounds.append(torch.cat([lower_all[i][j] for i in range(batch)]))
    lower_bounds = [t.to(device=device, non_blocking=True) for t in lower_bounds]

    upper_bounds = []
    for j in range(len(upper_all[0])):
        upper_bounds.append(torch.cat([upper_all[i][j] for i in range(batch)]))
    upper_bounds = [t.to(device=device, non_blocking=True) for t in upper_bounds]

    new_lAs = []
    for j in range(len(lAs[0])):
        new_lAs.append(torch.cat([lAs[i][j] for i in range(batch)]))
    new_lAs = [t.to(device=device, non_blocking=True) for t in new_lAs]

    slopes = []
    if slopes_all[0] is not None:
        if isinstance(slopes_all[0], dict):
            # Per-neuron slope, each slope is a dictionary.
            slopes = slopes_all
        else:
            for j in range(len(slopes_all[0])):
                slopes.append(torch.cat([slopes_all[i][j] for i in range(batch)]))

    # Non-contiguous bounds will cause issues, so we make sure they are contiguous here.
    lower_bounds = [t if t.is_contiguous() else t.contiguous() for t in lower_bounds]
    upper_bounds = [t if t.is_contiguous() else t.contiguous() for t in upper_bounds]

    new_masks = []
    for j in range(len(lower_bounds) - 1):  # Exclude the final output layer.
        new_masks.append(torch.logical_and(lower_bounds[j] < 0, upper_bounds[j] > 0).view(lower_bounds[0].size(0), -1).float())
        
    return new_masks, new_lAs, lower_bounds, upper_bounds, slopes, betas_all, intermediate_betas_all, selected_candidate_domains



def add_domain_parallel(lA, lb, ub, lb_all, up_all, domains, selected_domains, slope, beta, growth_rate=0, split_history=None, branching_decision=None, save_tree=False, decision_thresh=0, intermediate_betas=None, check_infeasibility=True, primals=None, priorities=None):
    """
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    add domains in two ways:
    1. add to a sorted list
    2. add to a binary tree
    # diving: we are adding diving domains if True
    """

    unsat_list = []
    batch = len(selected_domains)
    for i in range(batch):
        infeasible = False
        if lb[i] < decision_thresh:
            if check_infeasibility:
                for ii, (l, u) in enumerate(zip(lb_all[i][1:-1], up_all[i][1:-1])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

            if not infeasible:
                priority=0 if priorities is None else priorities[i].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                if branching_decision is not None:
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # first half batch: active neurons
                    new_history[branching_decision[i][0]][1].append(+1.0)  # first half batch: active neurons

                    # sanity check repeated split
                    if branching_decision[i][1] in selected_domains[i].history[branching_decision[i][0]][0]:
                        print('BUG!!! repeated split!')
                        print(selected_domains[i].history)
                        print(branching_decision[i])
                        raise RuntimeError

                left_primals = primals[i] if primals is not None else None
                left = ReLUDomain(lA[i], lb[i], ub[i], lb_all[i], up_all[i], slope[i], beta[i],
                                  selected_domains[i].depth+1, split_history=split_history[i],
                                  history=new_history,
                                  intermediate_betas=intermediate_betas[i],
                                  primals=left_primals, priority=priority)

                if save_tree:
                    selected_domains[i].left = left
                    left.parent = selected_domains[i]

                domains.add(left)

        infeasible = False
        if lb[i+batch] < decision_thresh:
            if check_infeasibility:
                for ii, (l, u) in enumerate(zip(lb_all[i+batch][1:-1], up_all[i+batch][1:-1])):
                    if (l-u).max() > 1e-6:
                        infeasible = True
                        print('infeasible detected when adding to domain!!!!!!!!!!!!!!')
                        break

            if not infeasible:
                priority=0 if priorities is None else priorities[i+batch].item()
                new_history = copy.deepcopy(selected_domains[i].history)
                if branching_decision is not None:
                    new_history[branching_decision[i][0]][0].append(branching_decision[i][1])  # second half batch: inactive neurons
                    new_history[branching_decision[i][0]][1].append(-1.0)  # second half batch: inactive neurons

                right_primals = primals[i + batch] if primals is not None else None
                right = ReLUDomain(lA[i+batch], lb[i+batch], ub[i+batch], lb_all[i+batch], up_all[i+batch],
                                   slope[i+batch],  beta[i+batch], selected_domains[i].depth+1, split_history=split_history[i+batch],
                                   history=new_history,
                                   intermediate_betas=intermediate_betas[i + batch],
                                   primals=right_primals, priority=priority)

                if save_tree:
                    selected_domains[i].right = right
                    right.parent = selected_domains[i]

                domains.add(right)
    return unsat_list
