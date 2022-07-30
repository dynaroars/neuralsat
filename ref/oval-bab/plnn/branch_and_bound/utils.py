import bisect
import torch
import torch.multiprocessing as mp
import copy
import traceback
import os
from math import ceil
from plnn.proxlp_solver.utils import override_numerical_bound_errors


def add_domain(candidate, domains):
    '''
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    '''
    bisect.insort_left(domains, candidate)


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
        if selected_candidate_domain.lower_bound < threshold:
            break

    return selected_candidate_domain


def inverted_pick_out(domains, threshold):
    '''
    Inverted pick out function might be better for SAT problems
    '''
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
        selected_candidate_domain = domains.pop(-1)
        if selected_candidate_domain.lower_bound < threshold:
            break

    return selected_candidate_domain


def n_satisfying_threshold(domains, threshold):
    '''
    Count all the domains in the `domains` sequence
    that have a lower bound lower than `threshold`.

    Returns: int, number of constraints satisfying threshold condition
    '''
    count = 0
    for candidate_domain in domains:
        if candidate_domain.lower_bound < threshold:
            count += 1
    return count


def smart_box_split(ndomain, dualnet, domain_lb, domain_width, useful_cutoff):
    '''
    Use box-constraints to split the input domain.
    Split by dividing the domain into two.
    We decide on which dimension to split by trying all splits with a cheap lower bound.

    `domain`:  A 2d tensor whose rows contain lower and upper limits
               of the corresponding dimension.
    Returns: A list of sub-domains represented as 2d tensors.
    '''
    # We're going to try all possible combinations and get the bounds for each,
    # and pick the one with the largest (lowest lower bound of the two part)
    domain = domain_lb + domain_width * ndomain
    largest_lowest_lb = -float('inf')
    largest_lowest_lb_dim = None
    split_lbs = None
    for dim in range(domain.shape[0]):
        # Split alongst the i-th dimension

        dom1 = domain.clone()
        dom1[dim, 1] = (dom1[dim, 1] + dom1[dim, 0]) / 2
        dom2 = domain.clone()
        dom2[dim, 0] = (dom2[dim, 1] + dom2[dim, 0]) / 2

        both_doms = torch.stack([dom1, dom2], 0)

        # TODO: needs to be adapted to do some sort of FSB as well (too many input dimensions)
        lbs = dualnet.get_lower_bounds(both_doms)

        lowest_lb = lbs.min()
        if lowest_lb > largest_lowest_lb:
            largest_lowest_lb = lowest_lb
            largest_lowest_lb_dim = dim
            split_lbs = lbs

    ndom1 = ndomain.clone()
    ndom1[largest_lowest_lb_dim, 1] = (ndom1[largest_lowest_lb_dim, 1] + ndom1[largest_lowest_lb_dim, 0]) / 2
    ndom2 = ndomain.clone()
    ndom2[largest_lowest_lb_dim, 0] = (ndom2[largest_lowest_lb_dim, 1] + ndom2[largest_lowest_lb_dim, 0]) / 2

    sub_domains = [ndom1, ndom2]

    return sub_domains


def prune_domains(domains, threshold):
    '''
    Remove domain from `domains`
    that have a lower_bound greater than `threshold`
    '''
    # this implementation relies on the domains being sorted according to lower bounds
    for i in range(len(domains)):
        if domains[i].lower_bound >= threshold:
            domains = domains[0:i]
        break
    return domains


def create_subproblem_stacks(repeat_size, dom_stack_entry, lbs_stack_entry, ubs_stack_entry, device):
    # Given lists of tensors for lbs and ubs, and input domain, repeat them repeat_size times onto
    # the chosen device
    splitted_lbs_stacks = [lbs.to(device).repeat(
        ((repeat_size,) + (1,) * (lbs.dim() - 1))) for lbs in lbs_stack_entry]
    splitted_ubs_stacks = [ubs.to(device).repeat(
        ((repeat_size,) + (1,) * (ubs.dim() - 1))) for ubs in ubs_stack_entry]
    splitted_domain = dom_stack_entry.repeat(((repeat_size,) + (1,) * (dom_stack_entry.dim() - 1)))
    return splitted_domain, splitted_lbs_stacks, splitted_ubs_stacks


def create_split_stacks(dom_stack_entry, lbs_stack_entry, ubs_stack_entry):
    # Create stacks for the bounding of the split subproblems (duplicates the subdomains, with the two copies for
    # the i-th subdomain in position 2*i, 2*i + 1.
    unsplit_batch_size = lbs_stack_entry[0].shape[0]
    splitted_lbs_stacks = [lbs.unsqueeze(1).repeat(
        ((1, 2,) + (1,) * (lbs.dim() - 1))).view(2*unsplit_batch_size, *lbs.shape[1:]) for lbs in lbs_stack_entry]
    splitted_ubs_stacks = [ubs.unsqueeze(1).repeat(
        ((1, 2,) + (1,) * (ubs.dim() - 1))).view(2 * unsplit_batch_size, *ubs.shape[1:]) for ubs in ubs_stack_entry]
    splitted_domain = dom_stack_entry.unsqueeze(1).repeat(
        ((1, 2,) + (1,) * (dom_stack_entry.dim() - 1))).view(2 * unsplit_batch_size, *dom_stack_entry.shape[1:])
    return splitted_domain, splitted_lbs_stacks, splitted_ubs_stacks


def set_subproblem_stacks_entry(entry_idx, dom_stack, lbs_stacks, ubs_stacks, dom_stack_entry, lbs_stack_entry,
                                ubs_stack_entry):
    # Set stack entry for lists of stack tensors for lbs and ubs, and a stack tensor for the input domain
    for clbs_stack, cubs_stack, clbs_entry, cubs_entry in zip(lbs_stacks, ubs_stacks, lbs_stack_entry, ubs_stack_entry):
        clbs_stack[entry_idx] = clbs_entry
        cubs_stack[entry_idx] = cubs_entry
    dom_stack[entry_idx] = dom_stack_entry


def get_subproblem_stacks_entry(entry_idx, dom_stack, lbs_stacks, ubs_stacks):
    # Get stack entry for lists of stack tensors for lbs and ubs, and a stack tensor for the input domain
    lbs_entry, ubs_entry = [], []
    for clbs_stack, cubs_stack in zip(lbs_stacks, ubs_stacks):
        lbs_entry.append(clbs_stack[entry_idx].unsqueeze(0))
        ubs_entry.append(cubs_stack[entry_idx].unsqueeze(0))
    return dom_stack[entry_idx].unsqueeze(0), lbs_entry, ubs_entry


def dump_domains_to_file(domains, dumped_domain_filelblist, max_domains_cpu, block_size):
    # when the number of domains overcomes max_domains_cpu, dump a block of size block_size into secondary memory
    # (the last domains in the sorting order are dumped)
    # the secondary memory filelist is used as FIFO
    dumping_folder = "/data0/current_domain_files/"
    file_base = f"block_pid{os.getpid()}_"
    if len(domains) >= max_domains_cpu:
        if not os.path.exists(dumping_folder):
            if not os.access(dumping_folder, os.W_OK):
                # If there is no write access to the dumping_folder, use the project directory instead
                dumping_folder = "./current_domain_files/"
            os.makedirs(dumping_folder)
        c_block = len(dumped_domain_filelblist)
        filename = dumping_folder + file_base + f"{c_block}"
        # Dump a block of domains.
        torch.save(domains[-block_size:], filename)
        dumped_domain_filelblist.append((filename, domains[-block_size].lower_bound))
        # Remove dumped domains from the CPU list.
        return domains[:-block_size]
    else:
        return domains


def get_domains_from_file(domains, dumped_domain_filelblist, block_size):
    # recover domains from secondary memory
    if len(domains) <= block_size and len(dumped_domain_filelblist) != 0:
        filename = dumped_domain_filelblist[-1][0]
        # Load a block of domains.
        loaded_doms = torch.load(filename, map_location='cpu')
        os.remove(filename)
        # Insert them in domain list.
        for cdom in loaded_doms:
            add_domain(cdom, domains)
        # Remove loaded domains from the file list.
        return dumped_domain_filelblist[:-1]
    else:
        return dumped_domain_filelblist


def delete_dumped_domains(dumped_domain_filelblist):
    for cfile, clb in dumped_domain_filelblist:
        os.remove(cfile)


def compute_last_bounds_sequentially(bounds_net, splitted_domain, splitted_lbs, splitted_ubs, batch_indices, share=False):
    # Compute the last layer bounds sequentially over the batch domains (used for Gurobi).

    for batch_idx in batch_indices:

        # Check primal feasibility and don't compute bounds if not satisfied
        clbs = [lbs[batch_idx].unsqueeze(0) for lbs in splitted_lbs]
        cubs = [ubs[batch_idx].unsqueeze(0) for ubs in splitted_ubs]
        primal_feasibility = check_primal_infeasibility(clbs, cubs, clbs[-1], cubs[-1])

        if primal_feasibility.all():
            # Problem seems to be feasible, can compute bounds
            bounds_net.build_model_using_bounds(
                splitted_domain[batch_idx],
                ([lbs[batch_idx] for lbs in splitted_lbs],
                 [ubs[batch_idx] for ubs in splitted_ubs]))
            updated_lbs = bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
            splitted_lbs[-1][batch_idx] = torch.max(updated_lbs, splitted_lbs[-1][batch_idx])
            c_ub_point = bounds_net.get_lower_bound_network_input().clone()
        else:
            # Dummy bounds.
            splitted_lbs[-1][batch_idx] = float('inf') * torch.ones_like(splitted_lbs[-1][batch_idx])
            c_ub_point = splitted_domain.select(-1, 0)[batch_idx].unsqueeze(0)
        # Store the output of the bounding procedure in a format consistent with the batched version.

        dom_ub_point = c_ub_point if batch_idx == batch_indices[0] else torch.cat([dom_ub_point, c_ub_point], 0)
        if share:
            dom_ub_point = share_tensors(dom_ub_point)

    # this is a dummy assigment: no parent initialisation for Gurobi (in case it was needed, we'd need a method to
    # concatenate solutions in ParentInit)
    dual_solutions = bounds_net.children_init

    return splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions


def check_primal_infeasibility(dom_lb_all, dom_ub_all, dom_lb, dom_ub):
    """
    Given intermediate bounds (lists of tensors) and final layer bounds, check whether these constitute an infeasible
    primal problem.
    This is checked via the dual, which is unbounded (lbs are larger than ubs, as we don't go to convergence).
    """
    batch_shape = dom_lb_all[0].shape[:1]
    feasible_output = torch.ones((*batch_shape, 1), device=dom_lb_all[0].device, dtype=torch.bool)
    for c_lbs, c_ubs in zip(dom_lb_all, dom_ub_all):
        feasible_output = feasible_output & (c_ubs - c_lbs >= 0).view((*batch_shape, -1)).all(dim=-1, keepdim=True)
    feasible_output = feasible_output & (dom_ub - dom_lb >= 0).view((*batch_shape, -1)).all(dim=-1, keepdim=True)
    return feasible_output


def get_lb_net_info(out_bounds_dict, current_net, n_nets):
    # Retrieve information concerning the current (and the next, if available) last layer bounding network.
    net_info = out_bounds_dict["nets"][current_net]
    bounds_net = net_info["net"]
    use_auto_iters = net_info["auto_iters"]
    if use_auto_iters:
        bounds_net.default_iters(set_min=True)
    net_info["max_iters_reached"] = False
    batch_size = net_info["batch_size"]
    next_net_info = out_bounds_dict["nets"][current_net + 1] if current_net < n_nets - 1 else None
    return bounds_net, net_info, next_net_info, use_auto_iters, batch_size


def estimate_branches(domain_lb, domain_impr_avg, decision_bound, criteria_dict):
    # Given a domain's info and the decision threshold, infer from statistics how many children this node will generate,
    # assuming a full binary tree underneath the current node.
    easy_away = (decision_bound - domain_lb) / domain_impr_avg if domain_impr_avg != 0 else 0
    easy_away = min(easy_away, 500)  # cap to avoid overflow
    if criteria_dict is not None:
        hard_away = (decision_bound - domain_lb - criteria_dict["lb_impr"]).clamp(0, None) / \
                    domain_impr_avg
        hard_away = min(hard_away.item(), 500)  # cap to avoid overflow (and convert to float to avoid numerical issues)
    else:
        hard_away = 0
    return easy_away, hard_away


def is_difficult_domain(domain, criteria_dict, expansion=2):
    # Given a domain and a dict of criteria, determine whether it is to be considered difficult.
    if criteria_dict is not None:
        #  heuristically determine whether the higher cost per bounding cuts off more subproblems than its overhead
        blowup_condition = (expansion**(domain.easy_away - domain.hard_away)) > criteria_dict["hard_overhead"]
        postpone = "postpone_batches" in criteria_dict and criteria_dict["postpone_batches"] > 0
        return blowup_condition and (not postpone)
    else:
        return False


def do_auto_iters(bounding_net, expected_improvement, criteria_dict, testnode_dict, intermediate_dict):

    # The expected improvement must be less than the tightening of using more iters, considering its overhead
    condition = expected_improvement * criteria_dict['overhead'] < criteria_dict["impr_margin"] \
        if expected_improvement != 0 else False

    # Increase/decrease the number of iterations for the last layer bounding based on expected LB improvement.
    if condition:
        # The number of iterations is increased if branching pays off less than increasing them by a small margin
        # ("step block").
        can_increase = bounding_net.increase_iters()
        print(f"=================================> Increase iters to {bounding_net.steps}")
        if can_increase:
            bounding_net.step_increase *= 2  # diminishing returns so increase the distance between two iter settings
            criteria_dict["+iters_lb"] = assess_impr_margin(
                bounding_net, testnode_dict["domain"], testnode_dict["intermediate_lbs"],
                testnode_dict["intermediate_ubs"], criteria_dict, intermediate_dict, c_lb=criteria_dict["+iters_lb"])
        else:
            criteria_dict["max_iters_reached"] = True
        if criteria_dict["impr_margin"] < 1e-2:
            # If the algorithm has almost converged, we can assume the max. number of iters was reached and start
            # checking for the tighter bounding.
            criteria_dict["max_iters_reached"] = True


def assess_impr_margin(bounding_net, domain, intermediate_lbs, intermediate_ubs, auto_iters_dict, intermediate_dict,
                       c_lb=None):
    # Compute the improvement margin of increasing the iterations by one step block for this problem and
    # bounding algo (increases number of iterations and re-bounds)
    if "pinit" not in auto_iters_dict:
        bounding_net.internal_init()
    else:
        bounding_net.initialize_from(auto_iters_dict["pinit"])
    bounding_net.build_model_using_bounds(domain, (intermediate_lbs, intermediate_ubs))
    if c_lb is None:
        c_lb = bounding_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        if intermediate_dict["joint_ib"]:
            # When jointly optimizing, we only care in the final LB to do autoiters
            c_lb = c_lb[0]
    c_iters = bounding_net.steps
    can_increase = bounding_net.increase_iters()
    new_iters = bounding_net.steps
    tighter_lb = bounding_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
    if intermediate_dict["joint_ib"]:
        # When jointly optimizing, we only care in the final LB to do autoiters
        tighter_lb = tighter_lb[0]
    while (tighter_lb - c_lb).mean() < 1e-4 and can_increase:
        # If the default step_increase is too small, keep on increasing it until the bounds change
        can_increase = bounding_net.increase_iters()
        new_iters = bounding_net.steps
        bounding_net.step_increase = bounding_net.steps - c_iters
        tighter_lb = bounding_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        if intermediate_dict["joint_ib"]:
            # When jointly optimizing, we only care in the final LB to do autoiters
            tighter_lb = tighter_lb[0]
    bounding_net.decrease_iters()
    auto_iters_dict["impr_margin"] = max((tighter_lb - c_lb).mean(), 1e-4 * torch.ones_like(c_lb))
    auto_iters_dict["overhead"] = new_iters / max(1, c_iters)
    print(f"Improvement margin for this problem and bounding algo: {auto_iters_dict['impr_margin']}")
    print(f"Overhead: {auto_iters_dict['overhead']}")
    return tighter_lb


def switch_to_hard(hard_domains, criteria_dict, batch_size):
    # Given the current queue of hard domains and a dict of criteria, determine whether it's really worth switching to
    # more expensive bounding by estimating how many batches would be required with the two bounding schemes.
    easy_children, hard_children = 0, 0
    for cdom in hard_domains:
        easy_children += 2**(cdom.easy_away+1)-1
        hard_children += 2**(cdom.hard_away+1)-1
    # condition = ceil(easy_children / batch_size) > ceil(hard_children / batch_size) * criteria_dict["hard_overhead"]
    condition = easy_children > hard_children * criteria_dict["hard_overhead"]
    if not condition:
        print(f"Postpone possible switch to hard bounding by {ceil(easy_children / batch_size)} batches.")
    return condition, ceil(easy_children / batch_size)


## Functions implementing CPU parallelization for the last layer bound computations.
def last_bounds_cpu_server(pid, bounds_nets, servers_queue, instructions_queue, barrier):
    # Function implementing a CPU process computing last layer bounds (in parallel) until BaB termination is sent.
    try:
        while True:
            # Full synchronizatin after every batch.
            barrier.wait()

            comm = instructions_queue.get(True)  # blocking get on queue
            if len(comm) == 1:
                if comm[0] == "terminate":
                    break
                elif comm[0] == "idle":
                    continue
                elif comm[0] == "switch_bounds_net":
                    # Switch to hard bounding network.
                    bounds_nets = bounds_nets[1]
                    continue
                else:
                    raise ChildProcessError(f"Message type not supported: {comm}")

            splitted_domain, splitted_lbs, splitted_ubs, slice_indices = comm
            c_bounds_net = bounds_nets[0] if type(bounds_nets) is tuple else bounds_nets
            splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = compute_last_bounds_sequentially(
                c_bounds_net, splitted_domain, splitted_lbs, splitted_ubs, slice_indices, share=True)

            # Send results to master
            servers_queue.put((splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions, slice_indices))

    except Exception as e:
        # Exceptions from subprocesses are not caught otherwise.
        print(e)
        print(traceback.format_exc())


def spawn_cpu_servers(p, bounds_net):
    # Create child processes to parallelize the last layer bounds computations over cpu. Uses multiprocessing.
    servers_queue = mp.Queue()
    instruction_queue = mp.Queue()
    barrier = mp.Barrier(p)
    cpu_servers = mp.spawn(last_bounds_cpu_server,
                           args=(copy.deepcopy(bounds_net), servers_queue, instruction_queue, barrier), nprocs=(p-1),
                           join=False)
    return cpu_servers, servers_queue, instruction_queue, barrier


def gurobi_switch_bounding_net(gurobi_dict):
    # Instruct the cpu servers (of gurobi-based last layer bounding) to change bounding net
    barrier = gurobi_dict["barrier"]
    instruction_queue = gurobi_dict["instruction_queue"]
    p = gurobi_dict["p"]
    barrier.wait()
    for _ in range(p-1):
        instruction_queue.put(("switch_bounds_net",))


def join_children(gurobi_dict, timeout):
    cpu_servers = gurobi_dict["cpu_servers"]
    barrier = gurobi_dict["barrier"]
    instruction_queue = gurobi_dict["instruction_queue"]
    gurobi = gurobi_dict["gurobi"]
    p = gurobi_dict["p"]

    if gurobi and p > 1:
        # terminate the cpu servers and wait for it.
        barrier.wait()
        for _ in range(p-1):
            instruction_queue.put(("terminate",))
        cpu_servers.join(timeout=timeout)


def subproblems_to_cpu(splitted_domain, splitted_lbs, splitted_ubs, share=False, squeeze=False):
    # Copy domain and bounds over to the cpu, sharing their memory in order for multiprocessing to access them directly.
    cpu_splitted_domain = splitted_domain.cpu()
    if squeeze:
        cpu_splitted_domain = cpu_splitted_domain.squeeze(0)
    if share:
        cpu_splitted_domain.share_memory_()
    cpu_splitted_lbs = []
    cpu_splitted_ubs = []
    for lbs, ubs in zip(splitted_lbs, splitted_ubs):
        cpu_lbs = lbs.cpu()
        cpu_ubs = ubs.cpu()
        if squeeze:
            cpu_lbs = cpu_lbs.squeeze(0)
            cpu_ubs = cpu_ubs.squeeze(0)
        if share:
            cpu_lbs.share_memory_()
            cpu_ubs.share_memory_()
        cpu_splitted_lbs.append(cpu_lbs)
        cpu_splitted_ubs.append(cpu_ubs)
    return cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs


def share_tensors(tensors):
    # Put a (list of) tensor(s) in shared memory. Copies to CPU in case it wasn't there.
    if isinstance(tensors, list):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cpu().share_memory_()
    else:
        tensors = tensors.cpu().share_memory_()
    return tensors


class ParentInit:
    """
    Abstract class providing all the methods necessary for parent initialisation within the context of Branch and Bound.
    For usage, see plnn/branch_and_bound/relu_branch_and_bound
    """
    def to_cpu(self):
        # Move content to cpu.
        pass

    def to_device(self, device):
        # Move content to device "device"
        pass

    def as_stack(self, stack_size):
        # Repeat the content of this parent init to form a stack of size "stack_size"
        return ParentInit()

    def set_stack_parent_entries(self, parent_solution, batch_idx):
        # Given a solution for the parent problem (at batch_idx), set the corresponding entries of the stack.
        pass

    def get_stack_entry(self, batch_idx):
        # Return the stack entry at batch_idx as a new ParentInit instance.
        return ParentInit()

    def get_lb_init_only(self):
        # Get instance of this class with only entries relative to LBs.
        return ParentInit()

    def expand_layer_batch(self, expand_size):
        # expand this parent-init's layer_batch (assumed to be =1) to "expand_size"
        raise NotImplementedError

    def contract_layer_batch(self):
        # contract this parent-init's layer_batch (assumed to be =!1) to 1 (the last entry is used)
        raise NotImplementedError

    def clone(self):
        # return a copy of this class
        raise NotImplementedError

    def get_bounding_score(self, x_idx):
        # Get a score for which intermediate bound to tighten on layer x_idx (larger is better)
        raise NotImplementedError

    @staticmethod
    def do_stack_list(clist, stack_size):
        # Utility function to be used within as_stack
        return [x[0].unsqueeze(0).repeat(((stack_size,) + (1,) * (x.dim() - 1))) for x in clist]

    @staticmethod
    def set_parent_entries(x, y, batch_idx):
        # Utility function to be used within set_stack_parent_entries
        x[2 * batch_idx] = y.clone()
        x[2 * batch_idx + 1] = y.clone()

    @staticmethod
    def get_entry_list(clist, batch_idx):
        # Utility function to be used within get_stack_entry
        return [csol[batch_idx].unsqueeze(0) for csol in clist]

    @staticmethod
    def lb_only_list(clist):
        # Utility function to be used within get_lb_init_only
        return [c_init[:, -1].unsqueeze(1) for c_init in clist]
