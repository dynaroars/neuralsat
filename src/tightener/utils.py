import logging
import torch
import tqdm
import time
import os

from attacker.pgd_attack.general import attack as pgd_attack
from helper.spec.objective import Objective, DnfObjectives
from helper.misc.adam_clipping import AdamClipping
from helper.misc.check import check_solution
from helper.misc.logger import logger

def generate_simple_specs(dnf_pairs, n_outputs):
    """
    Generate VNNLIB-based specification in negation format
    
    [[(0, 0.5, 'gt')],
     [(1, -0.5, 'gt')],
     [(1, 0.5, 'lt')],
     [(2, 0.1, 'gt')],
     [(2, 1.1, 'lt')],
     [(3, 0.2, 'gt')],
     [(3, 1.2, 'lt')],
     [(4, -0.1, 'gt')],
     [(0, 1.5, 'lt')],
     [(4, 0.31, 'lt')]]
    
    is equivalent to 
    
    ; Output constraints:
    (assert (or
        (and (>= Y_0 0.5))
        (and (>= Y_1 -0.5))
        (and (<= Y_1 0.5))
        (and (>= Y_2 0.1))
        (and (<= Y_2 1.1))
        (and (>= Y_3 0.2))
        (and (<= Y_3 1.2))
        (and (>= Y_4 -0.1))
        (and (<= Y_0 1.5))
        (and (<= Y_4 0.31))
    ))
    """
    
    all_cs = []
    all_rhs = []
    for cnf_pairs in dnf_pairs:
        cs = []
        rhs = []
        for output_i, rhs_i, direction in cnf_pairs:
            assert direction in ['lt', 'gt']
            c = torch.zeros(n_outputs)
            r = torch.tensor(rhs_i)
            c[output_i] = 1. 
            if direction == 'gt':
                c *= -1.
                r *= -1.
            cs.append(c)
            rhs.append(r)
        all_cs.append(torch.stack(cs))
        all_rhs.append(torch.stack(rhs))
        
    lengths = [len(_) for _ in all_cs]
    if len(set(lengths)) == 1:
        return torch.stack(all_cs).cpu(), torch.stack(all_rhs).cpu()
    return all_cs, all_rhs    
    

def falsify_dnf_pairs(model, input_lower, input_upper, n_outputs, candidate_neurons):
    assert len(candidate_neurons) > 0
    batch_size = 10
    device = input_lower.device
    
    # spec
    all_cs, all_rhs = generate_simple_specs(dnf_pairs=candidate_neurons, n_outputs=n_outputs)
    all_cs = all_cs.to(device)
    all_rhs = all_rhs.to(device)
    x_attack = (input_upper - input_lower) * torch.rand(input_lower.shape, device=device) + input_lower
    
    attack_candidates = []
    attack_samples = []  
    for batch_idx in range(0, len(all_cs), batch_size):
        new_cs = all_cs[batch_idx:batch_idx+batch_size]
        new_rhs = all_rhs[batch_idx:batch_idx+batch_size]
        data_min_attack = input_lower.unsqueeze(1).expand(-1, len(new_cs), *input_lower.shape[1:])
        data_max_attack = input_upper.unsqueeze(1).expand(-1, len(new_cs), *input_upper.shape[1:])
        # print(new_cs.shape, data_max_attack.shape, input_upper.shape)
        is_attacked, attack_images = pgd_attack(
            model=model,
            x=x_attack, 
            data_min=data_min_attack,
            data_max=data_max_attack,
            cs=new_cs,
            rhs=new_rhs,
            attack_iters=50, 
            num_restarts=20,
            timeout=5.0,
        )
        if is_attacked:
            with torch.no_grad():
                for restart_idx in range(attack_images.shape[1]): # restarts
                    for prop_idx in range(attack_images.shape[2]): # props
                        attack_candidate = candidate_neurons[batch_idx+prop_idx]
                        if attack_candidate in attack_candidates:
                            # print('skip candidate:', attack_candidate)
                            continue
                        adv = attack_images[:, restart_idx, prop_idx]
                        if check_solution(
                            net=model, 
                            adv=adv, 
                            cs=new_cs[prop_idx], 
                            rhs=new_rhs[prop_idx], 
                            data_min=data_min_attack[:, prop_idx], 
                            data_max=data_max_attack[:, prop_idx]
                        ):
                            attack_candidates.append(attack_candidate)
                            attack_samples.append(adv)
                            if os.environ.get('NEURALSAT_ASSERT'):
                                assert torch.all(adv >= input_lower)
                                assert torch.all(adv <= input_upper)

    return attack_candidates, attack_samples


def filter_dnf_pairs(model, input_lower, input_upper, n_outputs, candidate_neurons, n_iterations=20, patient_limit=2):
    attack_candidates = []  
    attack_samples = []  
    # pbar = tqdm.tqdm(range(n_iterations), desc='Filtering DNF Pairs')   
    patient = patient_limit
    for _ in range(n_iterations):
        new_candidates = [_ for _ in candidate_neurons if _ not in attack_candidates]
        if not len(new_candidates):
            break
        
        new_attack_candidates, new_samples = falsify_dnf_pairs(
            model=model,
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=n_outputs,
            candidate_neurons=new_candidates,
        )
        if not len(new_attack_candidates):
            patient -= 1
            if patient <= 0:
                break
        else:
            # reset patient
            patient = patient_limit
            
        attack_candidates += new_attack_candidates
        attack_samples += new_samples
        # pbar.set_postfix(attacked=len(attack_candidates), patient=patient)
    
    # print(f'{len(attack_candidates)=}, {attack_candidates=}')
    filtered_candidates = [_ for _ in candidate_neurons if _ not in attack_candidates]
    return filtered_candidates, torch.vstack(attack_samples) if len(attack_samples) else []
    
    
def extract_worst_bound(domains, objective_id):
    # print(f'{domains.output_lbs=}')
    # print(f'{domains.objective_ids=}')
    assert len(domains.output_lbs) > 0
    indices = domains.objective_ids == objective_id
    worst_bounds = domains.output_lbs[indices] - domains.rhs[indices]
    assert worst_bounds.numel() > 0, f'{worst_bounds=}'
    return worst_bounds.amin().item()

def extract_solved_objective(verifier, objective):
    verified_ids, falsified_ids, unsolved_ids_w_bounds = [], [], []
    attack_samples = []
    if verifier.adv is not None:
        output = verifier.net(verifier.adv).detach()
        cond = torch.matmul(objective.cs, output.unsqueeze(-1)).squeeze(-1) - objective.rhs
        falsified_ids = torch.where(cond.amax(dim=-1) < 0.0)[0].cpu().detach().numpy().tolist()
        attack_samples.append(verifier.adv)
        # print(f'{falsified_ids=}')
        
    if hasattr(verifier, 'domains_list') and len(verifier.domains_list):
        remaining_domains = verifier.domains_list.pick_out_worst_domains(len(verifier.domains_list), device='cpu')  
        unsolved_objective_ids = remaining_domains.objective_ids.unique().int()
    else:
        unsolved_objective_ids = []
    # print(f'{unsolved_objective_ids=}')
        
    for idx, value in enumerate(objective.ids):
        if idx in falsified_ids:
            continue
        elif value in unsolved_objective_ids:
            # TODO: use worst bound as tightened bound
            worst_bound = extract_worst_bound(remaining_domains, value)
            assert worst_bound <= 1e-6, f'Invalid {worst_bound=}'
            unsolved_ids_w_bounds.append((idx, worst_bound))
            # print(idx, value, worst_bound)
        else:
            # verified
            # worst_bound = extract_worst_bound(remaining_domains, value)
            # print(f'Verified {worst_bound=}')
            # assert worst_bound >= 0.0
            verified_ids.append(idx)
    # print(f'{unsolved_ids_w_bounds=}')
        
    return verified_ids, falsified_ids, unsolved_ids_w_bounds, attack_samples


def verify_dnf_pairs(verifier, input_lower, input_upper, n_outputs, candidate_neurons, batch=10, timeout=10.0, reference_bounds=None):
    # TODO: generalize for other verifier
    print('####### Start running other verifier here #######')
    old_log_level = logger.level
    if not os.environ.get("NEURALSAT_LOG_SUBVERIFIER"):
        logger.setLevel(logging.NOTSET)
        
    # properties
    all_cs, all_rhs = generate_simple_specs(dnf_pairs=candidate_neurons, n_outputs=n_outputs)
    
    # objectives
    objectives = []
    for spec_idx in range(len(all_cs)):
        input_bounds = torch.stack([input_lower.flatten(), input_upper.flatten()], dim=1).detach().cpu()
        objectives.append(
            Objective((
                input_bounds.numpy().tolist(), 
                (all_cs[spec_idx].numpy(), 
                 all_rhs[spec_idx].numpy())
            ))
        )
            
    dnf_objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=verifier.input_shape, 
    )
    # print(f'{dnf_objectives.cs.shape = }, {dnf_objectives.rhs.shape = }')
    
    if os.environ.get('NEURALSAT_ASSERT'):
        assert torch.equal(all_cs, dnf_objectives.cs)
        assert torch.equal(all_rhs, dnf_objectives.rhs)
    
    count = 0
    attack_samples = []  
    verified_candidates, falsified_candidates = [], []
    progress_bar = tqdm.tqdm(total=len(dnf_objectives), desc=f"Verifying intermediate properties")
    while len(dnf_objectives):
        objective = dnf_objectives.pop(batch)
        # print(f'{objective.cs = }, {objective.rhs = }')
        current_candidates = candidate_neurons[count:count+batch]
        assert len(objective.cs) == len(current_candidates), f'{len(objective.cs)=} {len(current_candidates)=}'
        assert torch.equal(objective.cs.nonzero()[..., -1], torch.tensor([p[0][0] for p in current_candidates]))
        
        count += len(current_candidates)
        try:
            verifier.start_time = time.time()
            stat = verifier._verify_one(objective, preconditions={}, reference_bounds=reference_bounds, timeout=timeout)
            # print(f'{stat=}') #, objective.cs.nonzero())
            progress_bar.set_postfix(status=stat, runtime=time.time() - verifier.start_time)
        except:
            print(f'[!] ERROR')
            if os.environ.get('NEURALSAT_DEBUG'):
                print(f'[!] debug')
                raise
            
        # print(f'\t- {current_candidates=}')

        # extract
        verified_ids, falsified_ids, unsolved_ids_w_bounds, adv = extract_solved_objective(verifier=verifier, objective=objective)
        verified_candidates.extend([current_candidates[i][0] for i in verified_ids])
        falsified_candidates.extend([current_candidates[i][0] for i in falsified_ids])
        for (unsolved_id, bound) in unsolved_ids_w_bounds:
            assert bound <= 1e-6, f'{bound=}'
            unsolved_candidate = current_candidates[unsolved_id][0]
            if unsolved_candidate[-1] == 'lt':
                new_candidate = (unsolved_candidate[0], unsolved_candidate[1] + bound, unsolved_candidate[2])
            else:
                new_candidate = (unsolved_candidate[0], unsolved_candidate[1] - bound, unsolved_candidate[2])
            print(f'[DEBUG] unverified {unsolved_candidate} but verified {new_candidate}')
            verified_candidates.append(new_candidate)
        attack_samples += adv
        progress_bar.update(len(current_candidates))
    progress_bar.close()
    
    print(f'Verified : total={len(verified_candidates)} indices={[p[0] for p in verified_candidates]}')
    print(f'Falsified: total={len(falsified_candidates)} indices={[p[0] for p in falsified_candidates]}')
    print('####### End running other verifier here #######')
    logger.setLevel(old_log_level)
    return verified_candidates, torch.vstack(attack_samples) if len(attack_samples) else []


def optimize_dnn(net, lower, upper, n_sample=50, n_iteration=50, is_min=True):
    assert torch.all(lower <= upper)
    lower_expand = lower.expand(n_sample, *[-1] * (lower.ndim - 1))
    upper_expand = upper.expand(n_sample, *[-1] * (upper.ndim - 1))
    X = (torch.empty_like(lower_expand).uniform_() * (upper_expand - lower_expand) + lower_expand).requires_grad_()

    lr = torch.max(upper_expand - lower_expand).item() / 8
    optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    Fs = []
    for _ in tqdm.tqdm(range(n_iteration), desc='Minimizing' if is_min else 'Maximizing'):
        inputs = torch.max(torch.min(X, upper), lower)
        outputs = net(inputs)
        Fs.append(outputs.detach())
        loss = outputs # torch.clamp(outputs, min=-1e-3) if is_min else torch.clamp(outputs, max=1e-3)
        loss = loss.sum() if is_min else -loss.sum()
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    
    Fs = torch.vstack(Fs)
    # print(f'{Fs.shape = }')
    if is_min:
        return Fs.min(0).values
    return Fs.max(0).values      


def optimize_dnn_2(net, lower, upper, n_sample=50, n_iteration=50, is_min=True):
    assert torch.all(lower <= upper)
    lower_expand = lower.expand(n_sample, *[-1] * (lower.ndim - 1))
    upper_expand = upper.expand(n_sample, *[-1] * (upper.ndim - 1))
    X = (torch.empty_like(lower_expand).uniform_() * (upper_expand - lower_expand) + lower_expand)
    
    delta_lower_limit = lower_expand - X
    delta_upper_limit = upper_expand - X
    delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
        
    lr = torch.max(upper_expand - lower_expand).item() / 8
    opt = AdamClipping(params=[delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)
    Fs = []
    for _ in tqdm.tqdm(range(n_iteration), desc='Minimizing' if is_min else 'Maximizing'):
        inputs = torch.max(torch.min((X + delta), upper_expand), lower_expand)
        output = net(inputs)
        Fs.append(output.detach())
        loss = output # torch.clamp(output, min=-1e-5)
        loss = loss.sum() if is_min else -loss.sum()
        loss.backward()
        opt.step(clipping=True, lower_limit=delta_lower_limit, upper_limit=delta_upper_limit, sign=1)
        opt.zero_grad(set_to_none=True)
        scheduler.step()
        
    Fs = torch.vstack(Fs)
    # print(f'{Fs.shape = }')
    if is_min:
        return Fs.min(0).values
    return Fs.max(0).values  


