from core.abstraction.abstractor import Abstractor
from core.lp_solver.lp_solver import LPSolver
from util.misc.logger import logger
import arguments

import torch.optim as optim
import gurobipy as grb
import torch

class RandomCexGenerator:

    def __init__(self, net, spec, n_sample=10):
        self.net = net
        self.spec = spec
        self.n_sample = n_sample
        
        self.device = net.device
        self.dtype = arguments.Config['dtype']
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=self.dtype, device=self.device).view(net.input_shape)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=self.dtype, device=self.device).view(net.input_shape)
        
        self.mat = []
        for prop_mat, prop_rhs in self.spec.mat:
            prop_mat = torch.from_numpy(prop_mat).float().to(self.device)
            prop_rhs = torch.from_numpy(prop_rhs).float().to(self.device)
            self.mat.append((prop_mat, prop_rhs))
        
    def distance_loss(self, x):
        dl = self.lbs_init - x
        du = x - self.ubs_init
        return dl + du
        
    def check(self, y):
        # print('\t', y[0, 9].item(), y[0, 8].item())
        loss = 0.0
        for prop_mat, prop_rhs in self.mat:
            # print(prop_mat, prop_rhs)
            vec = prop_mat.matmul(y.t())
            sat = torch.all(vec <= prop_rhs.reshape(-1, 1), dim=0)
            if (sat==True).any():
                return True, None
            loss += vec.mean()
        return False, loss
    
    
    def __call__(self, lower=None, upper=None):
        for _ in range(100):
            if (lower is None) or (upper is None):
                cex = (self.ubs_init - self.lbs_init) * torch.rand(self.n_sample, *self.net.input_shape[1:], device=self.device) + self.lbs_init
            else:
                cex = (upper - lower) * torch.rand(self.n_sample, *self.net.input_shape[1:], device=self.device) + lower
                
            assert torch.all(self.lbs_init <= cex) and torch.all(cex <= self.ubs_init)
            
            orig_cex = cex.clone()
            
            cex.requires_grad = True
            
            optimizer = optim.Adam([cex], lr=5e-3, betas=(0.6, 0.98))
            old_loss = 1e5
            
            for i in range(200):
                optimizer.zero_grad()
                stat, loss = self.check(self.net(cex))
                if stat:
                    print(_, i, stat)
                    exit()
                
                # loss += self.distance_loss(cex).mean()
                # self.net.zero_grad()
                loss.backward()
                optimizer.step()
                # print('grad:', cex.grad.sign().shape)
                
                # cex.data = cex.data + 2/255*cex.grad.sign()
                # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
                # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

                for ii in range(cex.shape[0]):
                    cex[ii].data[cex[ii] < self.lbs_init[0]] = orig_cex[ii][cex[ii] < self.lbs_init[0]]
                    cex[ii].data[cex[ii] > self.ubs_init[0]] = orig_cex[ii][cex[ii] > self.ubs_init[0]]
                    # cex[ii].data[cex[ii] < self.lbs_init[0]] = self.lbs_init[0, cex[ii] < self.lbs_init[0]]
                    # cex[ii].data[cex[ii] > self.ubs_init[0]] = self.ubs_init[0, cex[ii] > self.ubs_init[0]]
                assert torch.all(self.lbs_init <= cex) and torch.all(cex <= self.ubs_init)
                
                # cex[cex < ]
                
                if abs(old_loss - loss.item()) < 1e-5:
                    break
                
                old_loss = loss.item()
            print(f'[{_}, {i}] loss', loss.item())
            
            print(_, stat)
        exit()
        return stat
            
        
        
class ReLUTheory:

    "Theory for ReLU"

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.spec = spec

        self.layers_mapping = net.layers_mapping
        self.device = net.device

        # data type when create new tensor
        self.dtype = arguments.Config['dtype']

        # Abstraction (over-approximation)
        self.abstractor = Abstractor(net, spec)

        # implied nodes
        self.implications = {}

        # all domains
        self.all_domains = {}

        # extra conflict clauses
        self.extra_conflict_clauses = []

        # SATSolver decider
        assert decider is not None
        self.sat_decider = decider
        self.sat_decider.all_domains = self.all_domains
        self.sat_decider.abstractor = self.abstractor

        # number of calls to theory
        self.n_call = 0

        # solution if found
        self.assignment = None

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=self.dtype, device=self.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=self.dtype, device=self.device)

        assert len(self.ubs_init.flatten()) == len(self.lbs_init.flatten()) == self.net.n_input
        
        self.lp_model = None # for solving full assignment
        
        # self.cex_generator = RandomCexGenerator(net, spec)
        # self.cex_generator()
        

    def assignment_to_conflict_clause(self, assignment):
        conflict_clause = set()
        for variable, value in assignment.items():
            conflict_clause.add(-variable if value else variable)
        return conflict_clause


    def add_domain(self, domains):
        # FIXME: check if key existed
        for d in domains:
            if d.unsat:
                self.extra_conflict_clauses.append(self.assignment_to_conflict_clause(d.get_assignment()))
            self.all_domains[self.get_hash_assignment(d.get_assignment())] = d


    def get_hash_assignment(self, assignment):
        non_implied_assignment = self.get_non_implied_assignment(assignment)
        return hash(frozenset(non_implied_assignment.items()))


    def get_domain(self, assignment):
        return self.all_domains.get(self.get_hash_assignment(assignment), None)


    def get_non_implied_assignment(self, assignment):
        if len(assignment) == 0:
            return {}
        for key, value in assignment.items():
            if isinstance(value, dict):
                return {k: v['value'] for k, v in assignment.items() if v['description'] != 'tcp'}
            return assignment


    def get_last_assignment(self, assignment):
        non_implied_assignment_full_info = {k: v for k, v in assignment.items() if v['description'] != 'tcp'}
        if len(non_implied_assignment_full_info) == 1:
            return {}

        sorted_assignment_by_level = sorted(non_implied_assignment_full_info.items(), key=lambda item: item[1]['level'], reverse=True)
        last_decision, last_decision_dict = sorted_assignment_by_level[0]
        
        # debug
        last_level = last_decision_dict['level']
        assert len([k for k, v in non_implied_assignment_full_info.items() if v['level'] == last_level])
        
        del non_implied_assignment_full_info[last_decision]
        return non_implied_assignment_full_info


    def get_unassigned_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes


    def print_progress(self):
        self.n_call += 1
        n_full_domains = len(self.get_full_domains())
        if n_full_domains:
            msg = f'Iteration={self.n_call:<5} (#remain={len(self.get_valid_domains()):<5} #full={n_full_domains:<5} #total={len(self.all_domains)})'
        else:
            msg = f'Iteration={self.n_call:<5} (#remain={len(self.get_valid_domains()):<5} #total={len(self.all_domains)})'
        logger.info(msg)


    def get_valid_domains(self):
        return [d for _, d in self.all_domains.items() if (d.valid and not d.unsat)]


    def get_full_domains(self):
        return [d for _, d in self.all_domains.items() if (d.full and not d.unsat)]


    def process_implied_assignment(self, assignment):
        logger.debug('\tprocess_implied_assignment')
        # clear implications for next iteration
        self.implications = {}
        return True


    def process_init_assignment(self):
        logger.debug('\tprocess_init_assignment')
        init_domain = self.abstractor.forward(input_lower=self.lbs_init, input_upper=self.ubs_init, extra_params=None)
        if init_domain.unsat: 
            return False

        # get implications
        count = 1
        for lbs, ubs in zip(init_domain.lower_all[:-1], init_domain.upper_all[:-1]):
            if (lbs - ubs).max() > 1e-6:
                return False

            for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):
                if u <= 0:
                    self.implications[count + jj] = {'pos': False, 'neg': True}
                elif l > 0:
                    self.implications[count + jj] = {'pos': True, 'neg': False}
            count += lbs.numel()

        self.sat_decider.current_domain = init_domain
        self.all_domains[self.get_hash_assignment({})] = init_domain
        return True
        

    def process_extra_assignment(self, assignment):
        logger.debug('\tprocess_extra_assignment')
        raise NotImplementedError


    def process_new_assignment(self, assignment):
        logger.debug('\tprocess_new_assignment')
        last_assignment = self.get_last_assignment(assignment)
        # print('last assignment:', last_assignment)
        last_domain = self.get_domain(last_assignment)
        if last_domain is None:
            # there exists 'bcp' (extra) assignments
            self.process_extra_assignment(assignment)
        
        # get decisions for a batch
        decisions, selected_domains, extra_params = self.sat_decider.get_batch_decisions(last_domain)
        assert len(decisions) == len(selected_domains), print(f'#decisions={len(decisions)}, #domains={len(selected_domains)}')
        for var, domain in zip(decisions, selected_domains):
            domain.next_decision = var
        
        # compute abstraction
        domains = self.abstractor.forward(input_lower=self.lbs_init, input_upper=self.ubs_init, extra_params=extra_params)
        self.add_domain(domains)
        
        current_domain = self.get_domain(self.get_non_implied_assignment(assignment))
        assert current_domain is not None
        return current_domain


    def process_cached_assignment(self, assignment, current_domain):
        logger.debug('\tprocess_cached_assignment')
        if current_domain.unsat:
            return False

        # get implications
        count = 1
        for lbs, ubs in zip(current_domain.lower_all[:-1], current_domain.upper_all[:-1]):
            if (lbs - ubs).max() > 1e-6:
                return False

            for jj, (l, u) in enumerate(zip(lbs.flatten(), ubs.flatten())):
                if (count + jj) in assignment:
                    continue
                if u <= 0:
                    self.implications[count + jj] = {'pos': False, 'neg': True}
                elif l > 0:
                    self.implications[count + jj] = {'pos': True, 'neg': False}
            count += lbs.numel()

        self.sat_decider.current_domain = current_domain
        return True


    def process_extra_domains(self):
        logger.debug('\tprocess_extra_domains')
        decisions, selected_domains, extra_params = self.sat_decider.get_batch_decisions(None)
        if selected_domains:
            assert len(decisions) == len(selected_domains), print(f'#decisions={len(decisions)}, #domains={len(selected_domains)}')
            for var, domain in zip(decisions, selected_domains):
                domain.next_decision = var
            domains = self.abstractor.forward(input_lower=self.lbs_init, input_upper=self.ubs_init, extra_params=extra_params)
            self.add_domain(domains)

        if len(self.get_full_domains()):
            self.process_extra_full_domains()


    def process_extra_full_domains(self):
        logger.debug('\tprocess_extra_full_domains')
        
        # create LP solver instance
        if self.lp_model is None:
            self.lp_model = LPSolver(self.net, self.spec)

        # for idx, d in enumerate(self.all_domains.values()): # debug only
        for idx, d in enumerate(self.get_full_domains()):
            d.to_cpu()
            feasible, adv = self.lp_model.solve(lower_bounds=d.lower_all[:-1], upper_bounds=d.upper_all[:-1])
            # print(feasible)
            if not feasible:
                d.unsat = True
                d.valid = False
            else:
                if self.spec.check_solution(self.net(adv)):
                    self.assignment = adv
                else:
                    # invalid adv, but still feasible, set assignment for dummy array
                    self.assignment = torch.tensor([1, 2, 3], dtype=self.dtype, device=self.device)
                break

    def process_full_assignment(self, assignment):
        logger.debug('\tprocess_full_assignment')
        
        # create LP solver instance
        if self.lp_model is None:
            self.lp_model = LPSolver(self.net, self.spec)
            
        feasible, adv = self.lp_model.solve(assignment=assignment)
        
        if feasible:
            if self.spec.check_solution(self.net(adv)):
                self.assignment = adv
            else:
                # invalid adv, but still feasible, set assignment for dummy array
                self.assignment = torch.tensor([1, 2, 3], dtype=self.dtype, device=self.device)
            return True
        
        return False


    def propagate(self, assignment):
        logger.debug('ReLUTheory propagate')

        if len(assignment) == 0:
            return self.process_init_assignment()

        unassigned_nodes = self.get_unassigned_nodes(assignment)

        # full assignment
        if unassigned_nodes is None:
            return self.process_full_assignment(assignment)

        if len(self.implications) > 0:
            return self.process_implied_assignment(assignment)

        current_domain = self.get_domain(assignment)
        if current_domain is None:
            current_domain = self.process_new_assignment(assignment)

        self.process_extra_domains()

        return self.process_cached_assignment(assignment, current_domain)



