import gurobipy as grb
import torch

from core.helper.symbolic_network import SymbolicNetwork
from core.abstraction.abstractor import Abstractor
from util.misc.logger import logger
import arguments

class TheorySolver:

    "Interface class for theory solver"

    def __init__(self, net, spec, decider=None):
        self.theory = ReLUTheory(net, spec, decider)


    def get_assignment(self):
        return self.theory.assignment


    def propagate(self, assignment):
        return self.theory.propagate(assignment)


    def get_implications(self):
        return self.theory.implications


    def clear_implications(self):
        self.theory.implications = {}


    def get_early_stop_status(self):
        if self.get_assignment() is not None:
            return arguments.ReturnStatus.SAT
        valid_domains = self.theory.get_valid_domains()
        if len(valid_domains) == 0:
            return arguments.ReturnStatus.UNSAT
        if len(valid_domains) > arguments.Config['max_branch']:
            return arguments.ReturnStatus.UNKNOWN
        return None

    def print_progress(self):
        self.theory.print_progress()
        

class ReLUTheory:

    "Theory for ReLU"

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.spec = spec

        self.layers_mapping = net.layers_mapping
        self.device = net.device

        # data type when create new tensor
        self.dtype = arguments.Config['dtype']

        # Symbolic transformation to get output equations over inputs
        self.symbolic_transformer = SymbolicNetwork(net)

        # Abstraction (over-approximation)
        self.abstractor = Abstractor(net, spec)

        # LP solver
        # self.init_lp_solver()

        # implied nodes
        self.implications = {}

        # all domains
        self.all_domains = {}


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


    def init_lp_solver(self):
        self.lp_solver = grb.Model()
        self.lp_solver.setParam('Threads', 1)
        self.lp_solver.setParam('OutputFlag', False)
        self.lp_solver.setParam('FeasibilityTol', 1e-8)

        # variables in normal form
        self.lp_vars = [self.lp_solver.addVar(name=f'i{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) for i in range(self.net.n_input)]
        # variables in matrix form
        self.lp_mvars = grb.MVar(self.lp_vars)
        self.lp_solver.update()


    def add_domain(self, domains):
        for d in domains:
            # FIXME: check if key existed
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
        msg = f'Iteration={self.n_call:<5} (remain={len(self.get_valid_domains())}/{len(self.all_domains)})'
        logger.info(msg)


    def get_valid_domains(self):
        return [d for _, d in self.all_domains.items() if (d.valid and not d.unsat)]


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
        raise


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


    def process_full_assignment(self, assignment):
        logger.debug('\tprocess_full_assignment')
        raise NotImplementedError



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
        else:
            self.process_extra_domains()

        return self.process_cached_assignment(assignment, current_domain)



