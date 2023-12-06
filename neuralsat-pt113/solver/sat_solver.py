from collections import deque, Counter
import random
import torch
import copy
import time

from util.misc.logger import logger

class SATSolver:
    
    # __slots__ = '_literal_to_clause', '_variable_to_watched_clause', 'clauses', '_last_assigned_literals', 'assignment'

    def __init__(self, clauses):

        self._last_assigned_literals = False
        self.assignment = {}
        
        # add clauses
        if len(clauses):
            max_len = max([len(_) for _ in clauses])
            self.clauses = torch.zeros([len(clauses), max_len], dtype=torch.int)
            for idx, clause in enumerate(clauses):
                self.clauses[idx][:len(clause)] = torch.tensor(list(clause))
        else:
            self.clauses = torch.zeros(0)
            
            
    def assign(self, literal):
        if len(self.clauses) == 0:
            return True
        
        # save assignment
        if isinstance(literal, torch.Tensor):
            variable = literal.abs().item()
            value = literal.item() > 0
        elif isinstance(literal, int):
            variable = abs(literal)
            value = literal > 0
        else:
            raise NotImplementedError()
        
        if variable in self.assignment: 
            if self.assignment[variable] != (literal > 0):
                return False
            # logger.debug(f'Already assigned: {variable}')
            return True
            # raise ValueError('Already assigned')
            
        self.assignment[variable] = value
        self._last_assigned_literals = True
        
        # remove satisfied clauses
        remove_idx = torch.where(self.clauses == literal)[0]
        remain_mask = torch.ones(len(self.clauses), dtype=torch.bool)
        remain_mask[remove_idx] = False
        self.clauses = self.clauses[remain_mask]
        
        # remove unsatisfied literals
        self.clauses = torch.where(self.clauses == -literal, 0, self.clauses)
        
        # print('[+] Assigned:', literal, self.assignment)
        # print(self.clauses)
        return True
        
    def _bcp_step(self):
        if len(self.clauses) == 0:
            return True, []
            
        inferred_variables = []
 
        # clauses with single literal -> infer
        literals = self.clauses[torch.where(self.clauses.count_nonzero(dim=1) == 1)[0]].sum(dim=1)
        # print('inferred:', literals)
        for lit in literals:
            inferred_variables.append(lit.item())
            if not self.assign(lit): # conflict
                return False, []
            # print('BCP assign', lit)
            
        # clauses with 0 literal -> conflict
        if len(torch.where(self.clauses.count_nonzero(dim=1) == 0)[0]):
            return False, []
        
        return True, inferred_variables
        
    def bcp(self):
        if len(self.clauses) == 0:
            return True, []

        inferred_variables = []

        stat, inferred = self._bcp_step()
        if not stat:
            return False, []
            
        inferred_variables += inferred
        
        # print(self.clauses)
        while self._last_assigned_literals:
            # print('[+] Last assigned:', self._last_assigned_literals)
            self._last_assigned_literals = False
            stat, inferred = self._bcp_step()
            if not stat:
                return False, []
            inferred_variables += inferred

        # print('[+] BCP Infer:', inferred_variables)
        return True, inferred_variables  # no conflict


    def print_stats(self):
        print('[+] Current assignment:', self.assignment)
        
        print('[+] Remain clauses:')
        print(self.clauses)
        print()
        
        
    def __deepcopy__(self, memo):
        new_solver = SATSolver([])
        new_solver.clauses = self.clauses.clone()
        new_solver.assignment = copy.deepcopy(self.assignment)
        new_solver._last_assigned_literals = False
        return new_solver


    def multiple_assign(self, literals):
        if len(self.clauses) == 0:
            return True
        
        try:
            return self.multiple_assign_cpp(literals)
        except:
            return self.multiple_assign_python(literals)
    
    
    def multiple_assign_python(self, literals):
        logger.debug('[!] Parallel assign using Python')
        
        remain_mask = torch.ones(self.clauses.size(0), dtype=torch.bool, device=self.clauses.device)
        for lit in literals:
            remain_mask &= self.clauses.ne(lit).all(1)
        self.clauses = self.clauses[remain_mask]
        
        zero_mask = torch.zeros(self.clauses.size(), dtype=torch.bool, device=self.clauses.device)
        for lit in literals:
            zero_mask |= self.clauses.eq(-lit)

        # self.clauses = torch.where(zero_mask, 0, self.clauses)
        self.clauses[zero_mask] = 0
        return True
    
    
    def multiple_assign_cpp(self, literals):
        import haioc
        logger.debug('[!] Parallel assign using C++')
        
        xs = torch.tensor(literals).int()#.cuda()
        # self.clauses = self.clauses.cuda()
        self.clauses = self.clauses[haioc.any_eq_any(self.clauses, xs).logical_not_()]
        self.clauses = haioc.fill_if_eq_any(self.clauses, -xs, 0, inplace=True)
        return True
    

if __name__ == "__main__":
    # clauses = [
    #     [1, -2, -3 , -4],
    #     [3],
    #     [5, -2, 6],
    #     [1, -3],
    #     [-1, 2, -3],
    #     [-2, -5],
    #     [-1, 6, 5, 7, -2],
    #     # [-1, -3]
    # ]
    
    # s = SATSolver(clauses)
    
    # print(s.clauses)
    # logger.setLevel(1)
    
    # stat, inferred = s.bcp()
    # if not stat:
    #     print('Conflict')
    # else:
    #     print('Infer:', inferred)
        
    #     s.print_stats()
    # exit()
    seed = random.randint(0, 1000)
    # seed = 549
    print('seed:', seed)
    random.seed(seed)
    lens = list(range(1, 1000+1))
    n_clauses = 20000
    n_vars = 5000
    # clauses = []
    clauses = [random.sample(range(-n_vars, n_vars), random.choice(lens)) for _ in range(n_clauses)]
    clauses = [c for c in clauses if 0 not in c]
    # print(clauses)
    # clauses += [[i, -i] for i in range(1, n_vars+1)]
    lits = [i for i in random.sample(range(-n_vars,n_vars), int(0.5 * n_vars)) if i != 0]
    lits = [l for l in lits if -l not in lits]
    
    print('Assign literals:', lits[:10])

    ####################
    s = SATSolver(clauses)
    tic = time.time()
    for lit in lits:
        s.assign(lit)
    print('[old] assign:', time.time() - tic)    
    s.bcp()
    c1 = s.clauses.clone()
    
    ####################
    s = SATSolver(clauses)
    tic = time.time()
    s.multiple_assign_python(lits)
    print('[python] assign:', time.time() - tic)   
    s.bcp()
    c2 = s.clauses.clone()
    
    ####################
    
    s = SATSolver(clauses)
    tic = time.time()
    s.multiple_assign(lits)
    print('[cpp] assign:', time.time() - tic)   
    s.bcp()
    c3 = s.clauses.clone()
    
    
    ####################
    assert torch.equal(c1, c2)
    assert torch.equal(c2, c3)
    
    print(c1.shape, 'TRUE')
    
    
    
    # print((s.clauses[..., None] == torch.tensor(lits)[..., None]).nonzero())