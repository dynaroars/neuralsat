from .relu_theory import ReLUTheory
import arguments


class TheorySolver:

    "Interface class for theory solver"

    def __init__(self, net, spec, decider=None):
        self.theory = ReLUTheory(net, spec, decider)


    def get_assignment(self):
        return self.theory.assignment


    def propagate(self, assignment):
        if self.get_assignment() is not None:
            return True
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
        if len(valid_domains) > arguments.Config['max_hidden_branch']:
            return arguments.ReturnStatus.UNKNOWN
        return None


    def get_extra_conflict_clause(self):
        return self.theory.extra_conflict_clauses


    def clear_extra_conflict_clause(self):
        self.theory.extra_conflict_clauses = []

    
    def print_progress(self):
        self.theory.print_progress()
        

