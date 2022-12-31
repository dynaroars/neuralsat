from core.solver.theory_solver.theory_solver import TheorySolver
from core.solver.sat_solver.sat_solver import SATSolver
from core.solver.sat_solver.solver import Solver
from core.solver.heuristic.decider import Decider

import arguments

class SMTSolver(Solver):

    def __init__(self, net, spec):
        super().__init__()

        self.net = net

        # SAT solver decider
        decider = Decider(net)

        # Theory solver core
        self.theory_solver = TheorySolver(net, spec=spec, decider=decider)

        # SAT solver core
        layers_mapping = net.layers_mapping
        variables = [v for d in layers_mapping.values() for v in d]
        self.sat_solver = SATSolver(variables=variables, layers_mapping=layers_mapping, decider=decider, theory_solver=self)

        # torch.set_num_threads(1)


    def propagate(self):
        # print('SMT propagate')

        conflict_clause = None
        implied_assignments = []

        theory_sat = self.theory_solver.propagate(self.sat_solver.get_assignment())

        if arguments.Config['print_progress']:
            self.theory_solver.print_progress()

        # unreachable case
        if not theory_sat:
            # check if there is no more branch
            if arguments.Config['early_stop']:
                early_stop_status = self.theory_solver.get_early_stop_status()
                if early_stop_status:
                    self.sat_solver.set_early_stop(early_stop_status)
                    return conflict_clause, implied_assignments

            # TODO: Fixme: add multiple conflict clauses 
            self.theory_solver.clear_implications()

            conflict_clause = set()
            for variable, value, is_implied in self.sat_solver.iterable_assignment():
                if not is_implied:
                    conflict_clause.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            return conflict_clause, implied_assignments

        # reachable case 
        for node, value in self.theory_solver.get_implications().items():
            if value['neg']:
                implied_assignments.append(-node)
                continue
            if value['pos']:
                implied_assignments.append(node)
        
        return conflict_clause, implied_assignments


    def get_assignment(self) -> dict:
        return self.theory_solver.get_assignment()


    def solve(self, timeout=None) -> bool:
        return self.sat_solver.solve(timeout)


    def set_early_stop(self, status):
        self.sat_solver.set_early_stop(status)

