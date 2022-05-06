from pprint import pprint
import numpy as np
import copy

from dnn_solver.dnn_theorem_prover import DNNTheoremProver
from sat_solver.custom_sat_solver import CustomSATSolver
from sat_solver.sat_solver import Solver
from dnn_solver.utils import InputParser
import settings

class TheorySolver(Solver):

    def __init__(self, formula, vars_mapping, layers_mapping, first_var=None, 
        max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        self._solver = CustomSATSolver(formula,
                                       vars_mapping,
                                       layers_mapping,
                                       max_new_clauses=max_new_clauses,
                                       halving_period=halving_period,
                                       theory_solver=self)

    def get_assignment(self) -> dict:
        pass

    def solve(self) -> bool:
        return self._solver.solve()




class DNNSolver(TheorySolver):

    def __init__(self, dnn, spec):

        self.dnn = dnn
        vars_mapping, layers_mapping = InputParser.parse(self.dnn)

        super().__init__(formula=None, vars_mapping=vars_mapping, layers_mapping=layers_mapping)
        self.dnn_theorem_prover = DNNTheoremProver(self.dnn, copy.deepcopy(layers_mapping), spec=spec)

    def propagate(self):
        if settings.DEBUG:
            print('- Theory propagate\n')

        conflict_clause = None
        new_assignments = []

        assignment = {k: v['value'] for k, v in self._solver._assignment.items()}

        if settings.DEBUG:
            print('- Assignment:', assignment)

        # theory checking
        theory_sat, implications, is_full_assignment = self.dnn_theorem_prover(assignment)

        if not theory_sat:
            conflict_clause = set()
            if settings.DEBUG:
                print('    - Check T-SAT: `UNSAT`')
            for variable, value in self._solver.iterable_assignment():
                conflict_clause.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            if settings.DEBUG:
                print(f'    - Conflict clause: `{list(conflict_clause)}`')
                print()
            return conflict_clause, new_assignments

        if settings.DEBUG:
            print('    - Check T-SAT: `SAT`')

        if is_full_assignment:
            return conflict_clause, new_assignments

        # deduce next layers
        if settings.DEBUG:
            print(f'\n- Deduction')
        for node in implications:
            if settings.DEBUG:
                print(f'    - `node {node} <= 0`:', implications[node]['neg'])
            
            if implications[node]['neg']:
                new_assignments.append(-node)
                continue

            if settings.DEBUG:
                print(f'    - `node {node} > 0`:', implications[node]['pos'])

            if implications[node]['pos']:
                new_assignments.append(node)

        if settings.DEBUG:
            print(f'\n- New assignment: `{new_assignments}`')
            print()
        return conflict_clause, new_assignments


    def get_assignment(self) -> dict:
        pass

    def get_solution(self):
        return self.dnn_theorem_prover.solution

