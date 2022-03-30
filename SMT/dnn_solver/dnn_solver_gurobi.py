from pprint import pprint
import numpy as np
import copy

from sat_solver.custom_sat_solver import CustomSATSolver
from linear_solver.linear_solver import LinearSolver
from dnn_solver.dnn_constraint import DNNConstraintGurobi as DNNConstraint
from solver.solver import Solver
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

    def __init__(self, dnn, vars_mapping, layers_mapping):

        super().__init__(formula=None, vars_mapping=vars_mapping, layers_mapping=layers_mapping)

        # self.dnn = dnn
        self.constraint_generator = DNNConstraint(dnn, copy.deepcopy(layers_mapping))

    def propagate(self):
        if settings.DEBUG:
            print('- Theory propagate\n')

        conflict_clause = None
        new_assignments = []

        assignment = {k: v['value'] for k, v in self._solver._assignment.items()}

        if settings.DEBUG:
            print('- Assignment:', assignment)

        # theory checking
        theory_stat, implication_constraints = self.constraint_generator(assignment)

        if not theory_stat:
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
 
        # deduce next layers
        if settings.DEBUG:
            print(f'\n- Deduction')
        for node in implication_constraints:
            if settings.DEBUG:
                print(f'    - `node {node} <= 0`:', implication_constraints[node]['neg'])
            
            if implication_constraints[node]['neg']:
                new_assignments.append(-node)
                continue

            if settings.DEBUG:
                print(f'    - `node {node} > 0`:', implication_constraints[node]['pos'])

            if implication_constraints[node]['pos']:
                new_assignments.append(node)

        if settings.DEBUG:
            print(f'\n- New assignment: `{new_assignments}`')
            print()
        return conflict_clause, new_assignments


    def get_assignment(self) -> dict:
        pass
