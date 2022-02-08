from pprint import pprint
import numpy as np
import copy

from sat_solver.custom_sat_solver import CustomSATSolver
from linear_solver.linear_solver import LinearSolver
from dnn_solver.helpers import DNNConstraint
from solver.solver import Solver
import settings

class TheorySolver(Solver):

    def __init__(self, formula, vars_mapping, layers_mapping, first_var=None, max_new_clauses=float('inf'), halving_period=10000):
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

    def __init__(self, dnn, vars_mapping, layers_mapping, conditions):

        super().__init__(formula=None, vars_mapping=vars_mapping, layers_mapping=layers_mapping)

        self.dnn = dnn
        self.conditions = conditions
        self.vars_mapping = vars_mapping
        self.reversed_vars_mapping = {v: k for k, v in vars_mapping.items()}

        self.constraint_generator = DNNConstraint(dnn, conditions)

    def propagate(self):
        if settings.DEBUG:
            print('- Theory propagate\n')

        new_assignments = []
        conflict_clause = set()

        assignment = {self.reversed_vars_mapping[k]: v['value'] for k, v in self._solver._assignment.items()}

        if settings.DEBUG:
            print('- Assignment:', {k: v['value'] for k, v in self._solver._assignment.items()})

        # theory checking
        theory_constraints, implication_constraints = self.constraint_generator(assignment)

        # print('----------------------------')
        # print(theory_constraints)
        # print('----------------------------')
        # print(implication_constraints)
        # print('----------------------------')

        if settings.DEBUG:
            print(f'\n- Theory constraints: `{theory_constraints}`')
        stat = LinearSolver(theory_constraints).solve()
        if not stat[0]:
            if settings.DEBUG:
                print('    - Check T-SAT: `UNSAT`')
            for variable, value in self._solver.iterable_assignment():
                conflict_clause.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            if settings.DEBUG:
                print(f'    - Conflict clause: `{list(conflict_clause)}`')
                print()
            return conflict_clause, []

        if settings.DEBUG:
            print('    - Check T-SAT: `SAT`')
        conflict_clause = None
 
        # deduce next layers
        if settings.DEBUG:
            print(f'\n- Deduction')
        for node, constraints in implication_constraints.items():
            constraint_neg, constraint_pos = constraints

            if settings.DEBUG:
                print(f'    - Deduction: `{node} <= 0`')
                print(f'    - Constraints: `{constraint_neg}`')
            stat_neg = LinearSolver(constraint_neg).solve()
            if not stat_neg[0]:
                if settings.DEBUG:
                    print('        - Result:', True)
                new_assignments.append(-self.vars_mapping[node])
                continue
            else:
                if settings.DEBUG:
                    print('        - Result:', False)

            if settings.DEBUG:
                print(f'    - Deduction: `{node} > 0`')
                print(f'    - Constraints: `{constraint_pos}`')
            stat_pos = LinearSolver(constraint_pos).solve()
            if not stat_pos[0]:
                if settings.DEBUG:
                    print('        - Result:', True)
                new_assignments.append(self.vars_mapping[node])
            else:
                if settings.DEBUG:
                    print('        - Result:', False)

        if settings.DEBUG:
            print(f'    - New assignment: `{new_assignments}`')
            print()
        return conflict_clause, new_assignments




    def get_assignment(self) -> dict:
        pass
