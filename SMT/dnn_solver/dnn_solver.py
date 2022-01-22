from solver.theory_solver import TheorySolver
from simplex.simplex import Simplex
from simplex.parser import Parser
from pprint import pprint
import numpy as np
import copy

from dnn_solver.helpers import Utils, DNNConstraint
from sat_solver.sat_solver_2 import SATSolver2
from real_solver.real_solver import RealSolver
from solver.solver import Solver

class TheorySolver(Solver):

    def __init__(self, formula, vars_mapping, first_var=None, max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        self._solver = SATSolver2(formula,
                                  vars_mapping,
                                  max_new_clauses=max_new_clauses,
                                  halving_period=halving_period,
                                  theory_solver=self)

    def get_assignment(self) -> dict:
        pass

    def solve(self) -> bool:
        return self._solver.solve()




class DNNSolver(TheorySolver):

    def __init__(self, dnn, vars_mapping, conditions):

        super().__init__(formula=None, vars_mapping=vars_mapping)

        self.dnn = dnn
        self.conditions = conditions
        self.vars_mapping = vars_mapping
        self.reversed_vars_mapping = {v: k for k, v in vars_mapping.items()}

        self.constraint_generator = DNNConstraint(dnn, conditions)

    def propagate(self):
        print('- Theory propagate\n')

        new_assignments = []
        conflict_clause = set()

        assignment = {self.reversed_vars_mapping[k]: v['value'] for k, v in self._solver._assignment.items()}

        print('- Assignment:', assignment)

        # theory checking
        theory_constraints, implication_constraints = self.constraint_generator(assignment)

        # print('----------------------------')
        # print(theory_constraints)
        # print('----------------------------')
        # print(implication_constraints)
        # print('----------------------------')

        print('\n- Theory constraints:', theory_constraints)
        stat = RealSolver(theory_constraints).solve()
        if not stat[0]:
            print('    - Check T-SAT: `UNSAT`')
            for variable, value in self._solver.iterable_assignment():
                conflict_clause.add(-variable if value else variable)
            conflict_clause = frozenset(conflict_clause)
            print(f'    - Conflict clause: `{list(conflict_clause)}`')
            print()
            return conflict_clause, []

        print('    - Check T-SAT: `SAT`')
        conflict_clause = None
 
        # deduce next layers
        print(f'\n- Deduction')
        for node, constraints in implication_constraints.items():
            constraint_neg, constraint_pos = constraints

            print(f'    - Deduction: `{node} <= 0`')
            print(f'    - Constraints: `{constraint_neg}`')
            stat_neg = RealSolver(constraint_neg).solve()
            if not stat_neg[0]:
                new_assignments.append(-self.vars_mapping[node])
                continue
            print(f'    - Deduction: `{node} > 0`')
            print(f'    - Constraints: `{constraint_pos}`')
            stat_pos = RealSolver(constraint_pos).solve()
            if not stat_pos[0]:
                print('    - Result:', True)
                new_assignments.append(self.vars_mapping[node])

        print(f'    - New assignment: `{new_assignments}`')
        print()
        return conflict_clause, new_assignments




    def get_assignment(self) -> dict:
        pass
