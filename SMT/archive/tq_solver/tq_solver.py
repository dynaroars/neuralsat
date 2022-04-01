from tq_solver.linear_solver.linear_solver import LinearSolver
from solver.theory_solver import TheorySolver
from pprint import pprint
import numpy as np

class TQSolver(TheorySolver):

    def __init__(self, formula, tseitin_mappings, non_boolean_clauses, epsilon=np.float64(1e-5),
                 max_new_clauses=float('inf'), halving_period=10000):

        tseitin_variable_to_subterm, subterm_to_tseitin_variable = tseitin_mappings


        # # debug
        # tseitin_variable_to_subterm = {2: ('<=', (1.0,), -1.0), 3: ('<=', (-1.0,), 3.0)}
        
        # subterm_to_tseitin_variable = {('<=', (1.0,), -1.0): 2, ('<=', (-1.0,), 3.0): 3}

        # non_boolean_clauses = set({('<=', (-1.0,), 3.0), ('<=', (1.0,), -1.0)})
        # formula = frozenset({frozenset({1, -3, -2}), frozenset({2, -1}), frozenset({3, -1})})

        super().__init__(formula, tseitin_variable_to_subterm, non_boolean_clauses, max_new_clauses, halving_period)

        for clause in self._non_boolean_clauses:
            self._c = np.zeros(len(clause[1]), dtype=np.float64)
            break

        self._tseitin_variable_to_np = {}
        for clause in self._non_boolean_clauses:
            self._tseitin_variable_to_np[subterm_to_tseitin_variable[clause]] = {
                True: (np.array(clause[1], dtype=np.float64), np.array(clause[2], dtype=np.float64)),
                False: (-np.array(clause[1], dtype=np.float64), -np.array(clause[2] + epsilon, dtype=np.float64))
            }


    def propagate(self):
        conflict_clause, a_matrix, b = set(), [], []
        for variable, value in self._solver.iterable_assignment():
            if variable in self._tseitin_variable_to_np:
                a_matrix.append(self._tseitin_variable_to_np[variable][value][0])
                b.append(self._tseitin_variable_to_np[variable][value][1])
                conflict_clause.add(-variable if value else variable)

        if (not a_matrix) or LinearSolver(np.array(a_matrix), np.array(b), self._c).is_sat():
            conflict_clause = None
        else:
            conflict_clause = frozenset(conflict_clause)
        return conflict_clause, []


    # dummy 
    def get_assignment(self):
        a_matrix, b = [], []
        for variable, value in self._solver.iterable_assignment():
            if variable in self._tseitin_variable_to_np:
                a_matrix.append(self._tseitin_variable_to_np[variable][value][0])
                b.append(self._tseitin_variable_to_np[variable][value][1])
        s = LinearSolver(np.array(a_matrix), np.array(b), self._c)
        s.solve()
        return s.get_assignment()