from solver.theory_solver import TheorySolver
from simplex.simplex import Simplex
from simplex.parser import Parser
from pprint import pprint
import numpy as np

from sat_solver.sat_solver import SATSolver
from solver.solver import Solver

class TheorySolver(Solver):

    def __init__(self, formula, first_var=None, max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        self._solver = SATSolver(formula,
                                 first_var=first_var,
                                 max_new_clauses=max_new_clauses,
                                 halving_period=halving_period,
                                 theory_solver=self)


    def get_assignment(self) -> dict:
        # assignment: dict = {}
        # for variable, value in self._solver.iterable_assignment():
        #     if variable in self._tseitin_variable_to_subterm:
        #         assignment[self._tseitin_variable_to_subterm[variable]] = value
        # return assignment
        pass


    def solve(self) -> bool:
        return self._solver.solve()



class RealSolver(TheorySolver):

    def __init__(self, formula_str, max_new_clauses=float('inf'), halving_period=10000):


        self.parsed_input = Parser.parse(formula_str)
        print(self.parsed_input.formula)
        print(self.parsed_input.cnf)
        pprint(self.parsed_input.vars_dict)
        xi_int = self.parsed_input.vars_dict['xi']

        super().__init__(formula=self.parsed_input.formula, 
                         first_var=xi_int, 
                         max_new_clauses=max_new_clauses, 
                         halving_period=halving_period)

        # for clause in self._non_boolean_clauses:
        #     self._c = np.zeros(len(clause[1]), dtype=np.float64)
        #     break

        # self._tseitin_variable_to_np = {}
        # for clause in self._non_boolean_clauses:
        #     self._tseitin_variable_to_np[subterm_to_tseitin_variable[clause]] = {
        #         True: (np.array(clause[1], dtype=np.float64), np.array(clause[2], dtype=np.float64)),
        #         False: (-np.array(clause[1], dtype=np.float64), -np.array(clause[2] + epsilon, dtype=np.float64))
        #     }

        # pprint(self._tseitin_variable_to_np)


    def propagate(self):
        # print('propagate', list(self._solver.iterable_assignment()))
        conflict_clause, rows = set(), []
        pprint(self._solver._assignment)
        for var_int, value in self._solver.iterable_assignment():
            var_str = self.parsed_input.reversed_vars_dict[var_int if value else -var_int]

            if 'q' in var_str:
                print(var_int, var_str, value)
            else:
                print(var_int, var_str, value)

            if 'q' in var_str:
                rows.append(self.parsed_input.row_dict[var_str])

                conflict_clause.add(-var_int if value else var_int)

        print('===================>', rows, Simplex(self.parsed_input, rows).solve())

        if (not rows) or Simplex(self.parsed_input, rows).solve():
            conflict_clause = None
        else:
            conflict_clause = frozenset(conflict_clause)
            print('conflict_clause:', conflict_clause)
        return conflict_clause, []


    # dummy 
    def get_assignment(self):
        rows = []
        for var_int, value in self._solver.iterable_assignment():
            var_str = self.parsed_input.reversed_vars_dict[var_int if value else -var_int]
            if 'q' in var_str:
                rows.append(self.parsed_input.row_dict[var_str])
        s = Simplex(self.parsed_input, rows)
        s.solve()
        return s.get_assignment()
        
        # for variable, value in self._solver.iterable_assignment():
        #     if variable in self._tseitin_variable_to_np:
        #         a_matrix.append(self._tseitin_variable_to_np[variable][value][0])
        #         b.append(self._tseitin_variable_to_np[variable][value][1])
        # s = LinearSolver(np.array(a_matrix), np.array(b), self._c)
        # s.solve()
