from sat_solver.sat_solver import SATSolver
from solver.solver import Solver

class TheorySolver(Solver):

    def __init__(self, formula, tseitin_variable_to_subterm, non_boolean_clauses,
                 max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        self._formula = formula
        self._tseitin_variable_to_subterm = tseitin_variable_to_subterm
        self._non_boolean_clauses = non_boolean_clauses

        self._solver = SATSolver(formula,
                                 max_new_clauses=max_new_clauses,
                                 halving_period=halving_period,
                                 theory_solver=self)


    def get_assignment(self) -> dict:
        assignment: dict = {}
        for variable, value in self._solver.iterable_assignment():
            if variable in self._tseitin_variable_to_subterm:
                assignment[self._tseitin_variable_to_subterm[variable]] = value
        return assignment


    def solve(self) -> bool:
        return self._solver.solve()

