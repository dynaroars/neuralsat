from utils.formula_parser import FormulaParser
from tq_solver.tq_solver import TQSolver
from solver.solver import Solver

class SMTSolver(Solver):
    
    def __init__(self, formula=None, max_new_clauses=float('inf'), halving_period=10000):
        super().__init__()

        f, (tseitin_variable_to_subterm, subterm_to_tseitin_variable), non_boolean_clauses = FormulaParser.import_tq(formula)
        self._solver = TQSolver(
            *FormulaParser.import_tq(formula),
            max_new_clauses=max_new_clauses, 
            halving_period=halving_period
        )


    def get_assignment(self):
        return self._solver.get_assignment()


    def solve(self) -> bool:
        return self._solver.solve()