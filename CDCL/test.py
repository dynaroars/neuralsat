from util.dimacs_cnf_gen import DimacsGenerator
from util.utils import dimacs_parse as parse
from solver.dpll import solve
from solver.cdcl import CDCL

def test_dpll():
    formula, nvars = parse('data/dimacs_cnf.txt')
    print(formula)
    print('solve:')
    solve(formula, nvars)


def test_cdcl():
    filename = 'data/dimacs_cnf.txt'
    # problem = DimacsGenerator(num_vars=10, num_clauses=10, clause_length=5)
    # problem.export(filename)

    solver = CDCL()
    solver.solve(filename)
    # solver.stats.print_stats()

    # print('v ' + ' '.join([f'{var}' if solver._variable_to_assignment_nodes[var].value else f'{-var}' for var in solver._variable_to_assignment_nodes]) + ' 0')

    # print(solver._variable_to_assignment_nodes)
    print(solver.unsat_core)

if __name__ == '__main__':
    test_cdcl()