from util.utils import dimacs_parse as parse
from solver.dpll import solve
from solver.cdcl import CDCL

def test_dpll():
    formula, nvars = parse('data/dimacs_cnf.txt')
    print(formula)
    print('solve:')
    solve(formula, nvars)


def test_cdcl():

    solver = CDCL()
    solver.solve('data/dimacs_cnf.txt')
    solver.stats.print_stats()


if __name__ == '__main__':
    test_cdcl()