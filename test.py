from utils import dimacs_parse as parse
from dpll import *

def main():
    formula, nvars = parse('dimacs_cnf.txt')
    print(formula)
    print('solve:')
    solve(formula, nvars)

if __name__ == '__main__':
    main()