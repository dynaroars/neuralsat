from utils import dimacs_parse as parse
from settings import *
import random

class Clause:

    def __init__(self, clause):
        self.clause = clause
        self.value = 0 # 0 = UNASSIGNED, 1 = TRUE, -1 = FALSE
        self.size = len(self.clause)
        self.decision_level = [-1 for _ in self.clause]

        self.remove_redundant_literals()

    def remove_redundant_literals(self):
        'If contains two opposites literals => TRUE'
        for l in self.clause:
            if -l in self.clause:
                self.value = 1 #TRUE
                self.size = 0
                break

class LazyClause:
    def __init__(self, clause):
        self.clause = clause
        self.value = 0 # 0 = UNASSIGNED, 1 =  TRUE, -1 = FALSE

class CNFFormula:
    def __init__(self, list_clause):
        self.formula = [LazyClause(c) for c in list_clause if len(c) > 0]
        self.value = self.get_value()

    def get_value(self):
        list_values = [c.value for c in self.formula]
        if -1 in list_values:
            return -1
        if 0 in list_values:
            return 0
        return 1 

class CDCL: 
    def __init__(self, input_cnf_file):
        self.list_clause, self.nvars = parse(input_cnf_file)
        self.formula = CNFFormula(self.list_clause)


if __name__ == '__main__':
    CDCL('dimacs_cnf.txt')