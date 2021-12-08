import sys
import random

class Clause(object):
    """A Boolean clause randomly generated"""

    def __init__(self, num_vars, clause_length):
        """
        Initialization
            length: Clause length
            lits: List of literals
        """
        self.length = clause_length
        self.lits = None
        self.gen_random_clause(num_vars)

    def gen_random_clause(self, num_vars):
        self.lits = []
        length = random.randint(self.length//2, self.length)
        while len(self.lits) < length:  # Set the variables of the clause
            new_lit = random.randint(1, num_vars)  # New random variable
            if new_lit not in self.lits:  # If the variable is not already in the clause
                self.lits.append(new_lit)  # Add it to the clause
        for i in range(len(self.lits)):  # Sets a negative sense with a 50% probability
            if random.random() < 0.5:
                self.lits[i] *= -1  # Change the sense of the literal

    def __str__(self):
        return " ".join(map(str, self.lits)) + " 0" 


class DimacsGenerator(object):
    """A CNF formula randomly generated"""

    def __init__(self, num_vars, num_clauses, clause_length):
        """
        Initialization
            num_vars: Number of variables
            num_clauses: Number of clauses
            clause_length: Length of the clauses
            clauses: List of clauses
        """
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.clause_length = clause_length
        self.clauses = None
        self.gen_random_clauses()

    def gen_random_clauses(self):
        self.clauses = []
        while len(self.clauses) < self.num_clauses:
            clause = Clause(self.num_vars, self.clause_length)
            clause = str(clause)
            if clause not in self.clauses:
                self.clauses.append(clause)

    def show(self):
        print("c Random CNF formula")
        print(f"p cnf {self.num_vars} {self.num_clauses}")
        for clause in self.clauses:
            print(clause)


    def export(self, filename):
        with open(filename, 'w') as f:
            f.write("c Random CNF formula\n")
            f.write(f"p cnf {self.num_vars} {self.num_clauses}\n")
            for clause in self.clauses:
                f.write(clause + '\n')

if __name__ == '__main__':
    num_vars = 5
    num_clauses = 7
    clause_length = 5

    cnf_formula = DimacsGenerator(num_vars, num_clauses, clause_length)
    cnf_formula.show()
    cnf_formula.export('../data/dimacs_cnf.txt')