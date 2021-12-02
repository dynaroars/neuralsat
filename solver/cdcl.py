from heuristic.decide.vsids import PriorityQueue
from util.utils import dimacs_parse as parse
from settings import *

from collections import OrderedDict
import random
import json
import time
import sys
import os

class Statistics:

    "Class used to store the various statistics measuerd while solving"

    def __init__(self):
        # Input file in which the problem is stored
        self._input_file = "" 
        # Result of the SAT solver (SAT or UNSAT)
        self._result = ""
        # Path of files used to store the statistics for the solved problem
        self._output_statistics_file = ""
        self._output_assignment_file = ""
        # Number of variables in the problem
        self._num_vars = 0
        # Original number of clauses present in the problem
        self._num_orig_clauses = 0
        # Number of original clauses stored to get the assignment
        self._num_clauses = 0    
        # Number of clauses learned by the solver during the conflict analysis
        self._num_learned_clauses = 0
        # Number of decisions made by the solver
        self._num_decisions = 0
        # Number of implications made by the  solver 
        self._num_implications = 0
        # Time at which the solver starts solving the problem
        self._start_time = 0
        # Time at which the solver is done reading the problem
        self._read_time = 0
        # Time at which the solver has completed solving the problem
        self._complete_time = 0
        # Time which the solver spend while performing BCP
        self._bcp_time = 0
        # Time which the solver spend while deciding (in _decide method)
        self._decide_time = 0
        # Time which the solver spend while analyzing the conflicts
        self._analyze_time = 0
        # Time which the solver spend while backtracking
        self._backtrack_time = 0
        # Number of restarts
        self._restarts = 0
    
    def print_stats(self):
        "Print the stored statistics with appropriate labels of what the stats signify"

        print("=========================== STATISTICS ===============================")
        print("Solving formula from file:", self._input_file)
        print(f"Vars: {self._num_vars}, Clauses: {self._num_orig_clauses}, Stored clauses: {self._num_clauses}")
        print("Input Reading Time:", self._read_time - self._start_time)
        print("-------------------------------")
        print("Restarts:", self._restarts)
        print("Learned clauses:", self._num_learned_clauses)
        print("Decisions made:", self._num_decisions)
        print("Implications made:", self._num_implications)
        print("Time taken:", self._complete_time - self._start_time)
        print("----------- Time breakup ----------------------")
        print("BCP Time:", self._bcp_time)
        print("Decide Time:", self._decide_time)
        print("Conflict Analyze Time:", self._analyze_time)
        print("Backtrack Time:", self._backtrack_time)
        print("-------------------------------")
        print("RESULT:", self._result)
        print("Statistics stored in file:", self._output_statistics_file)
        
        if self._result == "SAT":
            print("Satisfying Assignment stored in file:", self._output_assignment_file)
        print("======================================================================")  



class AssignedNode:
    
    "Class to store the information about the variables being assigned"

    def __init__(self, var, value, level, clause): 
        # variable that is assigned
        self.var = var 
        # value assigned to the variable (True/False)
        self.value = value 
        # level at which the variable is assigned
        self.level = level 
        # The index of the clause which implies `var` if `var` is assigned through Implication
        # If var is decided, this is set to None
        self.clause = clause 
        # Index at which a node is placed in the `assignment_stack`
        # Initially it is -1 when node is created and has to be updated when pushed in `assignment_stack`.
        self.index = -1

    def __str__(self):
        return f'Var: {self.var}, Value: {self.value}, Level: {self.level}, \
            Clause: {self.clause}, Index: {self.index}'

class CDCL: 

    "Conflict-Driven Clause Learning (CDCL) Solver"

    def __init__(self, decider=DECIDER, restarter=RESTARTER):
        # Number of clauses stored
        self._num_clauses = 0 
        # Number of variables
        self._num_vars = 0   
        # Decision level (level at which the solver is in backtracking tree)
        self._level = 0
        # List of clauses where each clause is stored as a list of literals 
        self._clauses = []

        # Mapping a literal to the list of clauses the literal watches
        self._clauses_watched_by_l = {}
        # Mapping a clause to the list of literals that watch the clause
        self._literals_watching_c = {}

        # Mapping the variables to their assignment nodes
        # which contains the information about the value of the variable,
        # the clause which implied the variable (if it is implied),
        # and the level at which the variable is set
        self._variable_to_assignment_nodes = {}

        # A stack that stores the assignment nodes in order of their assignment
        self._assignment_stack = []
        
        # The decision heuristic to be used while solving the SAT problem
        assert decider in SUPPORTED_DECIDER, f'Invalid decider "{decider}"'
        self._decider = decider

        # Restart strategy
        if restarter == None:
            self._restarter = None
        else:
            assert restarter in SUPPORTED_RESTARTER, f'Invalid restarter "{restarter}"'
            if restarter == "GEOMETRIC":
                # Initialize the conflict limit
                self._conflict_limit = CONFLICT_LIMIT
                # Conflict limit will be multiplied after each restart
                self._limit_mult = LIMIT_MULT
                # Number of conflicts before restart
                self._conflicts_before_restart = 0
            elif restarter == 'LUBY':
                # Initialize by reset the luby sequencer
                reset_luby()
                # Base as 512
                self._luby_base = LUBY_BASE
                # Initialize the conflict limit
                self._conflict_limit = self._luby_base * get_next_luby_number()
                # Number of conflicts before restart
                self._conflicts_before_restart = 0
            self._restarter = restarter

        self.stats = Statistics()

    def _add_clause(self, clause):
        # remove duplicate instances
        # clause = list(OrderedDict.fromkeys(clause))

        # Unary clause
        if len(clause) == 1:
            # Get the literal
            lit = clause[0]
            value_to_set = lit > 0
            var = abs(lit)

            if var not in self._variable_to_assignment_nodes:
                # Increment the number of implications as it is an implication
                self.stats._num_implications += 1
                # Set clause None as we are not storing this clause
                node = AssignedNode(var=var, value=value_to_set, level=0, clause=None)
                # Set the node with var in the dictionary and push it in the stack
                self._variable_to_assignment_nodes[var] = node
                self._assignment_stack.append(node)
                # Set the index of the node to the position in stack at which it is pushed
                node.index = len(self._assignment_stack) - 1
            else:
                # If the variable is assigned, get its node
                node = self._variable_to_assignment_nodes[var]
                # If the set value does not match with the value_to_set, we have an contradiction 
                # and this has happened because of 2 conflicting unary clauses in the problem
                if node.value != value_to_set:
                    self.stats._result = "UNSAT"
                    # Return 0 to indicate that the problem has been solved. Proven UNSAT
                    return 0
            # Everything normal
            return 1

        clause_with_literals = []
        for lit in clause:
            var = abs(lit)
            if lit < 0:
                # Add num_vars to it to get the literal
                clause_with_literals.append(var + self._num_vars)
                if self._decider == "VSIDS":
                    # Score the literal appearing in the clause
                    self._lit_scores[var + self._num_vars] += 1
                elif self._decider == "MINISAT":
                    # Score the variable appearing in the clause
                    self._var_scores[var] += 1
            else:
                clause_with_literals.append(var)
                if self._decider == "VSIDS":
                    # Score the literal appearing in the clause
                    self._lit_scores[var] += 1
                elif self._decider == "MINISAT":
                    # Score the variable appearing in the clause
                    self._var_scores[var] += 1    

        # Set clause id to the number of clauses
        clause_id = self._num_clauses
        # Append the new clause to the clause list
        self._clauses.append(clause_with_literals)
        # Increase the clause counter
        self._num_clauses += 1
        
        # Make the first 2 literals as watch literals for this clause
        watch_literal1 = clause_with_literals[0]
        watch_literal2 = clause_with_literals[1]

        # Set the watch literals for the clause to the list containing the 2 watchers
        self._literals_watching_c[clause_id] = (watch_literal1, watch_literal2)
        # Add this clause_id to the watched clauses list of both the watchers
        self._clauses_watched_by_l.setdefault(watch_literal1,[]).append(clause_id)
        self._clauses_watched_by_l.setdefault(watch_literal2,[]).append(clause_id)

        # Everything normal
        return 1

    def _read_dimacs_cnf_file(self, input_cnf_file):
        clauses, num_vars = parse(input_cnf_file)
        self._num_vars = num_vars

        if self._decider == "VSIDS":
            self._lit_scores = [0 for i in range(2*self._num_vars+1)]  
        elif self._decider == "MINISAT":
            self._var_scores = [0 for i in range(self._num_vars+1)]
            self._phase = [0 for i in range(self._num_vars+1)]

        # Store the original number of clauses
        self.stats._num_orig_clauses = len(clauses)

        for clause in clauses:
            ret = self._add_clause(clause)
            # If 0 is returned, then stop as the problem is proved UNSAT
            if ret == 0:
                break

        # If the VSIDS decider is used
        if self._decider == "VSIDS":
            # Create a priority queue (max priority queue)
            self._priority_queue = PriorityQueue(self._lit_scores)
            # Quantity by which the scores of a literal will be increased when it is found in a conflict clause
            self._incr = INCREASE

            # Remove variables already assigned in the unary clauses, 
            for node in self._assignment_stack:
                self._priority_queue.remove(node.var)
                self._priority_queue.remove(node.var + self._num_vars)

        elif self._decider == "MINISAT":
            # Create a priority queue (max priority queue)
            self._priority_queue = PriorityQueue(self._var_scores)
            # Quantity by which the scores of a literal will be increased when it is found in a conflict clause
            self._incr = INCREASE
            # Scores will decay after each conflict
            self._decay = DECAY

            # Remove variables already assigned in the unary clauses, 
            for node in self._assignment_stack:
                self._priority_queue.remove(node.var)

    def solve(self, input_cnf_file):
        self.stats._input_file = input_cnf_file
        self.stats._start_time = time.time()

        # Read the input and process the clauses
        self._read_dimacs_cnf_file(input_cnf_file)
        
        self.stats._read_time = time.time()
        self.stats._num_vars = self._num_vars
        self.stats._num_clauses = self._num_clauses

        if self.stats._result == "UNSAT":
            self.stats._complete_time = time.time()
        else:
            first_time = True
