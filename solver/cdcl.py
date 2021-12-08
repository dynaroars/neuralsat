from heuristic.decide.priority_queue import PriorityQueue
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
        self._literals_watching_c[clause_id] = [watch_literal1, watch_literal2]
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
            
        if DEBUG:
            print('\n###### Read problem\n')
            print(f'- `num_clauses={self._num_clauses}`, `num_vars={self._num_vars}`')
            print('- literals watching clause:')
            # print(self._literals_watching_c)
            for c in self._literals_watching_c:
                print(f'    - clause {c} => literals: {self._literals_watching_c[c]}')
            print('- clauses watched by literal:')
            # print(self._clauses_watched_by_l)
            for l in self._clauses_watched_by_l:
                print(f'    - literal {l} => clauses: {self._clauses_watched_by_l[l]}')

    def _is_negative_literal(self, literal):
        return literal > self._num_vars

    def _get_var_from_literal(self, literal):
        'Gives the variable corresponding to literal.'
        # If the literal is negative, then _num_vars was added to  the variable to get the literal
        if self._is_negative_literal(literal):
            return literal - self._num_vars
        # If the literal is positive, it is same as the variable
        return literal

    def _boolean_constraint_propogation(self, is_first_time):
        "Main method that makes all the implications."
        '''
        - 2 cases:
            + run for the first time: traverse through all and make further implications
                                      because we can have many decisions already made due 
                                      to the implications by unary clauses.
            + otherwise: we only have to take the last made decision into account 
                         and make the implications and so we start from the last 
                         node in the assignment stack.
        '''
        if DEBUG:
            print(f'\n###### BCP `level={self._level}`\n')
            print('- `assignment_stack={}`'.format([(x.var, x.value, x.level) for x in self._assignment_stack]))


        # Point to the last decision
        last_assignment_pointer = 0 if is_first_time else len(self._assignment_stack) - 1

        # Traverse through all the assigned nodes in the stack 
        while last_assignment_pointer < len(self._assignment_stack):
            # Get the assigned node
            last_assigned_node = self._assignment_stack[last_assignment_pointer]
            # If the variable's value was set to True, then negative literal corresponding to the variable is falsed,
            # Else if it set False, the positive literal is falsed
            literal_that_is_falsed = last_assigned_node.var + self._num_vars if \
                last_assigned_node.value == True else last_assigned_node.var
            # print('last_assigned_node', last_assigned_node.var, literal_that_is_falsed)



            # Change the watch literals for all clauses watched by literal_that_is_falsed
            # Get the list of clauses watched by the falsed literal
            clauses_watched_by_falsed_literal = self._clauses_watched_by_l.setdefault(literal_that_is_falsed, []).copy()

            if DEBUG:
                print(f'- `last_assigned_var={last_assigned_node.var}`, `negation_literal={literal_that_is_falsed}`')
                print(f'- clause watched by negation literal:', clauses_watched_by_falsed_literal)
            # print('clauses_watched_by_falsed_literal', clauses_watched_by_falsed_literal, self._clauses_watched_by_l)
            # Traverse through them and find a new watch literal and if we are unable to find a new watch literal, we have an implication
            # If other watch literal is set to a value opposite of what is implied, we have a conflict
            itr = 0
            while itr < len(clauses_watched_by_falsed_literal):

                # Get the clause and its watch list
                clause_id = clauses_watched_by_falsed_literal[itr]
                watch_list_of_clause = self._literals_watching_c[clause_id]
                # print('clause_id', clause_id, watch_list_of_clause)

                # Get the other watch literal for this clause (other than the falsed one)
                other_watch_literal = watch_list_of_clause[0]
                if other_watch_literal == literal_that_is_falsed:
                    other_watch_literal = watch_list_of_clause[1]


                # Get the variable corresponding to the other watch literal
                other_watch_var = self._get_var_from_literal(other_watch_literal)
                is_negative_other = self._is_negative_literal(other_watch_literal)
                # print('other_watch_literal', other_watch_literal, other_watch_var, is_negative_other)

                if DEBUG:
                    print(f'- `clause={clause_id}`, `clause={self._clauses[clause_id]}`')
                    print(f'    - `other_literal={other_watch_literal}`, `other_var={other_watch_var}`')

                # If other watch literal is set, and is set so as to make clause to be true, then move to the next clause
                if other_watch_var in self._variable_to_assignment_nodes:
                    value_assgned = self._variable_to_assignment_nodes[other_watch_var].value
                    if (is_negative_other and value_assgned==False) or (not is_negative_other and value_assgned==True):
                        itr += 1
                        continue

                # We need to find a new literal to watch
                new_literal_to_watch = -1
                clause = self._clauses[clause_id]
                # print(clause_id, clause, watch_list_of_clause, self._variable_to_assignment_nodes)
                # Traverse through all literals
                for lit in clause:
                    if lit not in watch_list_of_clause:
                        var_of_lit = self._get_var_from_literal(lit)
                        if var_of_lit not in self._variable_to_assignment_nodes:
                            # If the literal is not set, it can be used as a watcher
                            new_literal_to_watch = lit
                            break
                        else:
                            # If the literal's variable is set in such a way that the literal is true
                            # we use it as new watcher as anyway the clause is satisfied
                            node = self._variable_to_assignment_nodes[var_of_lit]
                            is_negative = self._is_negative_literal(lit)
                            if (is_negative and node.value==False) or (not is_negative and node.value==True):
                                new_literal_to_watch = lit
                                break

                # print('new_literal_to_watch', new_literal_to_watch)



                # If new_literal_to_watch is not -1, then it means that we have a new literal to watch the clause
                if new_literal_to_watch != -1:
                    if DEBUG:
                        print(f'    - `clause={clause_id}` => 2-watched literals = {self._literals_watching_c[clause_id]}')

                    # Remove the falsed literal and add the new literal to watcher list of the clause
                    self._literals_watching_c[clause_id].remove(literal_that_is_falsed)
                    self._literals_watching_c[clause_id].append(new_literal_to_watch)

                    # Remove clause from the watched clauses list of the falsed literal
                    # and add it to the watched clauses list of the new literal
                    self._clauses_watched_by_l.setdefault(literal_that_is_falsed,[]).remove(clause_id)
                    self._clauses_watched_by_l.setdefault(new_literal_to_watch,[]).append(clause_id)

                    if DEBUG:
                        print(f'    - `new literal={new_literal_to_watch}`')
                        print(f'    - `clause={clause_id}` => 2-watched literals = {self._literals_watching_c[clause_id]}')

                else:
                    if DEBUG:
                        print(f'    - `new literal={new_literal_to_watch}` => implications value of `other_var={other_watch_var}`')
                        print('    - `assignment_stack={}`'.format([(x.var, x.value, x.level) for x in self._assignment_stack]))

                    if other_watch_var not in self._variable_to_assignment_nodes:
                        # We get no other watcher that means all the literals other than the other_watch_literal are false 
                        # and the other_watch_literal has to be made true for this clause to be true. 
                        # This is possible in this case as variable corresponding to the other_watch_literal is not set.

                        # Get the value to set the variable as not of if the other watch literal is negative.
                        value_to_set = not is_negative_other

                        # Create the AssignedNode with the variable, value, level and clause_id
                        assign_var_node = AssignedNode(other_watch_var, value_to_set, self._level, clause_id)
                        self._variable_to_assignment_nodes[other_watch_var] = assign_var_node

                        # Push the created node in the assignment stack and set its index to the position at which it is pushed.
                        self._assignment_stack.append(assign_var_node)
                        assign_var_node.index = len(self._assignment_stack) - 1

                        # If the VSIDS decider is used, then remove the 2 literals corresponding to the variable implied above
                        # as we maintain only the unassigned variables in the priority queue
                        if self._decider == "VSIDS":
                            self._priority_queue.remove(other_watch_var)
                            self._priority_queue.remove(other_watch_var + self._num_vars)
                        # If MINISAT decider is used, remove the variable which is now set from the priority queue 
                        # as we only maintain the unassigned varibles in the priority queue
                        elif self._decider == "MINISAT":
                            self._priority_queue.remove(other_watch_var)
                            # Use the value_to_set to set the phase of the variable
                            self._phase[other_watch_var] = int(value_to_set)

                        # Increment the number of implications in the stats object by 1
                        self.stats._num_implications += 1

                        if DEBUG:
                            print(f'    - `other_var={other_watch_var}`, `value={value_to_set}`')


                    # CONFLICT !!!
                    else:
                        # print('Conflict clause_id =', clause_id)
                        # If the GEOMETRIC restart strategy is used
                        if self._restarter == "GEOMETRIC":
                            # Increase the conflicts_before_restart by 1 as we have encountered a conflict
                            self._conflicts_before_restart += 1

                            # If the number of conflicts reach the limit, we RESTART
                            if self._conflicts_before_restart >= self._conflict_limit:
                                # Increment the restart counter in the stats object
                                self.stats._restarts += 1
                                # Set the conflicts before restart to 0 
                                self._conflicts_before_restart = 0
                                # Multiply the conflict limit by the predefined limit multiplier
                                self._conflict_limit *= self._limit_mult
                                if DEBUG:
                                    print('- return `RESTART`')
                                return "RESTART"
                        # If the LUBY restart strategy is used
                        elif self._restarter == "LUBY":
                            # Increase the conflicts_before_restart by 1 as we have encountered a conflict
                            self._conflicts_before_restart += 1

                            # If the number of conflicts reach the limit, we RESTART
                            if self._conflicts_before_restart >= self._conflict_limit:
                                # Increment the restart counter in the stats object
                                self.stats._restarts += 1
                                # Set the conflicts before restart to 0 
                                self._conflicts_before_restart = 0
                                # Multiply the conflict limit by the next luby number
                                self._conflict_limit = self._luby_base * get_next_luby_number()
                                if DEBUG:
                                    print('- return `RESTART`')
                                return "RESTART"

                        # Conflict is detected as the other_watch_literal is assigned
                        conflict_node = AssignedNode(None, None, self._level, clause_id)
                        self._assignment_stack.append(conflict_node)
                        # Set its index to the position at which it is pushed
                        conflict_node.index = len(self._assignment_stack) - 1

                        if DEBUG:
                            print(f'    - Conflict: `level={self._level}`, `clause={clause_id}`')
                            print('- return `CONFLICT`')
                        return "CONFLICT"

                # Increment itr to get the next clause
                itr += 1

            # Increment last_assignment_pointer to get the next assigned node to be used to make the implications
            last_assignment_pointer += 1

        if DEBUG:
            print('- return `NO_CONFLICT`')
        # If the loop finishes successfully, it means all the implications have been made without any conflict
        return "NO_CONFLICT"

    def _is_valid_clause(self, clause, level):
        'Checks if the passed clause is a valid conflict clause (with only one literal set at level)'
        
        # Count the literals set at level
        counter = 0
        # Store the maximum index of the literals encountered
        maxi = -1
        # Candidate literal that is assigned the latest at level
        cand = -1

        # For all literals in the clause, et the assignment node corresponding the variable of the literal
        for lit in clause:
            var = self._get_var_from_literal(lit)
            node = self._variable_to_assignment_nodes[var]
            
            # If the level at which the node is assigned is same as the passed level
            if node.level == level:
                # Increase the counter of literals assigned at passed level by 1
                counter += 1
                # Find the latest assigned node at this level
                if node.index > maxi:
                    maxi = node.index
                    cand = node

        return counter==1, cand

    def _binary_resolute(self, clause1, clause2, var):
        # Add the clause 1 and clause 2
        full_clause = clause1 + clause2
        # Remove duplicated literals
        full_clause = list(OrderedDict.fromkeys(full_clause))
        # Remove the positive and the negative literals from the combined list to resolvent clause
        full_clause.remove(var)
        full_clause.remove(var+self._num_vars)
        return full_clause 

    def _get_backtrack_level(self, conflict_clause, conflict_level):
        # Stores the backtrack level
        maximum_level_before_conflict_level = -1
        # Stores the only literal in the conflict_clause which is assigned at the conflict_level
        literal_at_conflict_level = -1

        for lit in conflict_clause:
            var = self._get_var_from_literal(lit)
            assigned_node = self._variable_to_assignment_nodes[var]
            # If the node's level is the conflict_level, set this lit to literal_at_conflict_level
            if assigned_node.level == conflict_level:
                literal_at_conflict_level = lit
            # Else, find the maximum of all levels other than the conflict level
            else:
                if assigned_node.level > maximum_level_before_conflict_level:
                    maximum_level_before_conflict_level = assigned_node.level
        # Return the backtrack level and the literal at conflict level
        print(conflict_clause, conflict_level, literal_at_conflict_level)
        return maximum_level_before_conflict_level, literal_at_conflict_level


    def _analyze_conflict(self):
        "Analyzes the conflict occurs during the Boolean Constrain Propogation (BCP)."

        if DEBUG:
            print(f'\n###### Analyze-Conflict `level={self._level}`\n')

        # The last node in the assignment stack is a conflict node
        conflict_node = self._assignment_stack[-1]
        # The conflict node is used to get the conflict level and the clause that caused the conflict
        conflict_level = conflict_node.level
        conflict_clause = self._clauses[conflict_node.clause]
        # remove conflict node from the assignment stack
        self._assignment_stack.pop()

        if DEBUG:
            print(f'- `conflict_level={conflict_level}`, `conflict_clause_id={conflict_node.clause}`: `conflict_clause={conflict_clause}`')


        # If the conflict is at level 0, then the problem is UNSAT
        if conflict_level == 0:
            return -1, None

        # Finding the conflict clause
        while True:
            # is_nice tells whether the conflict clause has only one literal set at the conflict level 
            # and prev_assigned_node is the latest assigned literal on the conflict level present in the conflict clause
            is_nice, prev_assigned_node = self._is_valid_clause(conflict_clause, conflict_level)
            # If the clause is nice, it is the final conflict clause, then break
            if is_nice:
                break
            # If the conflict clause is not the final clause, then replace it with its binary resolution
            # with the clause corresponding to the latest assigned literal
            clause = self._clauses[prev_assigned_node.clause]
            var = prev_assigned_node.var
            conflict_clause = self._binary_resolute(conflict_clause, clause, var)

            if DEBUG:
                print(f'    - prev_assigned_node: `var={var}`, `clause={clause}`')
                print(f'    - binary resolution => `conflict_clause={conflict_clause}`')


        # If the length of the learned conflict clause is more than 1
        if len(conflict_clause) > 1:
            # Add the number of learned clauses in the stats object
            self.stats._num_learned_clauses += 1
            # Get the clause_id for this clause
            clause_id = self._num_clauses
            # Increment the number of clauses and add the new clause to the clauses database
            self._num_clauses += 1
            self._clauses.append(conflict_clause)

            # Set the first 2 literals of the conflict_clause as its watchers.
            self._clauses_watched_by_l.setdefault(conflict_clause[0],[]).append(clause_id)
            self._clauses_watched_by_l.setdefault(conflict_clause[1],[]).append(clause_id)
            # Set the list containing the 2 watchers as the literals watching the clause
            self._literals_watching_c[clause_id] = [conflict_clause[0], conflict_clause[1]]

            # If VSIDS decider is used
            if self._decider == "VSIDS":
                # For all the literals appearing in the conflict clause, 
                # their scores are increased by _incr
                for l in conflict_clause:
                    self._lit_scores[l] += self._incr
                    self._priority_queue.increase_update(l, self._incr)
                # Increase _incr to give more weight to the recent conflict clausing literal
                self._incr += INCREASE
            # If MINISAT decider is used
            elif self._decider == "MINISAT":
                # For all variables corresponding to the literals appearing in the conflict clause, 
                # their scores are increased by _incr
                for l in conflict_clause:
                    var = self._get_var_from_literal(l)
                    self._var_scores[var] += self._incr
                    self._priority_queue.increase_update(var, self._incr)
                # To simulate the decay of all the previous var scores efficiently,
                # divide the _incr by decay instead of multiplying it to all the scores.
                self._incr /= self._decay

            # backtrack_level is the level to which the solver should jump back
            # conflict_level_literal is the single literal present in the conflict_clause at conflict_level
            backtrack_level, conflict_level_literal = self._get_backtrack_level(conflict_clause, conflict_level)
            # Get the variable related to the conflict_level_literal
            conflict_level_var = self._get_var_from_literal(conflict_level_literal)
            # Check if conflict_level_literal is negative
            is_negative_conflict_lit = self._is_negative_literal(conflict_level_literal)

            # After backtracking, the added clause will imply that the conflict_level_literal should be true 
            # If conflict_level_literal is negative, its variable should be set False, else it should be set True
            value_to_set = not is_negative_conflict_lit
            # Create an assignment node with conflict_level_var, value_to_set
            node = AssignedNode(conflict_level_var, value_to_set, backtrack_level, clause_id)

            if DEBUG:
                print(f'- Learned clause: `clause={conflict_clause}`')
                print(f'- Implication: `var={conflict_level_var}`, `value={value_to_set}`')
                print(f'- Backtrack: `backtrack_level={backtrack_level}`')
            return backtrack_level, node
        else:
            # If the clause has only one literal, then it is the one assigned at the conflict level
            literal = conflict_clause[0]
            var = self._get_var_from_literal(literal)
            is_negative_literal = self._is_negative_literal(literal)
            # Backtrack to level 0
            backtrack_level = 0
            # If conflict_level_literal is negative, its variable should be set False, else it should be set True
            value_to_set = not is_negative_literal
            # Create the node with var, value_to_set, backtrack_level(0)
            node = AssignedNode(var, value_to_set, backtrack_level, None)

            if DEBUG:
                print(f'- Learned clause: `clause={conflict_clause}`')
                print(f'- Implication: `var={var}`, `value={value_to_set}`')
                print(f'- Backtrack: `backtrack_level={backtrack_level}`')
            return backtrack_level, node

    def _backtrack(self, backtrack_level, node_to_add):
        "Backtrack the solver to the backtrack_level."



        if DEBUG:
            print(f'\n###### Backtrack `level={self._level}` to `level={backtrack_level}`\n')

        # Set level of the solver to the backtrack_level
        self._level = backtrack_level

        # Remove all nodes at level greater than backtrack_level from the assignment stack
        itr = len(self._assignment_stack) - 1
        while True:
            # If the stack is empty, then break
            if itr < 0:
                break
            # If a node with level less than equal to backtrack_level is reached, then break
            if self._assignment_stack[itr].level <= backtrack_level: 
                break
            else:
                # delete the node from the variable to node dictionary
                del self._variable_to_assignment_nodes[self._assignment_stack[itr].var]
                # delete the node from the assignment stack
                node = self._assignment_stack.pop(itr)

                # If VSIDS decider is used, then when we unset the variables, we push the two literals 
                # correspoding to the unset variable back into the priority queue with their scores
                if self._decider == "VSIDS":
                    self._priority_queue.add(node.var, self._lit_scores[node.var])
                    self._priority_queue.add(node.var+self._num_vars, self._lit_scores[node.var+self._num_vars])
                # If MINISAT decider is used, then when we unset the variables, we push the unset variable 
                # back into the priority queue with their scores
                elif self._decider == "MINISAT":
                    self._priority_queue.add(node.var, self._var_scores[node.var])

                # delete the node itself
                del node

                # move to the next node
                itr -= 1

        if DEBUG:
            print('- `assignment_stack={}`'.format([(x.var, x.value, x.level) for x in self._assignment_stack]))

        # node_to_add is None in case when backtrack is used to restart the solver
        # If node_to_add is not None
        if node_to_add:
            if DEBUG:
                print(f'- Add node: `var={node_to_add.var}`, `value={node_to_add.value}`')

            # Add the implied node to the variable to nodes dictionary
            self._variable_to_assignment_nodes[node_to_add.var] = node_to_add
            # Add the implied node to the assignment stack
            self._assignment_stack.append(node_to_add)
            node_to_add.index = len(self._assignment_stack) - 1

            # If VSIDS decider is used, then when we assign the variable,
            # we remove the two literals corresponing to the variable
            # as in the priority queue, we always keep the unassigned literals
            if self._decider == "VSIDS":
                self._priority_queue.remove(node_to_add.var)
                self._priority_queue.remove(node_to_add.var+self._num_vars)
            # If MINISAT decider is used
            elif self._decider == "MINISAT":
                self._priority_queue.remove(node_to_add.var)
                self._phase[node_to_add.var] = int(node_to_add.value)

            # Increment the number of implications 
            self.stats._num_implications += 1

    def _decide(self): 
        "Chooses an uassigned variable and assigns it with a value"

        # If ORDERED decider is used, get the smallest unassigned variable and set it to True
        if self._decider == "ORDERED":
            var = -1
            for x in range(1, self._num_vars+1):
                if x not in self._variable_to_assignment_nodes:
                    var = x
                    break
            value_to_set = True

        # If VSIDS decider is used, we get the literal with the highest score from the priority queue
        elif self._decider == "VSIDS":
            literal = self._priority_queue.get_top()
            # If it is -1, it means the queue is empty which means all variables are assigned
            if literal == -1:
                var = -1
            else:
                var = self._get_var_from_literal(literal)
                is_neg_literal = self._is_negative_literal(literal)
                value_to_set = not is_neg_literal
                # Remove the literal complementary to the above literal
                # as we have fixed the variable and so literal is no longer unassigned
                if is_neg_literal:
                    self._priority_queue.remove(var)
                else:
                    self._priority_queue.remove(var+self._num_vars)
        # If MINISAT decider is used, we get the variable with the highest score from the priority queue
        elif self._decider == "MINISAT":
            var = self._priority_queue.get_top()
            if var != -1:
                value_to_set = (self._phase[var] == 1)




        # If var is -1, it means all the variables are already assigned
        if var == -1:
            return -1

        # Increase the level by 1 as a decision is made
        self._level += 1

        if DEBUG:
            print(f'\n###### Decide `level={self._level}`\n')
            if var == -1:
                print(f'- `var={var}`')
            else:
                print(f'- `var={var}`, `value={value_to_set}`')

        # Create a new assignment node with var, value_to_set, level = _level
        new_node = AssignedNode(var, value_to_set, self._level, None)
        self._variable_to_assignment_nodes[var] = new_node
        self._assignment_stack.append(new_node)
        new_node.index = len(self._assignment_stack) - 1

        # Increase the number of decisions made in the stats object.
        self.stats._num_decisions += 1

        # return the var which is set
        return var

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
            # Indicating that BCP runs first time
            first_time = True
            
            # The main alogrithm loop
            while True:
                # Perform the BCP until there are no conflicts
                while True:
                    temp = time.time()
                    # Perform the BCP, result in {RESTART, CONFLICT, NO_CONFLICT}
                    result = self._boolean_constraint_propogation(first_time) 
                    # Increase the time spend in BCP
                    self.stats._bcp_time += time.time() - temp
                    # Break if no conflict
                    if result == "NO_CONFLICT":
                        break

                    # If "RESTART" is returned, it means we need to restart the solver
                    if result == "RESTART":
                        self._backtrack(0, None)
                        break

                    # Set first_time to False as we want it to be true only once initially
                    first_time = False

                    # If there is a conflict, call _analyze_conflict method to analyze it
                    temp = time.time()
                    backtrack_level, node_to_add = self._analyze_conflict()
                    # Increase the time spend in analyzing (stored in the stats object)
                    self.stats._analyze_time += time.time() - temp

                    # If backtrack level is -1, it means a conflict at level 0, so the problem is UNSAT.
                    if backtrack_level == -1:
                        # Store the result in the stats object
                        self.stats._result = "UNSAT"
                        # Store the time when the result is ready
                        self.stats._complete_time = time.time()
                        break

                    # Backtrack to the backtrack_level
                    temp = time.time()
                    self._backtrack(backtrack_level, node_to_add)
                    # Increase the time spend in backtracking (stored in the stats object)
                    self.stats._backtrack_time += time.time() - temp

                # Problem was proved to be UNSAT during BCP
                if self.stats._result == "UNSAT":
                    break

                # Set first_time to False as we want it to be true only once initially
                first_time = False

                # If all possible implications are made without conflicts,
                # then the solver decides on an unassigned variable
                temp = time.time()
                var_decided = self._decide()
                # Increase the time spend in deciding (stored in the stats object)
                self.stats._decide_time += time.time() - temp

                # If var_decided is -1, it means all the variables have been assigned 
                # without any conflict and so the input problem is satisfiable.
                if var_decided == -1:
                    # Store the result in the stats object
                    self.stats._result = "SAT"
                    # Store the time when the result is ready
                    self.stats._complete_time = time.time()
                    break



