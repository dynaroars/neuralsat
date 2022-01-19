from collections import deque, Counter
from solver.solver import Solver
from operator import or_
from functools import reduce

from pprint import pprint

class SATSolver2(Solver):

    def __init__(self, formula=None, meta=None, first_var=None, max_new_clauses=float('inf'), 
        halving_period=10000, theory_solver=None):
    
        super().__init__()
        if formula is None:
            formula = frozenset()

        self._first_var = first_var
        self._formula = formula
        self._max_new_clauses = max_new_clauses
        self._theory_solver = theory_solver
        # The time period after which all VSIDS counters are halved
        self._halving_period = halving_period  


        self._new_clauses = deque()
        self._assignment = dict()
        self._assignment_by_level = []
        self._satisfaction_by_level = []
        self._literal_to_clause = {}
        self._satisfied_clauses = set()

        # queue of the literals assigned in the last level
        self._last_assigned_literals = deque()      

        # A literal -> set(clause) dictionary.
        self._literal_to_watched_clause = {}        

        # VSIDS related fields
        self._unassigned_vsids_count = Counter()
        self._assigned_vsids_count = {}
        self._step_counter = 0    

        for clause in self._formula:
            self._add_clause(clause)

        self._all_vars = set({})
        self._mapping_vars = {}

        for key, value in meta.items():
            if key.startswith('a'):
                self._all_vars.add(value[0])
                self._mapping_vars[key] = value[0]

        self._reversed_mapping_vars = {v: k for k, v in self._mapping_vars.items()}

        # print(self._all_vars)
        # print(self._mapping_vars)
        # print(self._reversed_mapping_vars)

        # print('[+] self._literal_to_clause')
        # pprint(self._literal_to_clause)
        
        # print('[+] self._literal_to_watched_clause')
        # pprint(self._literal_to_watched_clause)


    def _add_clause(self, clause):
        """
        Initialize all clause data structures for the given clause.
        """
        # print('================> _add_clause', clause)
        for idx, literal in enumerate(clause):
            variable = abs(literal)
            if literal not in self._literal_to_clause:
                self._literal_to_clause[literal] = set()

            if variable not in self._literal_to_watched_clause:
                self._literal_to_watched_clause[variable] = set()

            if (literal not in self._unassigned_vsids_count) and \
                (literal not in self._assigned_vsids_count):
                self._unassigned_vsids_count[literal] = 0
                self._unassigned_vsids_count[-literal] = 0

            self._literal_to_clause[literal].add(clause)
            if idx <= 1:
                self._literal_to_watched_clause[variable].add(clause)

            if literal in self._unassigned_vsids_count:
                self._unassigned_vsids_count[literal] += 1
            elif literal in self._assigned_vsids_count:
                self._assigned_vsids_count[literal] += 1


    def get_assignment(self) -> dict:
        return {var: val for var, val in self.iterable_assignment()}


    def get_variable_assignment(self, variable) -> bool:
        # return self._assignment[variable]
        return self._assignment.get(variable, {"value": None})["value"]


    def iterable_assignment(self):
        """
        :return: a (variable: int, value: bool) tuple for every assigned variable.
        """
        for var in self._assignment:
            yield var, self._assignment[var]["value"]



    def _assign(self, clause, literal: int):
        """
        Assigns a satisfying value to the given literal.
        """
        # print('================> _assign')

        print('##### Assign\n')

        variable = abs(literal)
        # print(f'[+] variable={variable}')
        self._assignment[variable] = {
            "value": literal > 0,                           # Satisfy the literal
            "clause": clause,                               # The clause which caused the assignment
            "level": len(self._assignment_by_level) - 1,    # The decision level of the assignment
            "idx": len(self._assignment_by_level[-1])       # Defines an assignment order in the same level
        }
        # pprint(self._assignment[variable])


        self._all_vars.remove(variable)

        print(f'- Assign `variable={variable}`, `value={literal>0}`')
        print(f'- Unassigned variables = `{self._all_vars}`\n')


        # Update satisfied clauses
        newly_satisfied_clauses = set({})
        if literal in self._literal_to_clause:
            newly_satisfied_clauses = self._literal_to_clause[literal] - self._satisfied_clauses

        self._satisfaction_by_level[-1].extend(newly_satisfied_clauses)
        self._satisfied_clauses |= newly_satisfied_clauses

        # Update variable assignment
        self._assignment_by_level[-1].append(variable)
        self._last_assigned_literals.append(literal)
        for cur_sign in [variable, -variable]:
            self._assigned_vsids_count[cur_sign] = self._unassigned_vsids_count[cur_sign]
            del self._unassigned_vsids_count[cur_sign]



    def create_new_decision_level(self):
        # print('================> create_new_decision_level')
        self._assignment_by_level.append(list())
        self._satisfaction_by_level.append(list())
        if self._theory_solver:
            self._theory_solver.create_new_decision_level()


    def propagate(self) -> bool:
        print('##### Propagate\n')
        # print('[+] self._last_assigned_literals:', self._last_assigned_literals)

        if (not self._last_assigned_literals) and (self._theory_solver is not None) \
            and (not self._constraint_propagation_to_exhaustion(self._tcp)):
            return False
        # print('[+] self._last_assigned_literals:', self._last_assigned_literals)

        while self._last_assigned_literals:
            if (not self._constraint_propagation_to_exhaustion(self._bcp)) or \
                ((self._theory_solver is not None) and (not self._constraint_propagation_to_exhaustion(self._tcp))):
                return False
        return True


    def _constraint_propagation_to_exhaustion(self, propagation_func):
        """
        Performs constraint propagation using given function until exhaustion, 
        returns False iff formula is UNSAT.
        """
        # print('================> _constraint_propagation_to_exhaustion')
        conflict_clause = propagation_func()
        while conflict_clause is not None:
            conflict_clause, watch_literal, level_to_jump_to = self._conflict_resolution(conflict_clause)
            if level_to_jump_to == -1:
                return False
            self.backtrack(level_to_jump_to)
            self._add_conflict_clause(conflict_clause)
            self._assign(conflict_clause, watch_literal)
            conflict_clause = propagation_func()
        return True


    def backtrack(self, level: int):
        """
        Non-chronological backtracking.
        """
        # print('================> backtrack')
        print('##### Backtrack\n')

        print(f'- Backtrack to `level={level}`')

        self._last_assigned_literals = deque()
        while len(self._assignment_by_level) > level + 1:
            for variable in self._assignment_by_level.pop():
                self._unassign(variable)
            for clause in self._satisfaction_by_level.pop():
                self._satisfied_clauses.remove(clause)
        if self._theory_solver:
            self._theory_solver.backtrack(level)


    def _unassign(self, variable: int):
        """
        Unassigns the given variable.
        """
        print(f'- Unassign `variable={variable}`')

        del self._assignment[variable]
        self._all_vars.add(variable)
        for cur_sign in [variable, -variable]:
            self._unassigned_vsids_count[cur_sign] = self._assigned_vsids_count[cur_sign]
            del self._assigned_vsids_count[cur_sign]

    def _add_conflict_clause(self, conflict_clause):
        """
        Adds a conflict clause to the formula.
        """
        # print('================> _add_conflict_clause')
        print(f'- Add conflict clause: `{list(conflict_clause)}`\n')

        if self._max_new_clauses <= 0:
            return

        # Remove previous conflict clauses, if there are too many
        if len(self._new_clauses) == self._max_new_clauses:
            clause_to_remove = self._new_clauses.popleft()
            for literal in clause_to_remove:
                self._literal_to_clause[literal].discard(clause_to_remove)
                self._literal_to_watched_clause[abs(literal)].discard(clause_to_remove)

        self._new_clauses.append(conflict_clause)
        self._add_clause(conflict_clause)


    def _conflict_resolution(self, conflict_clause):
        """
        Learns conflict clauses using implication graphs, with the Unique Implication Point heuristic.
        """
        # print('================> _conflict_resolution', conflict_clause)
        print('##### Analyze\n')
        conflict_clause = set(conflict_clause)
        while True:
            last_literal, prev_max_level, max_level, max_level_count = self._find_last_literal(conflict_clause)
            clause_on_incoming_edge = self._assignment[abs(last_literal)]["clause"]
            if (max_level_count == 1) or (clause_on_incoming_edge is None):
                if max_level_count != 1:
                    # If the last literal was assigned because of the theory, there is no incoming edge
                    # The literal to reassign should be the decision literal of the same level
                    last_literal = self._assignment_by_level[max_level][0]
                    if self._assignment[last_literal]["value"]:
                        last_literal = -last_literal
                    conflict_clause.add(last_literal)
                # If the last assigned literal is the only one from the last decision level:
                # return the conflict clause, the next literal to assign (which should be the
                # watch literal of the conflict clause), and the decision level to jump to
                print(f'- Backtrack `level={prev_max_level}`\n')

                return frozenset(conflict_clause), last_literal, prev_max_level

            # Resolve the conflict clause with the clause on the incoming edge
            # Might be the case that the last literal was assigned because of the
            # theory, and in that case it is impossible to do resolution
            conflict_clause |= clause_on_incoming_edge
            conflict_clause.remove(last_literal)
            conflict_clause.remove(-last_literal)



    def _find_last_literal(self, clause):
        """
        :return: the last assigned literal in the clause, the second highest assignment level of literals in the clause,
        and the number of literals from the highest assignment level.
        """
        # print('================> _find_last_literal', clause)
        last_literal, prev_max_level, max_level, max_idx, max_level_count = None, -1, -1, -1, 0
        for literal in clause:
            variable = abs(literal)
            level, idx = self._assignment[variable]["level"], self._assignment[variable]["idx"]
            if level > max_level:
                prev_max_level = max_level
                last_literal, max_level, max_idx, max_level_count = literal, level, idx, 1
            elif level == max_level:
                max_level_count += 1
                if idx > max_idx:
                    last_literal, max_idx = literal, idx
            elif level > prev_max_level:
                prev_max_level = level

        if (prev_max_level == -1) and (max_level != -1):
            prev_max_level = max_level - 1
        return last_literal, prev_max_level, max_level, max_level_count


    def _bcp(self):
        """
        Performs BCP, as triggered by the last assigned literals. 
        If new literals are assigned as part of the BCP, BCP continues using them. 
        The BCP uses watch literals.
        :return: None, if there is no conflict. Otherwise, the conflict clause is returned.
        """
        # print('================> _bcp')
        while self._last_assigned_literals:
            watch_literal = self._last_assigned_literals.popleft()
            # print(f'[+] watch_literal={watch_literal}')
            if self._literal_to_watched_clause:
                for clause in self._literal_to_watched_clause[abs(watch_literal)].copy():
                    # print(f'[+] clause={clause}', clause in self._satisfied_clauses)
                    if clause not in self._satisfied_clauses:
                        conflict_clause = self._replace_watch_literal(clause, watch_literal)
                        if conflict_clause is not None:
                            return conflict_clause
        return None  # No conflict-clause


    def _tcp(self):
        """
        Theory constraint propagation.
        :return: the conflict-clause, or None if there is no conflict.
        """
        # print('================> _tcp')
        conflict_clause, new_assignments = self._theory_solver.propagate()
        if conflict_clause is not None:
            return conflict_clause
        for literal in new_assignments:
            self._assign(None, literal)
            # self._all_vars.remove(abs(literal))
        return None


    def _replace_watch_literal(self, clause, watch_literal: int):
        """
        - If the clause is satisfied, nothing to do.
        - Else, it is not satisfied yet:
          - If it has 0 unassigned literals, it is UNSAT.
          - If it has 1 unassigned literals, assign the correct value to the last literal.
          - If it has > 2 unassigned literals, pick one to become the new watch literal.
        """
        # print('================> _replace_watch_literal')
        watch_variable, replaced_watcher, unassigned_literals = abs(watch_literal), False, []
        # print(f'[+] watch_literal={watch_literal}')
        # print(f'[+] clause={clause}')
        # print(f'[+] self._assignment={self._assignment.keys()}')

        # print(f'[+] self._literal_to_watched_clause')
        # pprint(self._literal_to_watched_clause)

        for unassigned_literal in clause:
            unassigned_variable = abs(unassigned_literal)
            if unassigned_variable in self._assignment:
                continue
            unassigned_literals.append(unassigned_literal)

            # print(f'[+] unassigned_variable={unassigned_variable}')
            if replaced_watcher:
                # If we already replaced the watch_literal
                if len(unassigned_literals) > 1:
                    break
            elif clause not in self._literal_to_watched_clause[unassigned_variable]:
                # If the current literal is not already watching the clause, it can replace the watch literal
                self._literal_to_watched_clause[watch_variable].remove(clause)
                self._literal_to_watched_clause[unassigned_variable].add(clause)
                replaced_watcher = True

        # print(f'[+] self._literal_to_watched_clause')
        # pprint(self._literal_to_watched_clause)
        if len(unassigned_literals) == 0:
            # Clause is UNSAT, return it as the conflict-clause
            return clause

        if len(unassigned_literals) == 1:
            # The clause is still not satisfied, and has only one unassigned literal.
            # Assign the correct value to it. Because it is now watching the clause,
            # and was also added to self._last_assigned_literals, we will later on
            # check if the assignment causes a conflict
            self._assign(clause, unassigned_literals.pop())
        return None


    def _decide(self):
        """
        Decides which literal to assign next, using the VSIDS decision heuristic.
        """
        # print('================> _decide')
        # print(self._first_var, self._theory_solver, (self._first_var not in self._assignment))

        print('##### Decide\n')

        self.create_new_decision_level()
        if self._formula:
            if self._first_var and self._theory_solver and (self._first_var not in self._assignment):
                self._assign(None, self._first_var)
                print(self._first_var, self._assignment)
                del self._unassigned_vsids_count[self._first_var]
            else:
                literal, count = self._unassigned_vsids_count.most_common(1).pop()
                print(f'[+] literal={literal}, count={count}')
                self._assign(None, literal)
        else:
            for literal in self._all_vars:
                break
            print(f'- Choose: variable=`{literal}`\n')

            self._assign(None, literal)

    def _increment_step(self):
        """
        Maintain data structures related to VSIDS
        """
        # print('================> _increment_step')
        # print('[+] self._step_counter', self._step_counter)

        # print('[+] self._unassigned_vsids_count')
        # pprint(self._unassigned_vsids_count)

        # print('[+] self._assigned_vsids_count')
        # pprint(self._assigned_vsids_count)

        self._step_counter += 1
        if self._step_counter >= self._halving_period:
            self._step_counter = 0
            for literal in self._unassigned_vsids_count:
                self._unassigned_vsids_count[literal] /= 2
            for literal in self._assigned_vsids_count:
                self._assigned_vsids_count[literal] /= 2



    def _is_sat(self) -> bool:
        # print('================> _is_sat')
        # print('[+] self._formula:', self._formula)
        # print('[+] self._satisfied_clauses:', self._satisfied_clauses)
        # print('_assignment', [self._assignment.keys()])
        print('##### Check SAT\n')

        if not self._all_vars:
            print(f'- Unassigned variables = `{self._all_vars}` => `SAT`\n')
        else:
            print(f'- Unassigned variables = `{self._all_vars}` => `None`\n')

        if self._formula:
            return self._formula.issubset(self._satisfied_clauses)
        return not self._all_vars


    def _satisfy_unit_clauses(self):
        self.create_new_decision_level()
        for clause in self._formula:
            if len(clause) == 1:
                for literal in clause:
                    if abs(literal) not in self._assignment:
                        self._assign(clause, literal)


    def solve(self) -> bool:
        # print('================> solve')
        self._satisfy_unit_clauses()
        while True:
            # print('\n\nLoop')
            # print('[+] self._satisfied_clauses =', self._satisfied_clauses)
            self._increment_step()
            if not self.propagate():
                return False
            if self._is_sat():
                return True
            self._decide()



















