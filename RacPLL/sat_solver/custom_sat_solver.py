'''
    clone from: https://github.com/AvivYaish/SMTsolver/blob/master/smt_solver/sat_solver/sat_solver.py
'''

from collections import deque, Counter
from pprint import pprint
import sortedcontainers
import settings
import random
import copy
import time

from sat_solver.sat_solver import Solver
from utils.timer import Timers



class CustomSATSolver(Solver):

    def __init__(self, formula=None, variables=None, layers_mapping=None, decider=None, max_new_clauses=float('inf'), 
        halving_period=10000, theory_solver=None):
    
        super().__init__()
        if formula is None:
            self._formula = set()

        self._max_new_clauses = max_new_clauses
        self._halving_period = halving_period
        self._theory_solver = theory_solver

        self._new_clauses = deque()
        self._assignment = dict()
        self._assignment_by_level = []
        self._satisfaction_by_level = []
        self._literal_to_clause = {}
        self._satisfied_clauses = set()
        self._last_assigned_literals = deque()
        self._variable_to_watched_clause = {} 

        # VSIDS related fields
        self._unassigned_vsids_count = Counter()
        self._assigned_vsids_count = {}
        self._step_counter = 0

        self._layers_mapping = copy.deepcopy(layers_mapping)
        self._reversed_layers_mapping = {i: k for k, v in layers_mapping.items() for i in v}

        self._all_vars = sortedcontainers.SortedList(variables)

        Timers.tic('_add_clause')
        for var in self._all_vars:
            self._formula.add(frozenset({var, -var}))

        for clause in self._formula:
            self._add_clause(clause)
        Timers.toc('_add_clause')

        self.start = True

        self.decider = decider

        self._early_stop = False
        self._early_return_status = None

        self._crown_decision_mapping = {}
        for lid, lnodes in self._layers_mapping.items():
            for jj, node in enumerate(lnodes):
                self._crown_decision_mapping[node] = (lid, jj)



    def set_early_stop(self, status):
        self._early_stop = True
        self._early_return_status = status

    def get_current_assigned_node(self):
        # print('_assignment_by_level', self._assignment_by_level)
        dl = len(self._assignment_by_level) - 1
        # if dl == 0:
            # return None, 0, None
        if len(self._assignment_by_level[dl]):
            variable = self._assignment_by_level[dl][-1] # index: -1
        else:
            variable = None
        return variable, dl, [self._crown_decision_mapping.get(variable, None)]

    def _add_clause(self, clause):
        """
        Initialize all clause data structures for the given clause.
        """
        for idx, literal in enumerate(clause):
            variable = abs(literal)
            if literal not in self._literal_to_clause:
                self._literal_to_clause[literal] = set()
            if variable not in self._variable_to_watched_clause:
                self._variable_to_watched_clause[variable] = set()
            if (literal not in self._unassigned_vsids_count) and (literal not in self._assigned_vsids_count):
                self._unassigned_vsids_count[literal] = 0
                self._unassigned_vsids_count[-literal] = 0

            self._literal_to_clause[literal].add(clause)
            if idx <= 1:
                self._variable_to_watched_clause[variable].add(clause)
            if literal in self._unassigned_vsids_count:
                self._unassigned_vsids_count[literal] += 1
            elif literal in self._assigned_vsids_count:
                self._assigned_vsids_count[literal] += 1

    def _assign(self, clause, literal: int, is_implied=False):
        """
        Assigns a satisfying value to the given literal.
        """
        variable = abs(literal)
        self._assignment[variable] = {
            "value": literal > 0,                           # Satisfy the literal
            "clause": clause,                               # The clause which caused the assignment
            "level": len(self._assignment_by_level) - 1,    # The decision level of the assignment
            "idx": len(self._assignment_by_level[-1]),      # Defines an assignment order in the same level
            "is_implied": is_implied
        }

        self._all_vars.discard(variable)
        self._layers_mapping[self._reversed_layers_mapping[variable]].discard(variable)

        # Keep data structures related to satisfied clauses up to date
        newly_satisfied_clauses = self._literal_to_clause[literal] - self._satisfied_clauses
        self._satisfaction_by_level[-1].extend(newly_satisfied_clauses)
        self._satisfied_clauses |= newly_satisfied_clauses

        # Keep data structures related to variable assignment up to date
        self._assignment_by_level[-1].append(variable)
        self._last_assigned_literals.append(literal)
        for cur_sign in [variable, -variable]:
            self._assigned_vsids_count[cur_sign] = self._unassigned_vsids_count[cur_sign]
            del self._unassigned_vsids_count[cur_sign]

        # print({k: v['value'] for k, v in self._assignment.items()})


    def _unassign(self, variable: int):
        """
        Unassigns the given variable.
        """
        del self._assignment[variable]
            
        for cur_sign in [variable, -variable]:
            self._unassigned_vsids_count[cur_sign] = self._assigned_vsids_count[cur_sign]
            del self._assigned_vsids_count[cur_sign]

        self._all_vars.add(variable)
        self._layers_mapping[self._reversed_layers_mapping[variable]].add(variable)

    def get_assignment(self) -> dict:
        return {var: val for var, val in self.iterable_assignment()}

    def get_variable_assignment(self, variable) -> bool:
        return self._assignment.get(variable, {"value": None})["value"]

    def iterable_assignment(self):
        """
        :return: a (variable: int, value: bool) tuple for every assigned variable.
        """
        for var in self._assignment:
            yield var, self._assignment[var]["value"], self._assignment[var]["is_implied"]

    def _find_last_literal(self, clause, removed_vars=[]):
        """
        :return: the last assigned literal in the clause, the second highest assignment level of literals in the clause,
        and the number of literals from the highest assignment level.
        """
        last_literal, prev_max_level, max_level, max_idx, max_level_count = None, -1, -1, -1, 0
        for literal in clause:
            variable = abs(literal)

            # what??
            if (variable in removed_vars) or (variable not in self._assignment):
                continue

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
        # print('- [Find]:', last_literal, prev_max_level, max_level, max_level_count)
        return last_literal, prev_max_level, max_level, max_level_count

    def _conflict_resolution(self, conflict_clause):
        """
        Learns conflict clauses using implication graphs, with the Unique Implication Point heuristic.
        """
        # print('--------------_conflict_resolution--------------')
        conflict_clause = set(conflict_clause)
        removed_vars = []
        while True:
            last_literal, prev_max_level, max_level, max_level_count = self._find_last_literal(conflict_clause, removed_vars)
            if last_literal is None:
                return None, None, -1
            clause_on_incoming_edge = self._assignment[abs(last_literal)]["clause"]
            # print(last_literal, prev_max_level, max_level, max_level_count, clause_on_incoming_edge)
            if (max_level_count == 1) or (clause_on_incoming_edge is None):
                if max_level_count != 1:
                    # If the last literal was assigned because of the theory, there is no incoming edge
                    # The literal to reassign should be the decision literal of the same level
                    # print('    - last_literal before:', last_literal)
                    last_literal = self._assignment_by_level[max_level][0]
                    # print(self._assignment_by_level[max_level], conflict_clause)
                    # print('    - last_literal after:', last_literal)
                    if self._assignment[last_literal]["value"]:
                        last_literal = -last_literal
                    conflict_clause.add(last_literal)
                # If the last assigned literal is the only one from the last decision level:
                # return the conflict clause, the next literal to assign (which should be the
                # watch literal of the conflict clause), and the decision level to jump to
                return frozenset(conflict_clause), last_literal, prev_max_level

            # Resolve the conflict clause with the clause on the incoming edge
            # Might be the case that the last literal was assigned because of the
            # theory, and in that case it is impossible to do resolution
            # print('conflict_clause:', conflict_clause, 'last_literal:', last_literal)
            conflict_clause |= clause_on_incoming_edge
            # print()
            conflict_clause.remove(last_literal)
            conflict_clause.remove(-last_literal)
            removed_vars.append(abs(last_literal))

    def _bcp(self):
        """
        Performs BCP, as triggered by the last assigned literals. If new literals are assigned as part of the BCP,
        the BCP continues using them. The BCP uses watch literals.
        :return: None, if there is no conflict. If there is one, the conflict clause is returned.
        """
        # print('[bcp]', self._last_assigned_literals)
        
        # print('\t[variable to watched clause]')
        # for k, v in self._variable_to_watched_clause.items():
        #     print(k, '--->', v)
        # print()

        # print('\t[_satisfied_clauses]')
        # for k, v in enumerate(self._satisfied_clauses):
        #     print(k, '--->', v)
        # print()

        while self._last_assigned_literals:
            watch_literal = self._last_assigned_literals.popleft()
            # print('watch_literal:', watch_literal)
            for clause in self._variable_to_watched_clause[abs(watch_literal)].copy():
                if clause not in self._satisfied_clauses:
                    conflict_clause = self._replace_watch_literal(clause, watch_literal)
                    if conflict_clause is not None:
                        return conflict_clause
        return None  # No conflict-clause

    def _replace_watch_literal(self, clause, watch_literal: int):
        """
        - If the clause is satisfied, nothing to do.
        - Else, it is not satisfied yet:
          - If it has 0 unassigned literals, it is UNSAT.
          - If it has 1 unassigned literals, assign the correct value to the last literal.
          - If it has > 2 unassigned literals, pick one to become the new watch literal.
        """
        watch_variable, replaced_watcher, unassigned_literals = abs(watch_literal), False, []
        for unassigned_literal in clause:
            unassigned_variable = abs(unassigned_literal)
            if unassigned_variable in self._assignment:
                # If the current literal is assigned, it cannot replace the current watch literal
                continue
            unassigned_literals.append(unassigned_literal)

            if replaced_watcher:
                # If we already replaced the watch_literal
                if len(unassigned_literals) > 1:
                    break
            elif clause not in self._variable_to_watched_clause[unassigned_variable]:
                # If the current literal is not already watching the clause, it can replace the watch literal
                self._variable_to_watched_clause[watch_variable].remove(clause)
                self._variable_to_watched_clause[unassigned_variable].add(clause)
                replaced_watcher = True

        if len(unassigned_literals) == 0:
            # Clause is UNSAT, return it as the conflict-clause
            return clause
        if len(unassigned_literals) == 1:
            # The clause is still not satisfied, and has only one unassigned literal.
            # Assign the correct value to it. Because it is now watching the clause,
            # and was also added to self._last_assigned_literals, we will later on
            # check if the assignment causes a conflict
            literal = unassigned_literals.pop()
            self._assign(clause, literal, is_implied=True)
            # print(f'\t- [c] {clause}, {literal}')
        return None

    def _add_conflict_clause(self, conflict_clause):
        """
        Adds a conflict clause to the formula.
        """
        if self._max_new_clauses <= 0:
            return

        # Remove previous conflict clauses, if there are too many
        if len(self._new_clauses) == self._max_new_clauses:
            clause_to_remove = self._new_clauses.popleft()
            for literal in clause_to_remove:
                self._literal_to_clause[literal].discard(clause_to_remove)
                self._variable_to_watched_clause[abs(literal)].discard(clause_to_remove)

        self._new_clauses.append(conflict_clause)
        # print(f'\t- [add] {conflict_clause}')
        self._add_clause(conflict_clause)


    def _constraint_propagation_to_exhaustion(self, propagation_func):
        """
        Performs constraint propagation using the given function
        until exhaustion, returns False iff formula is UNSAT.
        """
        # print('- [BCP + TCP]', propagation_func)
        conflict_clause = propagation_func()
        while conflict_clause is not None:
            conflict_clause, watch_literal, level_to_jump_to = self._conflict_resolution(conflict_clause)
            # print('watch_literal:', watch_literal, '\nlen:', len(list(conflict_clause)), '\nconflict_clause:', conflict_clause)
            # print(len(list(conflict_clause)), conflict_clause)
            # print()
            if level_to_jump_to == -1:
                # An assignment that satisfies the formula's unit clauses causes a conflict, so the formula is UNSAT
                return False
            self.backtrack(level_to_jump_to)
            self._add_conflict_clause(conflict_clause)
            self._assign(conflict_clause, watch_literal)
            # print(f'- [Conflict] {abs(watch_literal)}={watch_literal}, dl={level_to_jump_to}')
            # print(f'\t- [cc] {conflict_clause}, {watch_literal}')

            conflict_clause = propagation_func()
        return True

    def _tcp(self):
        """
        Theory constraint propagation.
        """
        conflict_clause, new_assignments = self._theory_solver.propagate()
        if conflict_clause is not None:
            return conflict_clause
        for literal in new_assignments:
            self._assign(None, literal, is_implied=True)
            # print(f'- [Imply] {abs(literal)}={literal}')
        return None

    def propagate(self) -> bool:
        if self.start:
            self.start = False
            if not self._constraint_propagation_to_exhaustion(self._tcp):
                return False

        while self._last_assigned_literals:
            if (not self._constraint_propagation_to_exhaustion(self._bcp)) or \
                ((self._theory_solver is not None) and (not self._constraint_propagation_to_exhaustion(self._tcp))):
                return False
        return True

    def backtrack(self, level: int):
        """
        Non-chronological backtracking.
        """
        self._last_assigned_literals = deque()
        while len(self._assignment_by_level) > level + 1:
            for variable in self._assignment_by_level.pop():
                self._unassign(variable)
            for clause in self._satisfaction_by_level.pop():
                self._satisfied_clauses.remove(clause)
        if self._theory_solver:
            self._theory_solver.backtrack(level)

    def _increment_step(self):
        """
        Maintain data structures related to VSIDS
        """
        self._step_counter += 1
        if self._step_counter >= self._halving_period:
            self._step_counter = 0
            for literal in self._unassigned_vsids_count:
                self._unassigned_vsids_count[literal] /= 2
            for literal in self._assigned_vsids_count:
                self._assigned_vsids_count[literal] /= 2

    # def _decide(self):
    #     """
    #     Decides which literal to assign next, using the VSIDS decision heuristic.
    #     """
    #     # print('--------------_decide--------------')

    #     variable = self._all_vars[0]
    #     orig_variable = variable

    #     if settings.RANDOM_DECISION:
    #         variable = random.choice(self._layers_mapping[self._reversed_layers_mapping[orig_variable]])


    #     count = 0
    #     while variable in self._layers_mapping[self._reversed_layers_mapping[orig_variable]]:
    #         self.create_new_decision_level()
    #         if settings.RANDOM_DECISION:
    #             sign = random.choice([-1, 1])
    #             self._assign(None, sign * variable)
    #             print(f'toi chon: variable={variable}, value={sign==1}')
    #             print()
    #         else:
    #             count_pos = self._unassigned_vsids_count.get(variable, 0)
    #             count_neg = self._unassigned_vsids_count.get(-variable, 0)
    #             if count_pos >= count_neg:
    #                 self._unassigned_vsids_count.pop(variable)
    #                 self._assign(None, variable)
    #                 # print(f'- [Decide] `{variable}`=True\n')
    #             else:
    #                 self._unassigned_vsids_count.pop(-variable)
    #                 self._assign(None, -variable)
    #                 # print(f'- [Decide] `{variable}`=False\n')

    #         count += 1
    #         if settings.DEBUG:
    #             print(f'- [{count}] Choose: variable=`{variable}`\n')
    #         if count == settings.N_DECISIONS:
    #             break
    #         if not self._all_vars:
    #             break
                
    #         if settings.RANDOM_DECISION:
    #             variable = random.choice(self._layers_mapping[self._reversed_layers_mapping[orig_variable]])
    #         else:
    #             variable = self._all_vars[0]

    #     # print()
    #     # print()
    #     # print()

    #     # literal, count = self._unassigned_vsids_count.most_common(1).pop()

    def _decide(self):
        Timers.tic('Heuristic Decision Get')

        unassigned_variables = self._layers_mapping[self._reversed_layers_mapping[self._all_vars[0]]]
        variable, value = self.decider.get(unassigned_variables)
        self.create_new_decision_level()
        self._assign(None, variable if value else -variable)
        if settings.DEBUG:
            print(f'- Choose: variable=`{variable}`, value={value}\n')

        Timers.toc('Heuristic Decision Get')


    def create_new_decision_level(self):
        self._assignment_by_level.append(list())
        self._satisfaction_by_level.append(list())
        if self._theory_solver:
            self._theory_solver.create_new_decision_level()

    def _satisfy_unit_clauses(self):
        self.create_new_decision_level()
        for clause in self._formula:
            if len(clause) == 1:
                for literal in clause:
                    if abs(literal) not in self._assignment:
                        self._assign(clause, literal, is_implied=True)

    def _is_sat(self) -> bool:
        return self._formula.issubset(self._satisfied_clauses)

    def solve(self, timeout=None) -> bool:
        self._satisfy_unit_clauses()
        start = time.perf_counter()
        while True:
            if timeout is not None:
                if time.perf_counter() - start >= timeout:
                    return 'TIMEOUT'
            self._increment_step()
            if not self.propagate():
                return 'UNSAT'
            if self._is_sat():
                return 'SAT'
            if self._early_stop:
                return self._early_return_status
            self._decide()
