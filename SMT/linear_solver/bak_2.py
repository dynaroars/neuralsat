from simplex.clause import Clause, Formula
from simplex.simplex import Simplex
from simplex.parser import Parser
from pprint import pprint
import numpy as np
import copy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s |%(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        # logging.StreamHandler()
    ]
)

class ClauseConflictException(Exception):
    pass

class RealSolver:

    def __init__(self, formula_str):
        self.parsed_input = Parser.parse(formula_str)

    def simplex_feasible(self, assignment):
        equation_variables = {}
        rows = []
        for var in assignment:
            if 'q' in var:
                # print('    --->', var)
                if assignment[var]:
                    rows.append(self.parsed_input.row_dict[var])
                else:
                    rows.append(self.parsed_input.row_dict['~'+var])
        # print('    --->', self.parsed_input.row_dict, rows)

        self.mySimplex = Simplex(self.parsed_input, rows)
        if self.mySimplex.solve():
            return self.mySimplex
        return False


    def solve(self):
        logging.info('')

        unpicked_variables = self.parsed_input.cnf.get_set_variables()
        # print('[+] all_variables:', unpicked_variables)
        # # print(unpicked_variables)
        #set initial clause indexes
        counter = 0
        top_level_assignmen_tree = []
        if 'xi' in unpicked_variables:
            top_level_assignmen_tree = [{'xi': [{'xi': True}]}]
            unpicked_variables.remove('xi')

        for idx, clause in enumerate(self.parsed_input.cnf.clauses):
            clause.index = idx
            # # print(str(clause), clause.index)

        cnf_formula = copy.deepcopy(self.parsed_input.cnf)

        # print('[+] Formula:', cnf_formula)
        # print('[+] top_level_assignmen_tree:', top_level_assignmen_tree)
        # print('[+] unpicked_variables:', unpicked_variables)

        ret = self.loop(cnf_formula, top_level_assignmen_tree, unpicked_variables)

        while isinstance(ret, list):
            new_clause = ret[0]
            new_clause.index = len(cnf_formula.clauses)
            cnf_formula.clauses.append(new_clause)
            unpicked_variables = cnf_formula.get_set_variables()
            ret = self.loop(cnf_formula, top_level_assignmen_tree, unpicked_variables)
        
        real_sol = None
        if ret != False and self.mySimplex != None:
            real_sol = self.mySimplex.get_assignment()
        return ret, real_sol



    def loop(self, cnf_formula, assignment_tree, unpicked_variables):
        # print()
        logging.info('')

        assignments = {}
        nodes = {}
        counter = 0
        for dictionary in assignment_tree:
            for key, value in dictionary.items():
                # # print(key, value)
                nodes[key] = counter
                counter += 1
                for assign in value:
                    # # print(assign)
                    assignments.update(assign)

        # print('[+] nodes:', nodes)
        # print('[+] assignments:', assignments)

        current_level_cnf_formula = copy.deepcopy(cnf_formula)
        current_level_cnf_formula = self.simplify(current_level_cnf_formula, assignments)
        
        # print('[+] current_level_cnf_formula:', current_level_cnf_formula)

        if current_level_cnf_formula.assign(assignments):
            # print('[+] current_level_cnf_formula.assign(assignments):', True)
            return assignments
        elif current_level_cnf_formula.has_empty_clause():
            # print('[+] current_level_cnf_formula.has_empty_clause():', True)
            return False
        else:
            # print('[+] current_level_cnf_formula.assign(assignments):', False)
            # print('[+] current_level_cnf_formula.has_empty_clause():', False)

            choice_var_raw = current_level_cnf_formula.highest_frequent_variable()


            choice_var = choice_var_raw.replace("~", "")
            unpicked_variables.remove(choice_var)
            # print('[+] choice_var_raw:', choice_var_raw)
            # print('[+] unpicked_variables:', unpicked_variables)
            
            for var_assignment in [False, True]:
                old_assignment_tree = copy.deepcopy(assignment_tree)
                if choice_var == choice_var_raw:
                    choice_of_assignment = {choice_var : var_assignment}
                else:
                    choice_of_assignment = {choice_var : not var_assignment}

                # print('[+] choice_of_assignment:', choice_of_assignment)


                assignment_list = [choice_of_assignment]
                assignments.update(choice_of_assignment)
                # print('[+++] assignments for simplex:', assignments)
                # print('[+] assignment_list:', assignment_list)

                if self.simplex_feasible(assignments) == False:
                    # print('[!!!] simplex is not feasible')
                    continue

                # print('[+] simplex is feasible')

                test_cnf_formula = copy.deepcopy(current_level_cnf_formula)
                simplified_cnf_formula = self.simplify(test_cnf_formula, assignments)
                # print('[+] simplified_cnf_formula:', simplified_cnf_formula)
                if simplified_cnf_formula.has_empty_clause():
                    # print('[!] simplified_cnf_formula has_empty_clause\n')
                    continue

                merged_assignments = copy.deepcopy(assignments)
                remaining_variables = simplified_cnf_formula.get_set_variables()
                # print('[+] remaining_variables:', remaining_variables)
                unit_assignments = self.get_unit_clauses_assignments(simplified_cnf_formula)
                # print('[+] unit_assignments:', unit_assignments)
                # print()

                while unit_assignments:
                    if "Error" in unit_assignments:
                        alpha, beta, conflict_variable = unit_assignments["Error"]
                        clause_alpha = None
                        clause_beta = None
                        for clause in cnf_formula.clauses:
                            if clause.index == alpha:
                                clause_alpha = clause
                            if clause.index == beta:
                                clause_beta = clause
                                break
                        new_clause_variables = []
                        for variable in clause_alpha.variables:
                            var = variable.replace('~', '')
                            if var != conflict_variable:
                                new_clause_variables.append(variable)

                        for variable in clause_beta.variables:
                            var = variable.replace('~', '')
                            if var != conflict_variable and variable not in new_clause_variables:
                                new_clause_variables.append(variable)
                        new_clause = Clause(new_clause_variables)

                        backtrack_level = None
                        for var_raw in new_clause.variables:
                            var_raw = var_raw.replace('~', '')
                            if var_raw in nodes:
                                if backtrack_level is None:
                                    backtrack_level = nodes[var_raw]
                                elif nodes[var_raw] < backtrack_level:
                                    backtrack_level = nodes[var_raw]

                        if backtrack_level is None:
                            return [new_clause, None]
                        else:
                            return_target = "origin"
                            backtrack_level -= 1
                            if backtrack_level >= 0:
                                node_dict = assignment_tree[backtrack_level]
                                for key, value in node_dict.items():
                                    return_target = key
                            return [new_clause, return_target]

                    else:
                        try:
                            merged_assignments = self.merge_assignments(unit_assignments, merged_assignments)
                        except ClauseConflictException:
                            break

                        # print('[+++] merged_assignments for simplex:', merged_assignments)
                        if self.simplex_feasible(merged_assignments) == False:
                            # print('[!!!] simplex is not feasible')
                            break

                        simplified_cnf_formula = self.simplify(simplified_cnf_formula, merged_assignments)
                        if simplified_cnf_formula.has_empty_clause():
                            break
                        remaining_variables = simplified_cnf_formula.get_set_variables()

                        for key, value in unit_assignments.items():
                            assignment_list.append({key : value})
                        unit_assignments = self.get_unit_clauses_assignments(simplified_cnf_formula)

                # print('\n[+++] merged_assignments for simplex (outside while):', merged_assignments)
                if self.simplex_feasible(merged_assignments) == False:
                    # print('[!!!] simplex is not feasible')
                    continue

                if simplified_cnf_formula.has_empty_clause():
                    continue

                old_assignment_tree.append({choice_var: assignment_list})
                ret = self.loop(cnf_formula, old_assignment_tree, remaining_variables)

                while isinstance(ret, list):
                    if ret[1] is None:
                        new_clause = ret[0]
                        new_clause.index = len(cnf_formula.clauses)
                        cnf_formula.clauses.append(new_clause)
                        unpicked_variables = current_level_cnf_formula.get_set_variables()
                        ret = self.loop(cnf_formula, old_assignment_tree, unpicked_variables)
                    else:
                        if choice_var == ret[1]:
                            new_clause = ret[0]
                            new_clause.index = len(cnf_formula.clauses)
                            cnf_formula.clauses.append(new_clause)
                            #recursive call itself
                            unpicked_variables = current_level_cnf_formula.get_set_variables()
                            old_assignment_tree = copy.deepcopy(assignment_tree)
                            ret = self.loop(cnf_formula, old_assignment_tree, unpicked_variables)
                        else:
                            return ret

                if ret:
                    return ret
            return False


    def has_assignment_conflict(self, assignment_1, assignment_2):
        intersection = assignment_1.keys() & assignment_2.keys()
        for key in intersection:
            if assignment_1[key] != assignment_2[key]:
                return True
        return False

    def merge_assignments(self, assignment_1, assignment_2):
        if self.has_assignment_conflict(assignment_1, assignment_2):
            raise ClauseConflictException()
        else:
            a = copy.deepcopy(assignment_1)
            a.update(assignment_2)
            return a


    def get_unit_clauses_assignments(self, cnf_unit_clauses):
        new_assignments = {}
        clause_track = {}
        for unit_c in cnf_unit_clauses.clauses:
            if len(unit_c.variables) == 1:
                var = unit_c.variables[0]
                assignment = True
                if self.is_negated(var):
                    assignment = False
                var_identifier = var.replace('~', '')
                if var_identifier in new_assignments:
                    if new_assignments[var_identifier] != assignment:
                        #set None to conflict variable
                        new_assignments["Error"] = clause_track[var_identifier], unit_c.index, var_identifier
                        break
                else:
                    new_assignments[var_identifier] = assignment
                    clause_track[var_identifier] = unit_c.index
        return new_assignments


    def simplify(self, cnf_formula, assignment):
        ls_simplified_cnf_clauses = []
        for clause in cnf_formula.clauses:
            if not clause.assign(assignment):
                clause = self.remove_false_variables(clause, assignment)
                ls_simplified_cnf_clauses.append(clause)
        return Formula(ls_simplified_cnf_clauses)


    def remove_false_variables(self, clause, assignment):
        for var, value in assignment.items():
            if var in clause.variables and value == False:
                clause.remove(var)
            if self.negate(var) in clause.variables and value == True:
                clause.remove(self.negate(var))
        return clause


    def is_negated(self, variable):
        return '~' in variable

    def negate(self, variable):
        return '~' + variable





