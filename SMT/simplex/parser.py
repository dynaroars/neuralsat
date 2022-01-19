from pprint import pprint
import numpy as np
import re

from simplex.clause import Clause, Formula
from simplex.simplex import SimplexUtils

class ParsedInput:

    def __init__(self, A, b, cnf, row_dict, col_dict, vars_dict):
        self.A = A
        self.b = b
        self.cnf = cnf
        self.row_dict = row_dict
        self.col_dict = col_dict
        self.vars_dict = vars_dict
        self.reversed_vars_dict = {v: k for k, v in vars_dict.items()}

    @property
    def formula(self):
        self._formula = frozenset({})
        for clause in self.cnf.clauses:
            tmp_clause = frozenset({})
            for var in clause.variables:
                tmp_clause |= {self.vars_dict[var]}
            self._formula |= {tmp_clause}
        return self._formula


class ParserUtils:

    BOOL_OPERATORS = ['and', 'or', 'nand', 'nor', 'xor', 'xnor', '->']

    def tseitin_transform_and(var1, var2, var3):
        return '( ~{0} v ~{1} v {2} ) ^ ( {0} v ~{2} ) ^ ( {1} v ~{2} )'.format(var1, var2, var3)

    def tseitin_transform_or(var1, var2, var3):
        return '( {0} v {1} v ~{2} ) ^ ( ~{0} v {2} ) ^ ( ~{1} v {2} )'.format(var1, var2, var3)

    def tseitin_transform_not(var1, var2):
        return '( ~{0} v ~{1} ) ^ ( {0} v {1} )'.format(var1, var2)

    def tseitin_transform_nand(var1, var2, var3):
        return '( ~{0} v ~{1} v ~{2} ) ^ ( {0} v {2} ) ^ ( {1} v {2} )'.format(var1, var2, var3)

    def tseitin_transform_nor(var1, var2, var3):
        return '( {0} v {1} v {2} ) ^ ( ~{0} v ~{2} ) ^ ( ~{1} v ~{2} )'.format(var1, var2, var3)

    def tseitin_transform_xor(var1, var2, var3):
        return '( ~{0} v ~{1} v ~{2} ) ^ ( {0} v {1} v ~{2} ) ^ ( {0} v ~{1} v {2} ) ^ ( ~{0} v {1} v {2} )'.format(var1, var2, var3)

    def tseitin_transform_xnor(var1, var2, var3):
        return '( {0} v {1} v {2} ) ^ ( ~{0} v ~{1} v {2} ) ^ ( ~{0} v {1} v ~{2} ) ^ ( {0} v ~{1} v ~{2} )'.format(var1, var2, var3)

    def tseitin_transform_implies(var1, var2, var3):
        return '( {0} v {1} v {2} ) ^ ( ~{0} v {1} v ~{2} ) ^ ( {0} v ~{1} v {2} ) ^ ( ~{0} v ~{1} v {2} )'.format(var1, var2, var3)

    def is_literal(term):
        if term.count('(') == 0 and term.count(')') == 0:
            return True
        if ParserUtils.is_math_equation(term):
            return True
        return False

    def is_math_equation(term):
        match = re.search(r'^\([^\(]+(<|<=|>=|[^-]>)[^/)]+\)$', term)
        if match:
            return True
        return False

    def is_math_equality(term):
        match = re.search(r'^\([^\(]+[^!][=][^/)]+\)$', term)
        if match:
            return True
        return False

    def is_math_not_equals(term):
        match = re.search(r'^\([^\(]+[!=][^/)]+\)$', term)
        if match:
            return True
        return False

    def expand_equality(term):
        return '(and ' + term.replace('=', '<=') + ' ' + term.replace('=', '>=') + ')'

    def expand_not_equals(term):
        return '(or ' + term.replace('!=', '<') + ' ' + term.replace('!=', '>') + ')'


    def remove_outer_parens(term):
        if term[0] == '(' and term[-1] == ')':
            return term[1:-1]
        else:
            return term

    def balanced_parentheses(input_str):
        imbalanced = 0
        for c in list(input_str):
            if c == '(':
                imbalanced += 1
            elif c == ')':
                imbalanced -= 1
        return imbalanced == 0


class Parser:

    equation_counter = 0
    equation_dictionary = {}

    def get_operator_and_variables(term):
        match = re.search(r'^\(({0}) (\w+|\(.+\))\)$'.format('not'), term)
        if match:
            return match.group(1), match.group(2), None

        operators = '|'.join(ParserUtils.BOOL_OPERATORS)
        match = re.search(r'^\(({0}) (.+)\)$'.format(operators), term)
        if match:
            imbalanced = 0
            on_term_1 = True
            term_1 = ''
            term_2 = ''
            first_space = True
            terms = re.split(r'(\(|\)|\s)',match.group(2))
            terms = filter(lambda x: x != '', terms)

            for char in terms:
                if on_term_1:
                    if char == '(':
                        imbalanced += 1
                    elif char == ')':
                        imbalanced -= 1
                    term_1 += char
                    if not imbalanced:
                        on_term_1 = False
                else:
                    term_2 += char
            return match.group(1), term_2.strip(), term_1.strip()

    def get_associated_variable(str_equation):
        obj_simplex_form = SimplexUtils.convert_to_simplex_form(
            ParserUtils.remove_outer_parens(str_equation))

        if obj_simplex_form in Parser.equation_dictionary:
            return Parser.equation_dictionary[obj_simplex_form]
        else:
            new_var = 'q' + str(Parser.equation_counter)
            Parser.equation_dictionary[obj_simplex_form] = new_var
            Parser.equation_counter += 1
            return new_var

    def convert_to_cnf(formula_str):
        return Parser.add_to_cnf('', 'xi', formula_str)

    def add_to_cnf(current_cnf, output_variable, new_term):
        if ParserUtils.is_math_equality(new_term):
            new_term = ParserUtils.expand_equality(new_term)
        if ParserUtils.is_math_not_equals(new_term):
            new_term = ParserUtils.expand_not_equals(new_term)

        operator, left_term, right_term = Parser.get_operator_and_variables(new_term)

        # print(left_term, operator, right_term)

        if ParserUtils.is_math_equation(left_term):
            left_term = Parser.get_associated_variable(left_term)

        if right_term and ParserUtils.is_math_equation(right_term):
            right_term = Parser.get_associated_variable(right_term)

        left_var, right_var = left_term, right_term
        if not ParserUtils.is_literal(left_term):
            if not current_cnf:
                current_cnf += Parser.add_to_cnf(current_cnf, output_variable + 'L', left_term)
            else:
                current_cnf += ' ^ ' + add_to_cnf(current_cnf, output_variable + 'L', left_term)
            left_var = output_variable + 'L'

        right_cnf = ''
        if right_term and not ParserUtils.is_literal(right_term):
            if not right_cnf:
                right_cnf += Parser.add_to_cnf(right_cnf, output_variable + 'R', right_term)
            else:
                right_cnf += ' ^ ' + Parser.add_to_cnf(right_cnf, output_variable + 'R', right_term)
            right_var = output_variable + 'R'

        # print(left_var, right_var)

        if right_cnf and current_cnf:
            current_cnf += ' ^ ' + right_cnf
        elif not current_cnf and right_cnf:
            current_cnf += right_cnf

        new_term_append = ''
        if current_cnf != '':
            new_term_append = ' ^ '
        if operator == 'and':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_and(left_var, right_var, output_variable)
        elif operator == 'or':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_or(left_var, right_var, output_variable)
        elif operator == 'not':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_not(left_var, output_variable)
        elif operator == 'nand':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_nand(left_var, right_var, output_variable)
        elif operator == 'nor':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_nor(left_var, right_var, output_variable)
        elif operator == 'xor':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_xor(left_var, right_var, output_variable)
        elif operator == 'xnor':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_xnor(left_var, right_var, output_variable)
        elif operator == '->':
            current_cnf += new_term_append + ParserUtils.tseitin_transform_implies(left_var, right_var, output_variable)
        return current_cnf

    def convert_to_obj_cnf(str_cnf):
        ls_clauses = []
        str_clauses = str_cnf.split('^')
        for str_clause in str_clauses:
            ls_variables = re.split(r' |\(|\)|v',str_clause)
            ls_variables = list(filter(lambda x: x != '', ls_variables))
            ls_clauses.append(Clause(ls_variables))
            # print(str_clause, ls_variables)
        return Formula(ls_clauses)

    def convert_to_simplex_interface(dict_var_to_eqn):
        # find all variables
        row_dict, col_dict = {}, {}
        i = 0
        for key, eqn in dict_var_to_eqn.items():
            # print('key:', key, '====> equation:', eqn)
            row_dict[key] = i
            i += 1
            for key, val in eqn.left_hand_side.items():
                col_dict[key] = 1
        # align variables
        i = 0
        for key in col_dict.keys():
            col_dict[key] = i
            i += 1
        # print(row_dict, col_dict)
        
        # build matrix inputs into simplex
        m = len(dict_var_to_eqn)
        n = len(col_dict)
        A = np.zeros((m,n), np.float32)
        b = np.zeros(m, np.float32)
        for _, eqn in dict_var_to_eqn.items():
            # print(_, eqn)
            if eqn.operator == "<=":
                for key, val in eqn.left_hand_side.items():
                    # print(key, val)
                    A[row_dict[_], col_dict[key]] = val
                    # print(row_dict[_], col_dict[key], val, eqn.right_hand_side)
                b[row_dict[_]] = eqn.right_hand_side
            else:
                raise

        return A, b, row_dict, col_dict


    def parse(formula_str):

        if not ParserUtils.balanced_parentheses(formula_str):
            raise "Input does not have balanced parentheses."
    
        cnf = Parser.convert_to_cnf(formula_str)
        # print(cnf)
        obj_cnf = Parser.convert_to_obj_cnf(cnf)

        dict_var_to_eqn = {}
        for key, val in Parser.equation_dictionary.items():
            dict_var_to_eqn[val] = key
            dict_var_to_eqn['~'+val] = SimplexUtils.negate_simplex(key)

        A, b, row_dict, col_dict = Parser.convert_to_simplex_interface(dict_var_to_eqn)

        vars_dict = {}
        for i, var in enumerate(obj_cnf.vars):
            vars_dict[var] = i+1
            vars_dict['~' + var] = -(i+1)

        parsed_input = ParsedInput(A, b, obj_cnf, row_dict, col_dict, vars_dict)

        return parsed_input

