import numpy as np
import copy
import re


class Equation:

    def __init__(self, left_hand_side, operator, right_hand_side):
        self.right_hand_side = right_hand_side
        self.left_hand_side = left_hand_side
        self.operator = operator

    def __str__(self):
        s = ''
        s += str(self.left_hand_side)
        s += str(self.operator)
        s += str(self.right_hand_side)
        return s

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(str(self))

class SimplexUtils:

    EQUATION_OPERATORS = ['<', '<=', '=', '>=', '>']
    EPS = 0.000001

    POSTFIX_PP = 'pp'
    POSTFIX_P = 'p'

    def reverse_operator(operator):
        if operator == '=':
            return '='
        elif operator == '>=':
            return '<='
        elif operator == '>':
            return '<'
        elif operator == '<=':
            return '>='
        elif operator == '<':
            return '>'
        elif operator == '!=':
            return '!='
        else:
            raise NotImplementedError


    def convert_to_simplex_form(equation):
        obj_equation = SimplexUtils.split_equation(equation)
        obj_simplex_equation = SimplexUtils.simplexify_equation(obj_equation)
        return obj_simplex_equation


    def split_equation(equation):
        tokens = equation.split()
        left_hand_side = []
        right_hand_side = []
        left_side = True
        operator = None
        for token in tokens:
            if token in SimplexUtils.EQUATION_OPERATORS:
                left_side = False
                operator = token
                continue
            if left_side:
                left_hand_side.append(token)
            else:
                right_hand_side.append(token)
        return Equation(left_hand_side, operator, right_hand_side)

    def simplexify_equation(equation):
        '''
        convert to Ax <= b form 
        '''
        b_side_multiplier = -1
        b_side_epsilon = 0
        b_side = 0

        lhs, op, rhs = None, equation.operator, None

        # print(str(equation))
        
        if op == '<':
            op = '<='
            b_side_epsilon -= SimplexUtils.EPS
        if op == '>':
            op = '>='
            b_side_epsilon -= SimplexUtils.EPS
        
        if op == '<=':
            simplex_ax_side = SimplexUtils.move_terms_from_to(equation.right_hand_side, equation.left_hand_side)
        else:
            simplex_ax_side = SimplexUtils.move_terms_from_to(equation.left_hand_side, equation.right_hand_side)
            op = SimplexUtils.reverse_operator(op)
        
        try:
            b_side = b_side_multiplier * simplex_ax_side[None]
            del simplex_ax_side[None]
        except KeyError:
            pass
        
        b_side += b_side_epsilon
        simplex_ax_side = SimplexUtils.convert_to_unrestricted_form(simplex_ax_side)
        return Equation(simplex_ax_side, op, b_side)


    def convert_to_unrestricted_form(dict_expression):
        '''
        assumes there are no constants!
        '''
        dict_urs_form = {}
        for key in dict_expression:
            a, b = dict_expression[key], -dict_expression[key]
            dict_urs_form[key+SimplexUtils.POSTFIX_P] = a
            dict_urs_form[key+SimplexUtils.POSTFIX_PP] = b
        return dict_urs_form

    def move_terms_from_to(side_from, side_to):
        defaultdict_from = SimplexUtils.convert_to_dict(side_from)
        defaultdict_to = SimplexUtils.convert_to_dict(side_to)
        for key, value in defaultdict_from.items():
            if key not in defaultdict_to:
                defaultdict_to[key] = 0
            defaultdict_to[key] -= defaultdict_from[key]
        dict_no_zeros = SimplexUtils.remove_zero_terms(defaultdict_to)
        return dict_no_zeros


    def remove_zero_terms(expression):
        simplified_expression = {}
        for variable, coefficient in expression.items():
            if expression[variable] != 0:
                simplified_expression[variable] = coefficient
        return simplified_expression


    def convert_to_dict(expression):
        vars_coefficients = {}
        negate = False
        for term in expression:
            if term == '-':
                negate = True
            elif term == '+':
                negate = False
            else:
                coefficient, variable = SimplexUtils.parse_term(term)
                if negate:
                    coefficient *= -1
                if variable not in vars_coefficients:
                    vars_coefficients[variable] = 0
                vars_coefficients[variable] += coefficient
        return vars_coefficients


    def negate_simplex(equation):
        '''
        only works on simplex form
        '''
        lhs = copy.deepcopy(equation.left_hand_side)
        rhs = copy.deepcopy(equation.right_hand_side)
        for key in lhs:
            lhs[key] *= -1
        rhs *= -1
        rhs -= SimplexUtils.EPS
        return Equation(lhs, '<=', rhs)

    def parse_term(term):
        match = re.search(r'([-]?[0-9][\.]?[0-9]{0,})?(\*?([a-z]+[0-9]*))?$', term)
        coefficient = 1
        if match.group(1):
            if match.group(1) == '-':
                coefficient = -1
            else:
                coefficient = float(match.group(1))
        # the variable will be None if there isn't one
        variable = match.group(3)
        return (coefficient, variable)


class Simplex:

    def __init__(self, parsed_input, rows=None, c=None):
        if rows != None:
            self.A = parsed_input.A[rows]
            self.b = parsed_input.b[rows]
        else:
            self.A = parsed_input.A
            self.b = parsed_input.b

        self.col_dict = parsed_input.col_dict
        self.c = c
        self.B = []
        self.I = []

        # print('A', self.A)
        # print('b', self.b)

        # print(parsed_input.row_dict)
        # print(parsed_input.col_dict)

    def init_phase(self):
        m, n = self.A.shape
        if np.all(self.b >= 0):
            self.I = np.arange(1 + m + n)
            return True
        
        self.A = np.concatenate((
            -np.ones((m, 1), np.float32), 
            self.A, 
            np.identity(m, np.float32)), 1)

        idx = np.arange(self.A.shape[1])
        B = idx[-m:]
        I = idx[:n+1]
        self.c = np.zeros(1+m+n, np.float32)
        self.c[0] = -1

        # Force x0 to enter
        B, I, obj, _ = self.pivot(B, I, True)
        terminated = False
        while not terminated:
            B, I, obj, terminated = self.pivot(B, I, False)

        self.B = B
        self.I = I

        if obj < 0.0:
            return False # Infeasible
        return True # Feasible


    def pivot(self, B, I, forced):
        Ab = self.A[:,B]
        Ai = self.A[:,I]
        cb = self.c[B]
        ci = self.c[I]

        # pi = Ab \ cb

        pi = np.linalg.solve(np.transpose(Ab), cb)
        obj = np.dot(pi, self.b)
        c_hat = ci - np.dot(pi, Ai)

        # Choose enter index
        if forced:
            enter = 0
        else:
            if (c_hat.max() <= 0):
                return B, I, obj, True # Done
            enter = np.argmax(c_hat)

        b_hat = np.linalg.solve(Ab, self.b)
        a_j_hat = -np.linalg.solve(Ab, self.A[:, I[enter]])

        # Search for leave index
        if forced:
            leave = self.b.argmin()
            leavelim = -b_hat[leave] / a_j_hat[leave]
        else:
            leave = -1
            leavelim = np.Inf
            for i in np.arange(Ab.shape[0]):
                if a_j_hat[i] < 0:
                    ll = -b_hat[i] / a_j_hat[i]
                    if ll < leavelim:
                        leavelim = ll
                        leave = i
            if leave == -1:
                return B, I, obj, True # Unbounded

        temp = I[enter]
        I[enter] = B[leave]
        B[leave] = temp

        obj= np.dot(pi, self.b) + c_hat[enter] * leavelim
        return B, I, obj, False

    def get_assignment(self):
        result_dict = {}
        m, n = self.A.shape

        for i in self.I:
            result_dict[i] = 0

        idx = 0
        for i in self.B:
            Ab = self.A[:,self.B]
            Ai = self.A[:,self.I]
            b_hat = np.linalg.solve(Ab, self.b)
            result_dict[i] = b_hat[idx]
            idx += 1

        result = []
        for key, val in self.col_dict.items():
            if val <= n:
                result.append((key, result_dict[val+1]))
        orig_variables = {}
        for ans in result:
            var, value = ans[0], ans[1]
            if SimplexUtils.POSTFIX_PP in var:
                var = var.replace(SimplexUtils.POSTFIX_PP, '')
                value *= -1
            else:
                var = var.replace(SimplexUtils.POSTFIX_P, '')

            if var not in orig_variables:
                orig_variables[var] = 0
            orig_variables[var] += value
        return orig_variables

    def solve(self):
        return self.init_phase()

