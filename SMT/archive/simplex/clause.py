
class Clause:
    
    def __init__(self, variables):
        # expect in form [x1, x2, x3]
        self.variables = variables
        self.index = None

    def __str__(self):
        return '[' + ', '.join(self.variables) + ']'

    def assign(self, dict_assignments):
        # expect in form {x1: True, x2: False, x3: False}
        for var in self.variables:
            not_negated = True
            var_index = var
            if "~" in var:
                not_negated = False
                var_index = var.replace("~", "")
            if var_index in dict_assignments:
                if not_negated == dict_assignments[var_index]:
                    return True
        return False


    def remove(self, var):
        self.variables.remove(var)


class Formula:

    def __init__(self, clauses):
        self.clauses = clauses

    @property
    def vars(self):
        return list(set([v for c in self.clauses for v in c.variables if '~' not in v]))

    def __str__(self):
        s = ''
        for clause in self.clauses:
            s += str(clause.variables)
        return s

    def assign(self, dict_assignments):
        for clause in self.clauses:
            # print('    - check assign: ', clause, clause.assign(dict_assignments))
            if not clause.assign(dict_assignments):
                return False
        return True

    def get_set_variables(self):
        variables = set()
        for clause in self.clauses:
            for var in clause.variables:
                variables.add(var.replace("~", ""))
        return variables

    def has_empty_clause(self):
        return len(list(filter(lambda clause: 0 == len(clause.variables), self.clauses))) >= 1


    def highest_frequent_variable(self):
        dict_counts = {}
        for clause in self.clauses:
            for var in clause.variables:
                var = var.replace("~", "")
                if var in dict_counts:
                    dict_counts[var] += 1
                else:
                    dict_counts[var] = 1
        max_count = 0
        max_variable = ""
        for key, val in dict_counts.items():
            if val > max_count:
                max_count = val
                max_variable = key
        return max_variable