import copy
import z3


z3.set_option(rational_to_decimal=True)

class Utils:

    def And(term1, term2):
        return f'(and {term1} {term2})'

    def Not(term1):
        return f'(not {term1})'

    def Or(term1, term2):
        return f'(or {term1} {term2})'

    def Prove(term1, term2):
        return Utils.And(term1, Utils.Not(term2))


class DNNConstraint:

    def __init__(self, dnn, conditions):
        self.dnn = dnn
        self.conditions = conditions

    def _find_nodes(self, assignment):
        nodes = []
        assigned_nodes = set(list(assignment.keys()))
        for out_node, in_nodes in self.dnn.items():
            if out_node in assigned_nodes:
                continue
            tmp = set([i[1].replace('n', 'a') for i in in_nodes])
            if tmp.issubset(assigned_nodes) or \
                len(list(tmp)) == len(list(filter(lambda x: x.startswith('x'), tmp))): 
                nodes.append(out_node)
        return nodes

    def _construct_formula(self, node, assignment):
        variables = []
        node_name = str(node).replace('n', 'a')
        if assignment.get(node_name, None) is False:
            return [], z3.RealVal(0)

        output = 0
        for weight, name in self.dnn[node_name]:
            v = z3.Real(name)
            if not name.startswith('x'):
                variables.append(v)
            output += weight * v
        del v
        return variables, output

    def _construct_constraint_node(self, node, assignment):
        variables, formula = self._construct_formula(node, assignment)
        while variables:
            variables, formula = self._recursive(variables, formula, assignment)
        return z3.simplify(formula)

    def _recursive(self, vs, f, assignment):
        new_vs = []
        for v in vs:
            new_v, new_f = self._construct_formula(v, assignment)
            f = z3.substitute(f, (v, new_f))
            new_vs.extend(new_v)
        return new_vs, f

    def _generate_constraints(self, assignment):
        nodes = self._find_nodes(assignment)
        if not nodes:
            return None

        constraint = self.conditions['in']

        for node, status in assignment.items():
            f = self._construct_constraint_node(node, assignment)
            if status:
                # f = self._construct_constraint_node(node, assignment)
                constraint = Utils.And(constraint, '(%s)' % str(f > 0))
            # else:
                # flip_assignment = copy.deepcopy(assignment)
                # flip_assignment[node] = not status
                # f = self._construct_constraint_node(node, flip_assignment)
                # constraint = Utils.And(constraint, '(%s)' % str(f <= 0))

        implies = {}
        if nodes[0].startswith('y'):
            tmp = self.conditions['out']
            for node in nodes:
                tmp = tmp.replace(node, str(self._construct_constraint_node(node, assignment)))
            constraint = Utils.And(constraint, tmp) # prove(f, not(g)) = f and g
            # return constraint
        else:
            for node in nodes:
                f = self._construct_constraint_node(node, assignment)
                implies[node] = [Utils.Prove(constraint, '(%s)' % str(f <= 0)), Utils.Prove(constraint, '(%s)' % str(f > 0))]
        return constraint, implies


    def __call__(self, assignment):

        return self._generate_constraints(assignment)


if __name__ == '__main__':
        
    # dnn = {
    #     'a00': '1x0 - 1x1',
    #     'a01': '1x0 + 1x1',
    #     'a10': '0.5n00 - 0.2n01',
    #     'a11': '-0.5n00 + 0.1n01',
    #     'y0': '1n10 - 1n11',
    #     'y1': '-1n10 + 1n11',
    # }

    dnn = {
        'a00': [(1.0, 'x0'), (-1.0, 'x1')],
        'a01': [(1.0, 'x0'), (1.0, 'x1')],
        'a10': [(0.5, 'n00'), (-0.2, 'n01')],
        'a11': [(-0.5, 'n00'), (0.1, 'n01')],
        'y0' : [(1.0, 'n10'), (-1.0, 'n11')],
        'y1' : [(-1.0, 'n10'), (1.0, 'n11')],
    }

    assignment = {
        'a00': False,
        'a01': True,
        'a10': True,
        'a11': False,
    }


    conditions = {
        'in': '(and (x0 < 0) (x1 > 1))',
        'out': '(y0 > y1)'
    }


    dnn_constraint = DNNConstraint(dnn, conditions)
    print(dnn_constraint(assignment))
