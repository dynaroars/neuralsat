from dnn_solver.dnn_solver import DNNSolver

dnn = {
    'a00': [(1.0, 'x0'), (-1.0, 'x1')],
    'a01': [(1.0, 'x0'), (1.0, 'x1')],
    'a10': [(0.5, 'n00'), (-0.2, 'n01')],
    'a11': [(-0.5, 'n00'), (0.1, 'n01')],
    'y0' : [(1.0, 'n10'), (-1.0, 'n11')],
    'y1' : [(-1.0, 'n10'), (1.0, 'n11')],
}

vars_mapping = {
    'a00': 1,
    'a01': 2,
    'a10': 3,
    'a11': 4,
}

output_vars = ['y0', 'y1']

conditions = {
    'in': '(and (x0 < 0) (x1 > 1))',
    'out': '(y0 < y1)'
}

solver = DNNSolver(dnn, vars_mapping, conditions)
print(solver.solve())