from dnn_solver.dnn_solver import DNNSolver

meta = {
    'a00': [1, '1x0 - 1x1'],
    'a01': [2, '1x0 + 1x1'],
    'a10': [3, '0.5n00 - 0.2n01'],
    'a11': [4, '-0.5n00 + 0.1n01'],
    'y0': '1n10 - 1n11',
    'y1': '-1n10 + 1n11',
}

conditions = {
    'in': '(and (x0 < 1) (x1 > 2))',
    'out': '(y0 > y1)'
}

# print(1 in {1: {'value': False, 'clause': None, 'level': 0, 'idx': 0}})
# exit()
solver = DNNSolver(meta, conditions)
print(solver.solve())