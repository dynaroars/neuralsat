from sat_solver.sat_solver_2 import SATSolver


# clause1 = frozenset({-4, 5, -1})
# clause2 = frozenset({-4, 6})
# clause3 = frozenset({-6, -5, 7})
# clause4 = frozenset({8, -7})
# clause5 = frozenset({-7, -2, 9})
# clause6 = frozenset({-8, -9})
# clause7 = frozenset({-8, 9})
# formula = {clause1, clause2, clause3, clause4, clause5, clause6, clause7}

# assignment = {
#     1: {"value": True, "clause": None, "level": 1, "idx": 1},
#     2: {"value": True, "clause": None, "level": 2, "idx": 1},
#     3: {"value": True, "clause": None, "level": 3, "idx": 1},
#     4: {"value": True, "clause": None, "level": 4, "idx": 1},
# }


solver = SATSolver()
print(solver.solve())
print(solver.get_assignment())