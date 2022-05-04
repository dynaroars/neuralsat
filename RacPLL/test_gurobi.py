import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)
model.setParam('PoolSearchMode', 2)
model.setParam('PoolSolutions', 5)


x1 = model.addVar(name='x1', lb=-1, ub=1)
x2 = model.addVar(name='x2', lb=-2, ub=2)
# var = [x1, x2]

c1 = model.addConstr(-x1 - 0.5 * x2 - 1 >= 0)
model.update()

# v = x1 + 2*x2
# t = x1 - x2

# print(model)

model.setObjective(x1 + x2, grb.GRB.MAXIMIZE)

model.update()
model.reset()
model.optimize()
print(model.status == grb.GRB.OPTIMAL)

print(x1.X, x2.X)
# print(var[1].X)

model.setObjective(x1 + x2, grb.GRB.MINIMIZE)

model.update()
model.reset()
model.optimize()
print(model.status == grb.GRB.OPTIMAL)

print(x1.X, x2.X)
# print(var[1].X)