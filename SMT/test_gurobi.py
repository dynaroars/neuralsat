import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)

x1 = model.addVar(name='x1', lb=0, ub=1)
x2 = model.addVar(name='x2', lb=0, ub=1)
# var = [x1, x2]

v = x1 + 2*x2
t = x1 - x2
model.update()

# print(model)

model.setObjective(v + t, grb.GRB.MAXIMIZE)

model.update()
model.reset()
model.optimize()
print(model.status == grb.GRB.OPTIMAL)

# print(model.status == grb.GRB.OPTIMAL)

print(x1.X)
print(x2.X)
# print(var[1].X)