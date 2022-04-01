import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)

x1 = model.addVar(name='x1', lb=0, ub=1)
x2 = model.addVar(name='x2', lb=0, ub=1)
var = [x1, x2]

model.addConstr(x1 <= 0.5)
model.addConstr(0.22 <= x1 + x2)
model.addConstr(0.3 <= x2)
model.addConstr(x2 <= 0.4)
model.update()

print(model)

model.setObjective(-1000, grb.GRB.MAXIMIZE)

model.update()
model.reset()
print(model.status)
model.optimize()

print(model.status == grb.GRB.OPTIMAL)

print(var[0].X)
print(var[1].X)