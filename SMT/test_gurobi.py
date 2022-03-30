import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)

x1 = model.addVar(name='x1', lb=0, ub=1)
x2 = model.addVar(name='x2', lb=0, ub=1)
var = [x1, x2]
model.update()

for v in model.getVars():
    print(v, v.lb, v.ub)
    v.lb += 0.5
    v.ub += 1.5
# print(model)
model.update()

for v in model.getVars():
    print(v, v.lb, v.ub)


c1 = model.addConstr(x1 <= 2)
c1 = model.addConstr(0.1 <= x2)

model.setObjective(sum(var), grb.GRB.MAXIMIZE)

model.update()
model.reset()
model.optimize()

print(var[0].X)
print(var[1].X)