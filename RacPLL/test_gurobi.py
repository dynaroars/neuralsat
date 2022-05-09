import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)


x1 = model.addVar(name='x1', lb=-1, ub=1)
x2 = model.addVar(name='x2', lb=-2, ub=2)
# var = [x1, x2]

c1 = model.addConstr(x1 >= 0)
c2 = model.addConstr(x1 <= -0.5)
c3 = model.addConstr(x2 <= -0.5)
model.update()

# model.setObjective(0, grb.GRB.MINIMIZE)

# model.update()
# model.reset()
model.optimize()
print(model.status == grb.GRB.OPTIMAL)
print(model.status == grb.GRB.INFEASIBLE)

model.computeIIS()
print(c1.IISConstr)
print(c2.IISConstr)
print(c3.IISConstr)
# print(var[1].X)

