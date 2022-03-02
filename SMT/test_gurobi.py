import gurobipy as grb

model = grb.Model()
model.setParam('OutputFlag', False)
model.setParam('Threads', 1)

x1 = model.addVar(name='x1')
x2 = model.addVar(name='x2')
model.update()

# print(str(eval('1.1*x1 - 2*x2')))
model.addConstr(x1 <= 0)
model.addConstr(x1 >= 1e-5)
model.setObjective(0, grb.GRB.MAXIMIZE)
# print(model)
# model.remove(model.getConstrs())
model.update()
model.optimize()
print(model.status)
# print(model)
