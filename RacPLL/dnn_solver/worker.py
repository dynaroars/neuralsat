# from utils.terminatable_thread import *

import gurobipy as grb
import contextlib
import os

def implication_gurobi_worker(assignment, mat_dict, nodes, shared_queue, kwargs):
    if len(nodes) == 0:
        return None

    n_vars, lbs, ubs = kwargs
    with contextlib.redirect_stdout(open(os.devnull, 'w')):

        with grb.Env() as env, grb.Model(env=env) as model:
            model.setParam('OutputFlag', False)

            variables = [
                model.addVar(name=f'x{i}', lb=lbs[i], ub=ubs[i]) for i in range(n_vars)
            ]
            model.update()

            for node in mat_dict:
                status = assignment.get(node, None)
                if status is None:
                    continue
                mat = mat_dict[node]
                eqx = grb.LinExpr(mat[:-1], variables) + mat[-1]
                if status:
                    model.addLConstr(eqx >= 1e-6)
                else:
                    model.addLConstr(eqx <= 0)

            results = []
            for node in nodes:
                res = {'pos': False, 'neg': False}
                mat = mat_dict[node]
                obj = grb.LinExpr(mat[:-1], variables) + mat[-1]

                model.setObjective(obj, grb.GRB.MINIMIZE)
                model.update()
                model.reset()
                model.optimize()

                if model.status == grb.GRB.OPTIMAL:
                    res['pos'] = True if model.objval > 0 else False

                model.setObjective(obj, grb.GRB.MAXIMIZE)
                model.update()
                model.reset()
                model.optimize()

                if model.status == grb.GRB.OPTIMAL:
                    res['neg'] = True if model.objval <= 0 else False

                results.append((node, res))

            shared_queue.put(results)



    # except ThreadTerminatedError:
    #     # time.sleep(0.01)
    #     print(f'{name} terminated')
    #     return None