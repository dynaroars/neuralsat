# from utils.terminatable_thread import *

import gurobipy as grb
import numpy as np
import contextlib
import time
import os

def implication_gurobi_worker(assignment, mat_dict, nodes, shared_queue, concrete, use_mvar, kwargs):
    if len(nodes) == 0:
        return None

    n_vars, lbs, ubs = kwargs
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
    # tic = time.time()
    # if True:
        with grb.Env() as env, grb.Model(env=env) as model:
            model.setParam('OutputFlag', False)
            model.setParam('Threads', 1)

            variables = [
                model.addVar(name=f'x{i}', lb=lbs[i], ub=ubs[i]) for i in range(n_vars)
            ]
            model.update()

            if len(assignment) > 0:
                # tic = time.time()
                if use_mvar:
                    mvars = grb.MVar(variables)

                    lhs = np.zeros([len(mat_dict), len(variables)])
                    rhs = np.zeros(len(mat_dict))
                    # mask = np.zeros(len(mat_dict), dtype=np.int32)
                    for i, node in enumerate(mat_dict):
                        status = assignment.get(node, None)
                        if status is None:
                            continue
                        # mask[i] = 1
                        if status:
                            lhs[i] = -1 * mat_dict[node][:-1]
                            rhs[i] = mat_dict[node][-1] - 1e-6
                        else:
                            lhs[i] = mat_dict[node][:-1]
                            rhs[i] = -1 * mat_dict[node][-1]

                    model.addConstr(lhs @ mvars <= rhs) 

                else:
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

                # print('setup: use_mvar:', use_mvar, time.time() - tic)

            results = []
            # tic = time.time()
            for node in nodes:
                res = {'pos': False, 'neg': False}
                mat = mat_dict[node]
                obj = grb.LinExpr(mat[:-1], variables) + mat[-1]

                if concrete[node] > 0:
                    # model.setObjective(obj, grb.GRB.MINIMIZE)
                    ci = model.addLConstr(obj <= 0)
                    # model.update()
                    # model.reset()
                    model.optimize()

                    if model.status == grb.GRB.INFEASIBLE:
                        res['pos'] = True # if model.objval > 0 else False

                # if res['pos']:
                #     results.append((node, res))
                #     continue
                else:
                    ci = model.addLConstr(obj >= 1e-6)
                    # model.update()
                    # model.reset()
                    model.optimize()

                    if model.status == grb.GRB.INFEASIBLE:
                        res['neg'] = True # if model.objval <= 0 else False
                model.remove(ci)

                results.append((node, res))

            shared_queue.put(results)
            # print('imply:', len(nodes), time.time() - tic)


    # except ThreadTerminatedError:
    #     # time.sleep(0.01)
    #     print(f'{name} terminated')
    #     return None