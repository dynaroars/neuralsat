import pulp
import numpy as np


def main():
    prob = pulp.LpProblem('lp', pulp.LpMinimize)
    x = pulp.LpVariable('x', lowBound=None, upBound=5, cat=pulp.LpContinuous)
    y = pulp.LpVariable('y', lowBound=None, upBound=5, cat=pulp.LpContinuous)

    variables = np.array([x, y], dtype=object)

    prob.setObjective(np.sum(variables * [5, 4]))
    prob += x + y >= 8
    prob += 2 * x + y >= 10
    prob += x + 4 * y >= 11
    prob += 5 * x + 4 * y == 0
    print(prob)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(pulp.LpStatus[prob.status])
    print(x.value(), y.value())


if __name__ == '__main__':
    main()
