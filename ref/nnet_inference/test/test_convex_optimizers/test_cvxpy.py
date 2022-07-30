import cvxpy as cp
import numpy as np


def main():
    x = cp.Variable(name='x')
    y = cp.Variable(name='y')

    cx = cp.Variable(name='cx', boolean=True)
    cy = cp.Variable(name='cy', boolean=True)
    z = cp.Variable(name='z')

    variables = np.array([x, y], dtype=object)

    objective = cp.Minimize(np.sum(variables * [5, 4]))
    constraints = [
        x >= 0, y >= 0,  # lower bounds
        x <= 5, y <= 5,  # upper bounds
        x + y >= 8,
        2 * x + y >= 10,
        x + 4 * y >= 11,
        cx * x + cy * y == z,
        cx + cy == 1,
        # x + y == 8,
        # z <= cp.maximum(x, y),
    ]
    prob = cp.Problem(objective, constraints)
    print(prob)

    prob.solve()
    print(prob.status)
    print(prob.objective.value)
    print(x.value, y.value)
    print(cp.maximum(x, y))


if __name__ == '__main__':
    main()
