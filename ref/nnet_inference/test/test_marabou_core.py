from prophecy import import_marabou

import_marabou()

from maraboupy import MarabouCore as mc
from maraboupy.Marabou import createOptions


def main():
    # example for Corina net
    large = 999999
    eps = 1

    # setup input query
    ipq = mc.InputQuery()
    ipq.setNumberOfVariables(12 + 1)  # 1 additional variable for max of other outputs

    # set lower and upper bounds of variables
    ipq.setLowerBound(0, -5)
    ipq.setUpperBound(0, 5)
    ipq.setLowerBound(1, -5)
    ipq.setUpperBound(1, 5)

    ipq.setLowerBound(2, -large)
    ipq.setUpperBound(2, large)
    ipq.setLowerBound(3, -large)
    ipq.setUpperBound(3, large)

    ipq.setLowerBound(4, 0)
    ipq.setUpperBound(4, large)
    ipq.setLowerBound(5, 0)
    ipq.setUpperBound(5, 0)

    ipq.setLowerBound(6, -large)
    ipq.setUpperBound(6, large)
    ipq.setLowerBound(7, -large)
    ipq.setUpperBound(7, large)

    ipq.setLowerBound(8, 0)
    ipq.setUpperBound(8, large)
    ipq.setLowerBound(9, 0)
    ipq.setUpperBound(9, large)

    ipq.setLowerBound(10, -large)
    ipq.setUpperBound(10, large)
    ipq.setLowerBound(11, -large)
    ipq.setUpperBound(11, large)

    # Add network constraints
    eq00 = mc.Equation()
    eq00.addAddend(1, 0)
    eq00.addAddend(-1, 1)
    eq00.addAddend(-1, 2)
    eq00.setScalar(0)
    ipq.addEquation(eq00)

    eq01 = mc.Equation()
    eq01.addAddend(1, 0)
    eq01.addAddend(1, 1)
    eq01.addAddend(-1, 3)
    eq01.setScalar(0)
    ipq.addEquation(eq01)

    mc.addReluConstraint(ipq, 2, 4)
    mc.addReluConstraint(ipq, 3, 5)

    eq10 = mc.Equation()
    eq10.addAddend(0.5, 4)
    eq10.addAddend(-0.2, 5)
    eq10.addAddend(-1, 6)
    eq10.setScalar(0)
    ipq.addEquation(eq10)

    eq11 = mc.Equation()
    eq11.addAddend(-0.5, 4)
    eq11.addAddend(0.1, 5)
    eq11.addAddend(-1, 7)
    eq11.setScalar(0)
    ipq.addEquation(eq11)

    mc.addReluConstraint(ipq, 6, 8)
    mc.addReluConstraint(ipq, 7, 9)

    eq20 = mc.Equation()
    eq20.addAddend(1, 8)
    eq20.addAddend(-1, 9)
    eq20.addAddend(-1, 10)
    eq20.setScalar(0)
    ipq.addEquation(eq20)

    eq21 = mc.Equation()
    eq21.addAddend(-1, 8)
    eq21.addAddend(1, 9)
    eq21.addAddend(-1, 11)
    eq21.setScalar(0)
    ipq.addEquation(eq21)

    # Add output properties
    mc.addMaxConstraint(ipq, {11}, 12)

    eq_out = mc.Equation(mc.Equation.LE)
    eq_out.addAddend(1, 10)
    eq_out.addAddend(-1, 12)
    eq_out.setScalar(-0.5)
    ipq.addEquation(eq_out)

    # solve
    options = createOptions(verbosity=False)
    res, vars, stats = mc.solve(ipq, options)
    print(res)
    print(vars)
    print(stats)


if __name__ == '__main__':
    main()
