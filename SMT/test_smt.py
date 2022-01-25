from smt_solver import SMTSolver

def main():
    formula = '''
        (declare-fun x1 () Real) 
        (declare-fun x2 () Real) 
        (declare-fun x3 () Real) 
        (declare-fun x4 () Real) 
        (declare-fun x5 () Real) 

        (assert 
            (not (<=> (<= (-4.943x1+-0.052x2+-0.947x3+-1.763x4+0.338x5) 4.269) 
            (or (and (<= (3.18x1+-2.269x2+-4.53x3+-4.967x4+-4.368x5) 1.352) 
            (and (<= (3.18x1+-2.269x2+-4.53x3+-4.967x4+-4.368x5) 1.352) 
            (<= (4.837x1+-3.914x2+4.692x3+3.785x4+-3.853x5) 4.433))) 
            (=> (and (<= (3.18x1+-2.269x2+-4.53x3+-4.967x4+-4.368x5) 1.352) 
            (<= (4.837x1+-3.914x2+4.692x3+3.785x4+-3.853x5) 4.433)) 
            (<= (4.412x1+2.374x2+3.088x3+-2.112x4+-3.022x5) -4.806))))))
    '''

    formula = '''
        (declare-fun x1 () Real)
        (declare-fun x2 () Real)

        (assert (or (<= (x1) 1) (<= (-x2) -3))
        (assert (and (>= (x1) 2) (<= (-x1) -1))
    '''


    # solver = SMTSolver(formula)
    # print(solver.solve())
    # print(solver.get_assignment())
    from utils.formula_parser import FormulaParser
    print(FormulaParser.import_tq(formula)[-1])

if __name__ == '__main__':
    main()

