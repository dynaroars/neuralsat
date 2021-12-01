from utils import dimacs_parse as parse
from settings import *
import random

def bcp(formula, unit):
    "Boolean Constraint Propagation (BCP)"
    '''
        Input : an input formula (conjuntion of clause C), a litteral L
        Output : a modified formula

        Idea:
            if L in C  => remove C 
            if -L in C => remove -L from C
            else       => keep C
    '''

    modified = []
    for clause in formula:
        if unit in clause:
            continue
        if -unit in clause:
            c = [x for x in clause if x != -unit]
            if len(c) == 0:
                return -1
            modified.append(c)
        else:
            modified.append(clause)
    return modified


def get_counter(formula):
    "Count the number of occurences of each litteral in the formula"
    counter = {}
    for clause in formula:
        for literal in clause:
            if literal in counter:
                counter[literal] += 1
            else:
                counter[literal] = 1
    return counter

def unit_propagation(formula):
    "Unit Propagation (UP)"
    '''
        Input : a formula F
        Ouput : a modified formula
        
        Idea: 
        - find unit clauses (with len=1) 
        - then BCP w.r.t these unit variables    
    '''
    assignment = []
    unit_clauses = [c for c in formula if len(c) == 1]
    while len(unit_clauses) > 0:
        unit = unit_clauses[0]
        formula = bcp(formula, unit[0])
        assignment += [unit[0]]
        if formula == -1:
            return -1, []
        if not formula: 
            return formula, assignment
        unit_clauses = [c for c in formula if len(c) == 1]
    return formula, assignment

def pure_literal(formula):
    "Find pure litterals then BCP w.r.t. these pure litterals"
    counter = get_counter(formula)
    assignment = []
    pures = []
    for literal, _ in counter.items():
        if -literal not in counter: 
            pures.append(literal)
    for pure in pures:
        formula = bcp(formula, pure)
    assignment += pures
    return formula, assignment


def variable_selection(formula):
    "Variable selection heuristics: Randomly select"
    counter = get_counter(formula)
    return random.choice(list(counter.keys()))

def backtracking(formula, assignment):
    "Backtracking when formula is UNSAT"

    if not formula:
        return assignment

    if DEBUG:
        print('###### Backtracking')
        print(f'- `formula={formula}`')
        print(f'- `assignment={assignment}`')

    formula, pure_assignment = pure_literal(formula)
    if DEBUG:
        print(f'- Pure literals propagation: `pure_assignment={pure_assignment}` => `formula={formula}`')

    formula, unit_assignment = unit_propagation(formula)
    if DEBUG:
        print(f'- Units propagation: `unit_assignment={unit_assignment}` => `formula={formula}`')

    assignment = assignment + pure_assignment + unit_assignment

    if formula == -1:
        return []
    if not formula:
        return assignment

    if len(get_counter(formula)) > 0:
        variable = variable_selection(formula)
        if DEBUG:
            print(f'- Variable selection: `variable={variable}`')

        f = bcp(formula, variable)
        if DEBUG:
            print(f'- Boolean constraint propagation with `variable={variable}`: `formula={f}`')
        solution = backtracking(f, assignment + [variable])
        if not solution:
            f = bcp(formula, -variable)
            if DEBUG:
                print(f'- Boolean constraint propagation with `variable={-variable}`: `formula={f}`')
            solution = backtracking(f, assignment + [-variable])
        return solution
    else: 
        return assignment

def solve(formula, nvars):
    solution = backtracking(formula, [])
    print(solution)
    if solution:
        solution += [x for x in range(1, nvars + 1) 
                        if x not in solution 
                        and -x not in solution]
        solution.sort(key=lambda x: abs(x))
        print('s SATISFIABLE')
        print('v ' + ' '.join([str(x) for x in solution]) + ' 0')
    else:
        print('s UNSATISFIABLE')
