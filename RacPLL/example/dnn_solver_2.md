# Deep Neural Network Solver

### Overview

1. Architecture
    
    **custom** `DPLL` <===> `Linear` solver (`Simplex` solver <=> **standard** `DPLL`)

2. Custom DPLL:
    - Start with `formula=[]`, set of unassigned variables `vars={1, 2, ...}`
    - Stop when **all variables** are assigned

3. Procedure

    - Loop:
        - Propagate (DPLL + Theory)
        - Check SAT
        - Decide

### Example

##### Input

- DNN input format
```
dnn = {
    'a00': [1, '1x0 - 1x1'],
    'a01': [2, '1x0 + 1x1'],
    'a10': [3, '0.5n00 - 0.2n01'],
    'a11': [4, '-0.5n00 + 0.1n01'],
    'y0': '1n10 - 1n11',
    'y1': '-1n10 + 1n11',
}
```
- Conditions format
```
conditions = {
    'in': '(and (x0 < 0) (x1 > 1))',
    'out': '(y0 > y1)'
}
```

### Solving

##### Propagate

- Theory propagate

- Assignment: {}

- Theory constraints: `(and (x0 < 0) (x1 > 1))`
    - Check T-SAT: `SAT`

- Deduction
    - Deduction: `a0_0 <= 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (x0 + -1*x1 <= 0)))`
        - Result: True
    - Deduction: `a0_1 <= 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (x0 + x1 <= 0)))`
        - Result: False
    - Deduction: `a0_1 > 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (x0 + x1 > 0)))`
        - Result: False
    - New assignment: `[-1]`

##### Assign

- Assign `variable=1`, `value=False`
- Unassigned variables = `SortedList([2, 3, 4])`

- Theory propagate

- Assignment: {'a0_0': False}

- Theory constraints: `(and (x0 < 0) (x1 > 1))`
    - Check T-SAT: `SAT`

- Deduction
    - Deduction: `a0_1 <= 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (x0 + x1 <= 0)))`
        - Result: False
    - Deduction: `a0_1 > 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (x0 + x1 > 0)))`
        - Result: False
    - New assignment: `[]`

##### Check SAT

- Unassigned variables = `SortedList([2, 3, 4])` => `None`

##### Decide

- Choose: variable=`2`

##### Assign

- Assign `variable=2`, `value=True`
- Unassigned variables = `SortedList([3, 4])`

##### Propagate

- Theory propagate

- Assignment: {'a0_0': False, 'a0_1': True}

- Theory constraints: `(and (and (x0 < 0) (x1 > 1)) (x0 + x1 > 0))`
    - Check T-SAT: `SAT`

- Deduction
    - Deduction: `a1_0 <= 0`
    - Constraints: `(and (and (and (x0 < 0) (x1 > 1)) (x0 + x1 > 0)) (not (-0.2*x0 + -0.2*x1 <= 0)))`
        - Result: True
    - Deduction: `a1_1 <= 0`
    - Constraints: `(and (and (and (x0 < 0) (x1 > 1)) (x0 + x1 > 0)) (not (0.1*x0 + 0.1*x1 <= 0)))`
        - Result: False
    - Deduction: `a1_1 > 0`
    - Constraints: `(and (and (and (x0 < 0) (x1 > 1)) (x0 + x1 > 0)) (not (0.1*x0 + 0.1*x1 > 0)))`
        - Result: True
    - New assignment: `[-3, 4]`

##### Assign

- Assign `variable=3`, `value=False`
- Unassigned variables = `SortedList([4])`

##### Assign

- Assign `variable=4`, `value=True`
- Unassigned variables = `SortedList([])`

- Theory propagate

- Assignment: {'a0_0': False, 'a0_1': True, 'a1_0': False, 'a1_1': True}

- Theory constraints: `(and (and (and (and (x0 < 0) (x1 > 1)) (x0 + x1 > 0)) (0.1*x0 + 0.1*x1 > 0)) (-0.1*x0 + -0.1*x1 > 0.1*x0 + 0.1*x1))`
    - Check T-SAT: `UNSAT`
    - Conflict clause: `[1, 3, -4, -2]`

##### Analyze

- Backtrack `level=0`

##### Backtrack

- Backtrack to `level=0`
- Unassign `variable=2`
- Unassign `variable=3`
- Unassign `variable=4`
- Add conflict clause: `[1, 3, -4, -2]`

##### Assign

- Assign `variable=2`, `value=False`
- Unassigned variables = `SortedList([3, 4])`

- Theory propagate

- Assignment: {'a0_0': False, 'a0_1': False}

- Theory constraints: `(and (x0 < 0) (x1 > 1))`
    - Check T-SAT: `SAT`

- Deduction
    - Deduction: `a1_0 <= 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (0 <= 0)))`
        - Result: True
    - Deduction: `a1_1 <= 0`
    - Constraints: `(and (and (x0 < 0) (x1 > 1)) (not (0 <= 0)))`
        - Result: True
    - New assignment: `[-3, -4]`

##### Assign

- Assign `variable=3`, `value=False`
- Unassigned variables = `SortedList([4])`

##### Assign

- Assign `variable=4`, `value=False`
- Unassigned variables = `SortedList([])`

- Theory propagate

- Assignment: {'a0_0': False, 'a0_1': False, 'a1_0': False, 'a1_1': False}

- Theory constraints: `(and (and (x0 < 0) (x1 > 1)) (0 > 0))`
    - Check T-SAT: `UNSAT`
    - Conflict clause: `[1, 2, 3, 4]`

##### Analyze

- Backtrack `level=-1`

### Result

- `UNSAT`