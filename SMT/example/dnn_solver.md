# Deep Neural Network Solver

### Overview

1. Architecture
    
    **Custom** `DPLL` <===> `Linear` solver (`Simplex` (phase 1) <=> **Standard** `DPLL`)

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
    'in': '(and (x0 < 1) (x1 > 2))',
    'out': '(y0 > y1)'
}
```

### Solving

##### Propagate

- Theory propagate

    - Check T-SAT: `SAT`
    - Constraints: `(and (and (and (and (and (and (and (and (x0 < 1) (x1 > 2)) (a00 = 1x0 - 1x1)) (a01 = 1x0 + 1x1)) (a10 = 0.5n00 - 0.2n01)) (a11 = -0.5n00 + 0.1n01)) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (not (a00 <= 0)))`
    - Deduction: `a00 <= 0`
    - New assignment: `[-1]`

##### Assign

- Assign `variable=1`, `value=False`
- Unassigned variables = `{2, 3, 4}`


- Theory propagate

    - Check T-SAT: `SAT`
    - New assignment: `[]`

##### Check SAT

- Unassigned variables = `{2, 3, 4}` => `None`

##### Decide

- Choose: variable=`2`

##### Assign

- Assign `variable=2`, `value=True`
- Unassigned variables = `{3, 4}`

##### Propagate

- Theory propagate

    - Check T-SAT: `SAT`
    - Constraints: `(and (and (and (and (and (and (and (and (and (and (and (and (x0 < 1) (x1 > 2)) (a00 = 1x0 - 1x1)) (n00 = 0)) (a01 = 1x0 + 1x1)) (n01 = 1x0 + 1x1)) (a10 = 0.5n00 - 0.2n01)) (a11 = -0.5n00 + 0.1n01)) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (a00 <= 0)) (a01 > 0)) (not (a10 <= 0)))`
    - Deduction: `a10 <= 0`
    - Constraints: `(and (and (and (and (and (and (and (and (and (and (and (and (x0 < 1) (x1 > 2)) (a00 = 1x0 - 1x1)) (n00 = 0)) (a01 = 1x0 + 1x1)) (n01 = 1x0 + 1x1)) (a10 = 0.5n00 - 0.2n01)) (a11 = -0.5n00 + 0.1n01)) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (a00 <= 0)) (a01 > 0)) (not (a11 > 0)))`
    - Deduction: `a11 > 0`
    - New assignment: `[-3, 4]`

##### Assign

- Assign `variable=3`, `value=False`
- Unassigned variables = `{4}`

##### Assign

- Assign `variable=4`, `value=True`
- Unassigned variables = `set()`

- Theory propagate

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
- Unassigned variables = `{3, 4}`

- Theory propagate

    - Check T-SAT: `SAT`
    - Constraints: `(and (and (and (and (and (and (and (and (and (and (and (and (x0 < 1) (x1 > 2)) (a00 = 1x0 - 1x1)) (n00 = 0)) (a01 = 1x0 + 1x1)) (n01 = 0)) (a10 = 0.5n00 - 0.2n01)) (a11 = -0.5n00 + 0.1n01)) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (a00 <= 0)) (a01 <= 0)) (not (a10 <= 0)))`
    - Deduction: `a10 <= 0`
    - Constraints: `(and (and (and (and (and (and (and (and (and (and (and (and (x0 < 1) (x1 > 2)) (a00 = 1x0 - 1x1)) (n00 = 0)) (a01 = 1x0 + 1x1)) (n01 = 0)) (a10 = 0.5n00 - 0.2n01)) (a11 = -0.5n00 + 0.1n01)) (y0 = 1n10 - 1n11)) (y1 = -1n10 + 1n11)) (a00 <= 0)) (a01 <= 0)) (not (a11 <= 0)))`
    - Deduction: `a11 <= 0`
    - New assignment: `[-3, -4]`

##### Assign

- Assign `variable=3`, `value=False`
- Unassigned variables = `{4}`

##### Assign

- Assign `variable=4`, `value=False`
- Unassigned variables = `set()`

- Theory propagate

    - Check T-SAT: `UNSAT`
    - Conflict clause: `[1, 2, 3, 4]`

##### Analyze

- Backtrack `level=-1`

### Result

- `UNSAT`




### TODO

- [x] Try with larger DNN => Simplex Issue: Singular Matrix
- [x] Input interface
- [x] Adopt substitution + Optimize constraints in Theory propagate.
- [x] Optimize `Simplex`
- [ ] Finding Unsat core when unsat occurred (Actually don't need that)
- [ ] Compare with Reluplex
- [ ] Behavior SAT/Unsat (time consuming)


### Dummy Ideas

- [ ] Decide multiple variables in a same layer at once? If UNSAT then backtrack + add conflict clause
- [ ] Strategy for choosing variables?
- [ ] What if some chosen variables can be used to imply the others? Choosing wrong value leads to UNSAT => conflict clause => next time the algorithm should decide smarter?
