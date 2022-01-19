# Conflict-Driven Clause Learning (CDCL) Solver

## Example 1

##### Input: DIMACS CNF format
`num_vars=10`, `num_clauses=10` and `clause_length=5`

```
c Random CNF formula
p cnf 10 10
-3 10 4 -9 -5 0
-1 9 0
-9 -1 -6 4 0
-9 6 2 8 0
-9 -3 -1 0
-1 -8 0
3 1 -4 -7 10 0
2 7 -3 0
-9 2 10 -3 0
-9 -1 0

```

##### Solving


###### Read problem

- `num_clauses=10`, `num_vars=10`
- literals watching clause:
    - clause 0 => literals: [13, 10]
    - clause 1 => literals: [11, 9]
    - clause 2 => literals: [19, 11]
    - clause 3 => literals: [19, 6]
    - clause 4 => literals: [19, 13]
    - clause 5 => literals: [11, 18]
    - clause 6 => literals: [3, 1]
    - clause 7 => literals: [2, 7]
    - clause 8 => literals: [19, 2]
    - clause 9 => literals: [19, 11]
- clauses watched by literal:
    - literal 13 => clauses: [0, 4]
    - literal 10 => clauses: [0]
    - literal 11 => clauses: [1, 2, 5, 9]
    - literal 9 => clauses: [1]
    - literal 19 => clauses: [2, 3, 4, 8, 9]
    - literal 6 => clauses: [3]
    - literal 18 => clauses: [5]
    - literal 3 => clauses: [6]
    - literal 1 => clauses: [6]
    - literal 2 => clauses: [7, 8]
    - literal 7 => clauses: [7]

###### BCP `level=0`

- `assignment_stack=[]`
- return `NO_CONFLICT`

###### Decide `level=0`

- `var=1`, `value=True`

###### BCP `level=1`

- `assignment_stack=[(1, True)]`
- `last_assigned_var=1`, `negation_literal=11`
- clause watched by negation literal: [1, 2, 5, 9]
- `clause=1`, `clause=[11, 9]`
    - `other_literal=9`, `other_var=9`
    - `new literal=-1` => implications value of `other_var=9`
    - `assignment_stack=[(1, True)]`
    - `other_var=9`, `value=True`
- `clause=2`, `clause=[19, 11, 16, 4]`
    - `other_literal=19`, `other_var=9`
    - `clause=2` => 2-watched literals = [19, 11]
    - `new literal=16`
    - `clause=2` => 2-watched literals = [19, 16]
- `clause=5`, `clause=[11, 18]`
    - `other_literal=18`, `other_var=8`
    - `new literal=-1` => implications value of `other_var=8`
    - `assignment_stack=[(1, True), (9, True)]`
    - `other_var=8`, `value=False`
- `clause=9`, `clause=[19, 11]`
    - `other_literal=19`, `other_var=9`
    - `new literal=-1` => implications value of `other_var=9`
    - `assignment_stack=[(1, True), (9, True), (8, False)]`
    - Conflict: `level=1`, `clause=9`
- return `CONFLICT`

###### Analyze-Conflict `level=1`

- `conflict_level=1`, `conflict_clause=9`: [19, 11]
    - prev_assigned_node: `var=9`, `clause=[11, 9]`
    - binary resolution => `conflict_clause=[11]`
- Implication: `var=1`, `value=False`
- Backtrack: `backtrack_level=0`

###### Backtrack `level=1` to `level=0`

- `assignment_stack=[]`
- Add node: `var=1`, `value=False`

###### BCP `level=0`

- `assignment_stack=[(1, False)]`
- `last_assigned_var=1`, `negation_literal=1`
- clause watched by negation literal: [6]
- `clause=6`, `clause=[3, 1, 14, 17, 10]`
    - `other_literal=3`, `other_var=3`
    - `clause=6` => 2-watched literals = [3, 1]
    - `new literal=14`
    - `clause=6` => 2-watched literals = [3, 14]
- return `NO_CONFLICT`

###### Decide `level=0`

- `var=2`, `value=True`

###### BCP `level=1`

- `assignment_stack=[(1, False), (2, True)]`
- `last_assigned_var=2`, `negation_literal=12`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=1`

- `var=3`, `value=True`

###### BCP `level=2`

- `assignment_stack=[(1, False), (2, True), (3, True)]`
- `last_assigned_var=3`, `negation_literal=13`
- clause watched by negation literal: [0, 4]
- `clause=0`, `clause=[13, 10, 4, 19, 15]`
    - `other_literal=10`, `other_var=10`
    - `clause=0` => 2-watched literals = [13, 10]
    - `new literal=4`
    - `clause=0` => 2-watched literals = [10, 4]
- `clause=4`, `clause=[19, 13, 11]`
    - `other_literal=19`, `other_var=9`
    - `clause=4` => 2-watched literals = [19, 13]
    - `new literal=11`
    - `clause=4` => 2-watched literals = [19, 11]
- return `NO_CONFLICT`

###### Decide `level=2`

- `var=4`, `value=True`

###### BCP `level=3`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True)]`
- `last_assigned_var=4`, `negation_literal=14`
- clause watched by negation literal: [6]
- `clause=6`, `clause=[3, 1, 14, 17, 10]`
    - `other_literal=3`, `other_var=3`
- return `NO_CONFLICT`

###### Decide `level=3`

- `var=5`, `value=True`

###### BCP `level=4`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True)]`
- `last_assigned_var=5`, `negation_literal=15`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=4`

- `var=6`, `value=True`

###### BCP `level=5`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True), (6, True)]`
- `last_assigned_var=6`, `negation_literal=16`
- clause watched by negation literal: [2]
- `clause=2`, `clause=[19, 11, 16, 4]`
    - `other_literal=19`, `other_var=9`
    - `clause=2` => 2-watched literals = [19, 16]
    - `new literal=11`
    - `clause=2` => 2-watched literals = [19, 11]
- return `NO_CONFLICT`

###### Decide `level=5`

- `var=7`, `value=True`

###### BCP `level=6`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True)]`
- `last_assigned_var=7`, `negation_literal=17`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=6`

- `var=8`, `value=True`

###### BCP `level=7`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True), (8, True)]`
- `last_assigned_var=8`, `negation_literal=18`
- clause watched by negation literal: [5]
- `clause=5`, `clause=[11, 18]`
    - `other_literal=11`, `other_var=1`
- return `NO_CONFLICT`

###### Decide `level=7`

- `var=9`, `value=True`

###### BCP `level=8`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True), (8, True), (9, True)]`
- `last_assigned_var=9`, `negation_literal=19`
- clause watched by negation literal: [2, 3, 4, 8, 9]
- `clause=2`, `clause=[19, 11, 16, 4]`
    - `other_literal=11`, `other_var=1`
- `clause=3`, `clause=[19, 6, 2, 8]`
    - `other_literal=6`, `other_var=6`
- `clause=4`, `clause=[19, 13, 11]`
    - `other_literal=11`, `other_var=1`
- `clause=8`, `clause=[19, 2, 10, 13]`
    - `other_literal=2`, `other_var=2`
- `clause=9`, `clause=[19, 11]`
    - `other_literal=11`, `other_var=1`
- return `NO_CONFLICT`

###### Decide `level=8`

- `var=10`, `value=True`

###### BCP `level=9`

- `assignment_stack=[(1, False), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True), (8, True), (9, True), (10, True)]`
- `last_assigned_var=10`, `negation_literal=20`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=9`

- `var=-1`

##### Result
```
s SATISFIABLE
v -1 2 3 4 5 6 7 8 9 10 0
```




## Example 2

##### Input: DIMACS CNF format
`num_vars=10`, `num_clauses=10` and `clause_length=5`

```
c Random CNF formula
p cnf 10 10
-3 -1 -10 0
7 -6 -2 -4 0
-3 -2 10 0
5 -4 7 -3 0
5 -4 -6 0
8 3 -10 2 -9 0
7 9 -8 -4 0
-2 -6 -4 -3 -1 0
3 -1 -2 -4 -8 0
-5 -7 8 0

```

##### Solving




###### Read problem

- `num_clauses=10`, `num_vars=10`
- literals watching clause:
    - clause 0 => literals: [13, 11]
    - clause 1 => literals: [7, 16]
    - clause 2 => literals: [13, 12]
    - clause 3 => literals: [5, 14]
    - clause 4 => literals: [5, 14]
    - clause 5 => literals: [8, 3]
    - clause 6 => literals: [7, 9]
    - clause 7 => literals: [12, 16]
    - clause 8 => literals: [3, 11]
    - clause 9 => literals: [15, 17]
- clauses watched by literal:
    - literal 13 => clauses: [0, 2]
    - literal 11 => clauses: [0, 8]
    - literal 7 => clauses: [1, 6]
    - literal 16 => clauses: [1, 7]
    - literal 12 => clauses: [2, 7]
    - literal 5 => clauses: [3, 4]
    - literal 14 => clauses: [3, 4]
    - literal 8 => clauses: [5]
    - literal 3 => clauses: [5, 8]
    - literal 9 => clauses: [6]
    - literal 15 => clauses: [9]
    - literal 17 => clauses: [9]

###### BCP `level=0`

- `assignment_stack=[]`
- return `NO_CONFLICT`

###### Decide `level=1`

- `var=1`, `value=True`

###### BCP `level=1`

- `assignment_stack=[(1, True, 1)]`
- `last_assigned_var=1`, `negation_literal=11`
- clause watched by negation literal: [0, 8]
- `clause=0`, `clause=[13, 11, 20]`
    - `other_literal=13`, `other_var=3`
    - `clause=0` => 2-watched literals = [13, 11]
    - `new literal=20`
    - `clause=0` => 2-watched literals = [13, 20]
- `clause=8`, `clause=[3, 11, 12, 14, 18]`
    - `other_literal=3`, `other_var=3`
    - `clause=8` => 2-watched literals = [3, 11]
    - `new literal=12`
    - `clause=8` => 2-watched literals = [3, 12]
- return `NO_CONFLICT`

###### Decide `level=2`

- `var=2`, `value=True`

###### BCP `level=2`

- `assignment_stack=[(1, True, 1), (2, True, 2)]`
- `last_assigned_var=2`, `negation_literal=12`
- clause watched by negation literal: [2, 7, 8]
- `clause=2`, `clause=[13, 12, 10]`
    - `other_literal=13`, `other_var=3`
    - `clause=2` => 2-watched literals = [13, 12]
    - `new literal=10`
    - `clause=2` => 2-watched literals = [13, 10]
- `clause=7`, `clause=[12, 16, 14, 13, 11]`
    - `other_literal=16`, `other_var=6`
    - `clause=7` => 2-watched literals = [12, 16]
    - `new literal=14`
    - `clause=7` => 2-watched literals = [16, 14]
- `clause=8`, `clause=[3, 11, 12, 14, 18]`
    - `other_literal=3`, `other_var=3`
    - `clause=8` => 2-watched literals = [3, 12]
    - `new literal=14`
    - `clause=8` => 2-watched literals = [3, 14]
- return `NO_CONFLICT`

###### Decide `level=3`

- `var=3`, `value=True`

###### BCP `level=3`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, True, 3)]`
- `last_assigned_var=3`, `negation_literal=13`
- clause watched by negation literal: [0, 2]
- `clause=0`, `clause=[13, 11, 20]`
    - `other_literal=20`, `other_var=10`
    - `new literal=-1` => implications value of `other_var=10`
    - `assignment_stack=[(1, True, 1), (2, True, 2), (3, True, 3)]`
    - `other_var=10`, `value=False`
- `clause=2`, `clause=[13, 12, 10]`
    - `other_literal=10`, `other_var=10`
    - `new literal=-1` => implications value of `other_var=10`
    - `assignment_stack=[(1, True, 1), (2, True, 2), (3, True, 3), (10, False, 3)]`
    - Conflict: `level=3`, `clause=2`
- return `CONFLICT`

###### Analyze-Conflict `level=3`

- `conflict_level=3`, `conflict_clause_id=2`: `conflict_clause=[13, 12, 10]`
    - prev_assigned_node: `var=10`, `clause=[13, 11, 20]`
    - binary resolution => `conflict_clause=[13, 12, 11]`
- Learned clause: `clause=[13, 12, 11]`
- Implication: `var=3`, `value=False`
- Backtrack: `backtrack_level=2`

###### Backtrack `level=3` to `level=2`

- `assignment_stack=[(1, True, 1), (2, True, 2)]`
- Add node: `var=3`, `value=False`

###### BCP `level=2`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2)]`
- `last_assigned_var=3`, `negation_literal=3`
- clause watched by negation literal: [5, 8]
- `clause=5`, `clause=[8, 3, 20, 2, 19]`
    - `other_literal=8`, `other_var=8`
    - `clause=5` => 2-watched literals = [8, 3]
    - `new literal=20`
    - `clause=5` => 2-watched literals = [8, 20]
- `clause=8`, `clause=[3, 11, 12, 14, 18]`
    - `other_literal=14`, `other_var=4`
    - `clause=8` => 2-watched literals = [3, 14]
    - `new literal=18`
    - `clause=8` => 2-watched literals = [14, 18]
- return `NO_CONFLICT`

###### Decide `level=3`

- `var=4`, `value=True`

###### BCP `level=3`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3)]`
- `last_assigned_var=4`, `negation_literal=14`
- clause watched by negation literal: [3, 4, 7, 8]
- `clause=3`, `clause=[5, 14, 7, 13]`
    - `other_literal=5`, `other_var=5`
    - `clause=3` => 2-watched literals = [5, 14]
    - `new literal=7`
    - `clause=3` => 2-watched literals = [5, 7]
- `clause=4`, `clause=[5, 14, 16]`
    - `other_literal=5`, `other_var=5`
    - `clause=4` => 2-watched literals = [5, 14]
    - `new literal=16`
    - `clause=4` => 2-watched literals = [5, 16]
- `clause=7`, `clause=[12, 16, 14, 13, 11]`
    - `other_literal=16`, `other_var=6`
    - `clause=7` => 2-watched literals = [16, 14]
    - `new literal=13`
    - `clause=7` => 2-watched literals = [16, 13]
- `clause=8`, `clause=[3, 11, 12, 14, 18]`
    - `other_literal=18`, `other_var=8`
    - `new literal=-1` => implications value of `other_var=8`
    - `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3)]`
    - `other_var=8`, `value=False`
- `last_assigned_var=8`, `negation_literal=8`
- clause watched by negation literal: [5]
- `clause=5`, `clause=[8, 3, 20, 2, 19]`
    - `other_literal=20`, `other_var=10`
    - `clause=5` => 2-watched literals = [8, 20]
    - `new literal=2`
    - `clause=5` => 2-watched literals = [20, 2]
- return `NO_CONFLICT`

###### Decide `level=4`

- `var=5`, `value=True`

###### BCP `level=4`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3), (8, False, 3), (5, True, 4)]`
- `last_assigned_var=5`, `negation_literal=15`
- clause watched by negation literal: [9]
- `clause=9`, `clause=[15, 17, 8]`
    - `other_literal=17`, `other_var=7`
    - `new literal=-1` => implications value of `other_var=7`
    - `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3), (8, False, 3), (5, True, 4)]`
    - `other_var=7`, `value=False`
- `last_assigned_var=7`, `negation_literal=7`
- clause watched by negation literal: [1, 6, 3]
- `clause=1`, `clause=[7, 16, 12, 14]`
    - `other_literal=16`, `other_var=6`
    - `new literal=-1` => implications value of `other_var=6`
    - `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3), (8, False, 3), (5, True, 4), (7, False, 4)]`
    - `other_var=6`, `value=False`
- `clause=6`, `clause=[7, 9, 18, 14]`
    - `other_literal=9`, `other_var=9`
    - `clause=6` => 2-watched literals = [7, 9]
    - `new literal=18`
    - `clause=6` => 2-watched literals = [9, 18]
- `clause=3`, `clause=[5, 14, 7, 13]`
    - `other_literal=5`, `other_var=5`
- `last_assigned_var=6`, `negation_literal=6`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=5`

- `var=9`, `value=True`

###### BCP `level=5`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3), (8, False, 3), (5, True, 4), (7, False, 4), (6, False, 4), (9, True, 5)]`
- `last_assigned_var=9`, `negation_literal=19`
- clause watched by negation literal: []
- return `NO_CONFLICT`

###### Decide `level=6`

- `var=10`, `value=True`

###### BCP `level=6`

- `assignment_stack=[(1, True, 1), (2, True, 2), (3, False, 2), (4, True, 3), (8, False, 3), (5, True, 4), (7, False, 4), (6, False, 4), (9, True, 5), (10, True, 6)]`
- `last_assigned_var=10`, `negation_literal=20`
- clause watched by negation literal: [0, 5]
- `clause=0`, `clause=[13, 11, 20]`
    - `other_literal=13`, `other_var=3`
- `clause=5`, `clause=[8, 3, 20, 2, 19]`
    - `other_literal=2`, `other_var=2`
- return `NO_CONFLICT`

###### Decide `level=7`

- `var=-1`

##### Result
```
s SATISFIABLE
v 1 2 -3 4 -8 5 -7 -6 9 10 0
```