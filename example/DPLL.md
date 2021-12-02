# The Davis–Putnam–Logemann–Loveland Algorithm

## Example

##### Input: DIMACS CNF format
`num_vars=5`, `num_clauses=11` and `clause_length=5`

```
c
c Random CNF formula
c
p cnf 5 11
3 1 5 -4 0
-1 -5 0
2 0
-4 -5 -2 0
3 -1 -4 2 0
2 -3 1 -4 -5 0
-4 2 1 3 0
-5 3 0
5 -3 -2 0
5 3 2 0
1 -3 -2 0

```

##### Solving

###### Backtracking level 0

- `formula=[[3, 1, 5, -4], [-1, -5], [2], [-4, -5, -2], [3, -1, -4, 2], [2, -3, 1, -4, -5], [-4, 2, 1, 3], [-5, 3], [5, -3, -2], [5, 3, 2], [1, -3, -2]]`
- `assignment=[]`
- Pure literals propagation: `pure_assignment=[-4]` => `formula=[[-1, -5], [2], [-5, 3], [5, -3, -2], [5, 3, 2], [1, -3, -2]]`
- Units propagation: `unit_assignment=[2]` => `formula=[[-1, -5], [-5, 3], [5, -3], [1, -3]]`
- Variable selection: `variable=5`
- Boolean constraint propagation with `variable=5`: `formula=[[-1], [3], [1, -3]]`

###### Backtracking level 1
- `formula=[[-1], [3], [1, -3]]`
- `assignment=[-4, 2, 5]`
- Pure literals propagation: `pure_assignment=[]` => `formula=[[-1], [3], [1, -3]]`
- Units propagation: `unit_assignment=[]` => `formula=-1`
- Boolean constraint propagation with `variable=-5`: `formula=[[-3], [1, -3]]`

###### Backtracking level 2

- `formula=[[-3], [1, -3]]`
- `assignment=[-4, 2, -5]`
- Pure literals propagation: `pure_assignment=[-3, 1]` => `formula=[]`
- Units propagation: `unit_assignment=[]` => `formula=[]`

##### Result
```
s SATISFIABLE
v 1 2 -3 -4 -5 0
```