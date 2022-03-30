### TODO

- [x] Try with larger DNN => `Simplex` Issue: Singular Matrix
- [x] Input interface (`keras` model)
- [x] Adopt substitution + optimize constraints in theory propagate.
- [x] Optimize `Simplex` (copy from `scipy` library)
- [x] Compare with Reluplex (relation among input dimensions)
- [ ] Evaluate behavior `SAT`/`UNSAT` (time consuming)
- [ ] Use cache mechanism in constraint generation
- [ ] Verify the correctness of the solver
- [ ] Find some benchmarks (comparing with `Reluplex`)

- [ ] Run Corina's example (with bias) with each methods 

### Dummy Ideas

- [x] Decide multiple variables in a same layer at once
- [x] Strategy for choosing variables (from `low` to `high` id)
- [x] Strategy for assigning variables (randomly `True`/`False`)
- [ ] Think of better strategy for choosing/assigning variables
- [ ] Abstract Interpretation
- [ ] Connected Component

- related works:
    - find adversarial example => need specific input to run
    - counterexample-guided search (promising) 120
    - MILP 149


engineer:
    - parallel:
        + devide input space 

dummy:
    - optimize:
        + MILP for layer decision (requires lower and upper bounds)
        + heuristic

    - over-approximate + prune search space:
        + abstract interpretation: ERAN
        + symbolic interval: check boundaries on each node
        + reachability