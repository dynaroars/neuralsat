### TODO

- [x] Try with larger DNN => `Simplex` Issue: Singular Matrix
- [x] Input interface
- [x] Adopt substitution + Optimize constraints in Theory propagate.
- [x] Optimize `Simplex` (copy from scipy library)
- [x] Compare with Reluplex (relation among input dimensions)
- [ ] Find `Unsat Core` when unsat occurred (Actually don't need that???)
- [ ] Evaluate behavior `SAT`/`Unsat` (time consuming)
- [ ] Use cache mechanism in constraint generation
- [ ] Verify the correctness of the solver
- [ ] Find some benchmarks (comparing with `Reluplex`)

### Dummy Ideas

- [x] Decide multiple variables in a same layer at once
- [x] Strategy for choosing variables (from low to high id)
- [x] Strategy for assigning variables (randomly True/False)
- [ ] Think of better strategy for choosing/assigning variables
