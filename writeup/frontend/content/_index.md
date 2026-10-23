+++
title = 'Leaderboard'
description = 'Which abstraction and search verify the most instances of a robustness experiment.'
+++

Which **abstraction + search** pair proves the most? Five abstractions (IBP,
zonotope, CROWN, α-CROWN and the constrained NeuralSAT abstraction) each run
with two searches (branch and bound, DPLL(T)) on the same instances, one
after the other.

{{< leaderboard >}}

**How to read it.**

- The **score is the number of solved instances**: `unsat` (the property is
  proven) or `sat` (a real counterexample was found) within the time limit.
  Ties go to the lower **PAR-2**, the average runtime where an unsolved
  instance counts as twice the limit.
- `unknown` is everything else: the budget ran out, or no neuron was left to
  split and the bounds were still not enough.
- Only the **NeuralSAT abstraction cuts the region with the constraints** of
  the instance (a halfspace and an $\ell_2$ ball). The others bound the whole
  $\ell_\infty$ ball and only use the constraints to discard counterexamples, so a
  win for NeuralSAT measures what the constraints buy, not tighter bounds on
  the same region.
- Rows that differ only in the search show what splitting and learning add on
  top of the same bounds.

{{< cactus >}}
Cactus plot: solved instances (x) against cumulative runtime (y); a curve that
reaches further right and stays lower is better. Instances solved by every
configuration are left out.
{{< /cactus >}}

The instances, the scoring and the checks of the code are on the
[Results](results/) page. New here? Start with the [walkthrough](walkthrough/),
or read the [About](about/) page for an overview of the code.
