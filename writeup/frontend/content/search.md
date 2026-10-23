+++
title = 'Search'
toc = true
description = 'What branch and bound and DPLL(T) do, step by step, on a recorded run.'
+++

Bounding alone often fails: the bounds are sound but loose, and the margin
lower bound stays below 0 although the property is true. A **search** splits
the problem into smaller regions where the bounds are tighter. Both searches
here split on the phase of a ReLU or LeakyReLU neuron, and both treat the
abstraction (IBP, zonotope, CROWN, α-CROWN or the NeuralSAT abstraction) as a
black box that bounds a region. They differ in how they
organise the regions and what they remember.

## Shared ingredients

**A region is a phase assignment.** An assignment such as
`{+r1.4, -r1.2}` says "neuron 4 of ReLU layer 1 is active and neuron 2 is
inactive". The abstraction bounds the network under those splits (see
[phase splits](../bounding/#phase-splits)). The assignment `{}` is the whole
region.

**A region is proven** when the lower bound of **every** margin is above 0, or
when some neuron interval is empty (the splits contradict each other). A
proven region has no counterexample.

**Falsify first.** Before any splitting, a PGD attack looks for an input in
the region that changes the prediction. A real counterexample ends the search
with `sat`; the status `sat` is **never** derived from bounds, only from a
concrete input that was checked, and that satisfies the constraints.

**Which neuron to split.** `pick_neuron` takes every unstable ReLU/LeakyReLU
neuron ($l < 0 < u$) and chooses the one with the largest triangle area
$\tfrac{-l\,u}{u - l}$: the neuron whose relaxation loses the most.

**How a search ends.** `unsat` when every region is proven. `unknown` when
the budget is spent, or when a region has no unstable neuron left to split but
is still not proven (the splits only clamp bounds and do not enforce the
split constraint on the symbolic bound, so a fully split region can stay
loose; see [Bounding](../bounding/#phase-splits)).

## Branch and bound

BaB keeps a queue of regions. It takes one, bounds it, throws it away if it is
proven, and otherwise splits it in two.

```text
bab(model, x0, eps, label, bound_fn, max_nodes):
    if falsify finds x: return sat, x
    queue = [ {} ]                                   # the whole region
    while queue is not empty and nodes < max_nodes:
        assignment = queue.pop_front()
        bounds = bound_fn(splits = assignment)       # one call to the abstraction
        if proven(bounds): continue                  # nothing left to do here
        neuron = pick_neuron(bounds)
        if no neuron: stuck = true; continue
        queue.push( assignment + {neuron: active} )
        queue.push( assignment + {neuron: inactive} )
    return unknown if queue is not empty or stuck else unsat
```

The two children cover the parent: every input has the neuron either
non-negative or non-positive. Search is breadth first. The cost is the number
of calls to the abstraction, one per region.

### A recorded run

A real run of the code on a small instance: a `2-4-4-2` ReLU network,
$\varepsilon = {{< instance eps >}}$, with the CROWN abstraction. At the root
the margin lower bound is {{< instance root_margin >}}, below 0, so CROWN alone
cannot prove the property. Neuron names are `r<layer>.<neuron>`.

{{< bab-trace >}}

The root is not proven, so BaB splits `r1.4`. The active child is proven
immediately. The inactive child is still slightly negative, so it is split on
`r1.2`, and both grandchildren are proven. Five regions were bounded and the
status is `unsat`.

## DPLL(T)

DPLL(T) is the search of SAT solvers with a **theory** attached. The Boolean
variables are the neuron phases, and the theory solver is the abstraction. It
differs from BaB in what it remembers: when a region is proven it works out
**why** and learns a clause, so that other regions are not explored twice.

### Why DPLL(T)?

The reason to use DPLL(T) for verification is **restarts**. A solver often
restarts on the same instance, that is, the same network and the same
specification, for example with a different branching heuristic because the
first one made little progress. BaB has nothing to carry over: its only memory
is the queue of open regions, so after a restart it starts again from the root
and bounds regions it had already proven.

A learned clause is different. It says "the region where these phases hold has
no counterexample", which is a fact about the instance, not about the order in
which a particular run happened to visit regions. So when the solver restarts
with another branching heuristic, the **literals learned in the previous run,
under different branches, are kept and reused**. The new run decides different
neurons in a different order; but whenever its trail makes all literals of an
old clause but one true, unit propagation uses the old clause. Regions that
the earlier run proved are not bounded again, however the new run gets to
them. The restart is cheap, and the work of every earlier run counts.

This implementation runs one search per call and keeps its clauses only for
that call; it has no restart loop. The reuse is nevertheless built in: a clause
is just a list of literals that `propagate` reads, so handing the clauses of
one run to the next is all a restart needs.

**Vocabulary.**

- A *literal* is a phase, such as `+r1.4`. The *trail* is the list of literals
  assigned so far.
- A *decision* is a literal chosen by the search (active first). An *implied*
  literal is forced by a clause.
- A *clause* is an OR of literals, learned from a proof. The clause
  `-r1.2` says: "a counterexample must have neuron `r1.2` inactive".

```text
dpll(model, x0, eps, label, bound_fn, max_checks):
    if falsify finds x: return sat, x
    trail = []; clauses = []
    while checks < max_checks:
        trail, clash = propagate(trail, clauses)     # unit propagation
        if clash:                                    # a clause is false
            decisions = explain(trail, the false clause)
        else:
            bounds, effective = bound_fn(splits = trail, core = true)
            if proven(bounds):                       # a conflict, in SAT terms
                decisions = explain(trail, effective_literals(bounds, effective))
            else:
                neuron = pick_neuron(bounds)
                if neuron: trail.push(neuron: active, decision); continue
                stuck = true; decisions = all decisions on the trail
        if decisions is empty: return unknown if stuck else unsat
        clauses.add( not (all of decisions) )        # learn
        trail = backjump(trail, last decision in decisions)
    return unknown
```

**Propagate.** If all literals of a clause but one are false, the last one
must be true, so it is added to the trail with the clause as its *reason*.

**Explain.** A proven region says that the literals of the trail together make
the property true. A smaller set of them is enough. Two things shrink the
set, and neither costs another call to the abstraction:

- only the splits that **changed a bound** matter (`split_effect`). A split on
  a neuron that was already stable the same way is a no-op, and leaving it out
  gives bit-identical bounds;
- when the proof is an **empty region**, splits that only affect layers after
  the empty one are irrelevant.

An implied literal is replaced by the negations of the other literals of its
reason, until only decisions remain. The clause says "the region where these
decisions hold is proven", so it is valid for every later search.

**Backjump.** Remove the last decision of the clause and everything after it.
All other literals of the clause are still on the trail, so the clause is now
unit: `propagate` flips the removed decision. Because the clause only mentions
the decisions it needs, the jump can go back over several decisions at once,
skipping the regions in between.

### A recorded run

The same instance, as a sequence of events. A *theory check* is a call to the
abstraction (the cost); the other lines are bookkeeping.

{{< dpll-trace >}}

Reading it: the root is not proven, so it decides `+r1.4` (step 2) and that
region is proven (step 3). The learned clause is `-r1.4`, "`r1.4` must be
inactive"; backjump (5) removes the decision and the clause forces `-r1.4`
(6). That region is not proven, so it decides `+r1.2` (8), which is proven (9),
and learns `-r1.2`. After the backjump `-r1.2` is forced (12) and the region
with both neurons inactive is proven (13). Its explanation needs **no**
decision, so the learned clause is empty, and an empty clause means the whole
region is proven: `unsat`. Five theory checks, the same as BaB.

## BaB or DPLL(T)

| | BaB | DPLL(T) |
|---|---|---|
| order | breadth first queue | depth first, with backjumping |
| memory | the queue of open regions | the trail and the learned clauses |
| per region | one bound call | one bound call, plus propagation and explanation (no extra calls) |
| restarts | start from the root again | keep the clauses: what an earlier run proved, under other branches, is reused |
| saves work when | never revisits a region by construction | a clause is **shorter** than the decision path, so a conflict covers regions the plain tree would still visit, or the solver restarts |
| answers | `unsat` / `sat` / `unknown` | the same |

A learned clause saves work inside one run only when it is shorter than the
path of decisions that led to it; across restarts it saves work even when it
is not. On the benchmark of the [results](../results/)
every split is chosen on a neuron that still changes a bound, so the clauses
are as long as the decision path and the two searches make the same number of
bound calls. Inside one run DPLL(T) would pay off when a proof needs only a few splits out
of a deep path, for example on networks where much of the structure is
irrelevant to the margin. The benchmark here does not restart, so it does not
measure the reuse across restarts that motivates DPLL(T).
