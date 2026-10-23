'''DPLL(T): ReLU phases as Boolean variables, the abstraction as theory.

A literal ``((j, k), sign)`` says neuron ``k`` of ReLU layer ``j`` has phase
``sign`` (+1 active, -1 inactive). The trail is a list of
``(neuron, sign, reason)``: ``reason`` is None for a decision and the clause
that forced the literal otherwise. Loop:

1. ``propagate``: unit propagation over the learned clauses.
2. Theory check: bound the network under the trail. A proven subdomain is a
   conflict; otherwise ``decide`` an unstable neuron.
3. On a conflict, ``explain`` it: the proof only depends on the splits that
   changed a bound (reported by the abstraction with ``core=True``, and only
   up to the first empty layer when the proof is an empty subdomain). Their
   reasons are followed back to decisions. The clause negating those
   decisions is learned and ``backjump`` makes it unit, flipping the deepest
   one. Explaining costs no extra call to the abstraction.

A learned clause says ``the region where all these phases hold is proven``,
so a counterexample must violate one of its literals. An empty clause means
the whole region is proven.
'''

from abstraction.lirpa import crown
from search.common import (
    falsify,
    first_infeasible,
    pick_neuron,
    proven,
    split_tensors,
)


def assignment_of(trail):
    return {neuron: sign for neuron, sign, _ in trail}


def propagate(trail, clauses):
    '''Unit propagation. Returns ``(trail, clash)``.

    ``clash`` is the clause with every literal false, or None.
    '''
    trail = list(trail)
    assigned = assignment_of(trail)
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            if any(assigned.get(neuron) == sign for neuron, sign in clause):
                continue
            free = [(n, s) for n, s in clause if n not in assigned]
            if not free:
                return trail, clause
            if len(free) == 1:
                (neuron, sign), = free
                trail.append((neuron, sign, clause))
                assigned[neuron] = sign
                changed = True
    return trail, None


def effective_literals(layers, bounds, effective, assignment):
    '''Assigned literals that the proof of ``bounds`` depends on.

    These are the splits that changed a bound. If a layer is empty, the splits
    that clamp later layers are left out as well.
    '''
    last = first_infeasible(bounds)
    positions = {i: p for p, i in enumerate(crown.bounded_layers(layers))}
    return [(neuron, sign) for neuron, sign in assignment.items()
            if bool(effective[neuron[0]][neuron[1]])
            and (last is None or positions[neuron[0] - 1] <= last)]


def explain(trail, literals):
    '''Decisions behind trail ``literals``, in trail order.

    A decision explains itself; an implied literal is replaced by the
    negations of the other literals of the clause that forced it.
    '''
    reasons = {neuron: reason for neuron, _, reason in trail}
    decisions, seen, todo = set(), set(), list(literals)
    while todo:
        neuron, sign = todo.pop()
        if neuron in seen:
            continue
        seen.add(neuron)
        if reasons[neuron] is None:
            decisions.add((neuron, sign))
        else:
            todo.extend((n, -s) for n, s in reasons[neuron] if n != neuron)
    return [(n, s) for n, s, reason in trail
            if reason is None and (n, s) in decisions]


def backjump(trail, decision):
    # Remove the given decision and everything after it from the trail.
    index = next(i for i, (neuron, sign, reason) in enumerate(trail)
                 if reason is None and (neuron, sign) == decision)
    return trail[:index]


def dpll(model, x0, eps, label, bound_fn=crown.layer_bounds,
         constraints=(), max_checks=1000):
    '''Decide whether ``label`` is the argmax on the whole input region.

    Returns ``(status, witness)`` like ``bab.bab``. ``max_checks`` bounds the
    number of calls to the abstraction. ``bound_fn`` must accept ``core=True``
    (see ``lirpa.crown.layer_bounds``).
    '''
    layers = crown.from_sequential(model, x0.shape)
    witness = falsify(layers, x0, eps, label, constraints)
    if witness is not None:
        return 'sat', witness
    folded = crown.margin_layers(model, x0.shape, label)

    def check(assignment):
        splits = split_tensors(folded, assignment)
        return bound_fn(folded, x0, eps, splits=splits, core=True)

    trail, clauses, checks, stuck = [], [], 0, False
    while checks < max_checks:
        trail, clash = propagate(trail, clauses)
        decisions = [(n, s) for n, s, reason in trail if reason is None]
        if clash is not None:
            decisions = explain(trail, [(n, -s) for n, s in clash])
        else:
            assignment = assignment_of(trail)
            bounds, effective = check(assignment)
            checks += 1
            if proven(bounds):
                decisions = explain(trail, effective_literals(
                    folded, bounds, effective, assignment))
            else:
                neuron = pick_neuron(folded, bounds)
                if neuron is not None:
                    trail.append((neuron, +1, None))
                    continue
                stuck = True  # nothing left to split, still not proven
        if not decisions:
            return ('unknown' if stuck else 'unsat'), None
        clauses.append([(n, -s) for n, s in decisions])
        trail = backjump(trail, decisions[-1])
    return 'unknown', None
