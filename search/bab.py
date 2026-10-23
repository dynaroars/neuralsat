'''Branch and bound: split unstable ReLUs, prune subdomains by bounding.

Each subdomain is a ReLU phase assignment. The abstraction bounds the
margin ``f_label - f_k`` on it; if the lower bound is positive (or the
subdomain is empty) the subdomain is proven, otherwise the unstable neuron
with the largest relaxation area is split into its active and inactive
halves. Search is breadth first.
'''

from collections import deque

from abstraction.lirpa import crown
from search.common import (
    falsify,
    pick_neuron,
    proven,
    split_tensors,
)


def bab(model, x0, eps, label, bound_fn=crown.layer_bounds,
        constraints=(), max_nodes=1000):
    '''Decide whether ``label`` is the argmax on the whole input region.

    Returns ``(status, witness)``: ``('unsat', None)``, ``('sat', x)`` with
    a concrete counterexample ``x``, or ``('unknown', None)`` when the
    budget runs out or a subdomain has no unstable neuron left to split but
    is still not proven. ``constraints`` only filter counterexamples; the
    abstraction behind ``bound_fn`` is responsible for using them in bounds.
    '''
    layers = crown.from_sequential(model, x0.shape)
    witness = falsify(layers, x0, eps, label, constraints)
    if witness is not None:
        return 'sat', witness
    folded = crown.margin_layers(model, x0.shape, label)
    queue, nodes, stuck = deque([{}]), 0, False
    while queue and nodes < max_nodes:
        assignment = queue.popleft()
        nodes += 1
        splits = split_tensors(folded, assignment)
        bounds = bound_fn(folded, x0, eps, splits=splits)
        if proven(bounds):
            continue
        neuron = pick_neuron(folded, bounds)
        if neuron is None:
            stuck = True
            continue
        queue.extend({**assignment, neuron: sign} for sign in (+1, -1))
    return ('unknown' if queue or stuck else 'unsat'), None
