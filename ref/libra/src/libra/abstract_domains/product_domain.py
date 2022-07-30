"""
Bias Abstract Domain: Product between DeepPoly and Neurify
============================

Disjunctive relation abstract domain to be used for **algorithmic bias analysis**.

"""
from copy import deepcopy
from typing import Set, List, Dict
from math import isclose

from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from apronpy.manager import PyManager

from libra.abstract_domains.deeppoly_domain import IntervalLattice
from libra.abstract_domains.state import State
from libra.core.expressions import Lyra2APRON, NegationFreeExpression, VariableIdentifier, Expression
from libra.core.utils import copy_docstring
from libra.abstract_domains.bounds_domain import BoundsDomain

class ProductState(State):

    """
    .. document private methods
    .. automethod:: DeepPolyState._assign
    .. automethod:: DeepPolyState._assume
    .. automethod:: DeepPolyState._output
    .. automethod:: DeepPolyState._substitute
    """
    def __init__(self, inputs: Set[VariableIdentifier], domains, precursory: State = None):
        super().__init__(precursory=precursory)

        for domain in domains:
            assert(isinstance(domain, BoundsDomain) and isinstance(domain, State))

        self._domains = domains

        self.inputs = {k.name for k in inputs}
        self.bounds = dict()
        for k in self.inputs:
            self.bounds[k] = IntervalLattice(0, 1)

        self.flag = None

    @copy_docstring(State.bottom)
    def bottom(self):
        for domain in self._domains:
            domain.bottom()
        return self

    @copy_docstring(State.top)
    def top(self):
        for domain in self._domains:
            domain.top()
        return self

    def __repr__(self):
        items = sorted(self.bounds.items(), key=lambda x: x[0])
        return ", ".join("{} -> {}".format(variable, value) for variable, value in items)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any([domain.is_bottom() for domain in self._domains])

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all([domain.is_top() for domain in self._domains])

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'ProductState') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'ProductState') -> 'ProductState':
        for self_domain, other_domain in zip(self._domains, other._domains):
            assert type(self_domain) is type(other_domain)
            self_domain._join(other_domain)
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'ProductState') -> 'ProductState':
        for self_domain, other_domain in zip(self._domains, other._domains):
            assert type(self_domain) is type(other_domain)
            self_domain._meet(other_domain)
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'ProductState') -> 'ProductState':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'ProductState':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'ProductState':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'ProductState':
        for domain in self._domains:
            domain.assume(deepcopy(condition), manager, bwd)
        for input in self.inputs:
            self._share_bounds(str(input))
        return self

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'ProductState':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def forget(self, variables: List[VariableIdentifier]) -> 'ProductState':
        raise NotImplementedError(f"Call to _forget is unexpected!")

    def _share_bounds(self, var_name):
        lower = max([domain.get_bounds(var_name).lower for domain in self._domains])
        upper = min([domain.get_bounds(var_name).upper for domain in self._domains])
        if upper < lower and isclose(lower, upper):
            upper, lower = lower, upper
        self.bounds[var_name] = IntervalLattice(lower, upper)

        for domain in self._domains:
            domain.resize_bounds(var_name, self.bounds[var_name])

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'ProductState':
        for domain in self._domains:
            domain.affine(left, right)

        for lhs in left:
            self._share_bounds(str(lhs))

        return self

    def set_flag(self, lower, upper, active, inactive):
        if upper <= 0 or inactive:
            self.flag = -1
        elif 0 <= lower or active:
            self.flag = 1
        else:
            self.flag = None

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'ProductState':
        d1_lower = self._domains[0].get_bounds(str(stmt)).lower
        assert all([d1_lower == domain.get_bounds(str(stmt)).lower for domain in self._domains])
        d1_upper = self._domains[0].get_bounds(str(stmt)).upper
        assert all([d1_upper == domain.get_bounds(str(stmt)).upper for domain in self._domains])
        self.set_flag(d1_lower, d1_upper, active, inactive)

        for domain in self._domains:
            domain.relu(stmt, active, inactive)

        self._share_bounds(str(stmt))

        return self

    def outcome(self, outcomes: Set[VariableIdentifier]):
        found = None
        if self.is_bottom():
            found = '⊥'
        else:
            for chosen in outcomes:
                outcome = self.bounds[chosen.name]
                lower = outcome.lower
                unique = True
                remaining = outcomes - {chosen}
                for discarded in remaining:
                    alternative = self.bounds[discarded.name]
                    upper = alternative.upper
                    if lower <= upper:
                        unique = False
                        break
                if unique:
                    found = chosen
                    break
        return found

    _negation_free = NegationFreeExpression()
    _lyra2apron = Lyra2APRON()
