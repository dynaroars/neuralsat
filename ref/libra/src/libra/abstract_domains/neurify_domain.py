"""
Bia Abstract Domain: Neurify
============================

Disjunctive relation abstract domain to be used for **algorithmic bias analysis**.

"""
from typing import Set, List, Dict
from copy import deepcopy

from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from apronpy.manager import PyManager

from libra.abstract_domains.state import State
from libra.core.expressions import Lyra2APRON, NegationFreeExpression, VariableIdentifier, Expression, \
    BinaryComparisonOperation, BinaryBooleanOperation, Literal
from libra.abstract_domains.deeppoly_domain import IntervalLattice, texpr_to_dict, evaluate
from libra.core.utils import copy_docstring
from libra.abstract_domains.bounds_domain import BoundsDomain

LOW = 0
UP = 1

class NeurifyState(State, BoundsDomain):

    """Neurify [Wang et al.] state.

    .. document private methods
    .. automethod:: DeepPolyState._assign
    .. automethod:: DeepPolyState._assume
    .. automethod:: DeepPolyState._output
    .. automethod:: DeepPolyState._substitute
    """
    def __init__(self, inputs: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        self.inputs = {input.name for input in inputs}

        # PRE: all the inputs in [0, 1]
        self.bounds = {input: (IntervalLattice(0, 0), IntervalLattice(1, 1))
                       for input in self.inputs}

        # the free coeff is indicated by '_' in the dictionary self.poly
        self.poly = {input: ({'_': 0.0}, {'_': 1.0}) for input in self.inputs}

        self.flag = None

    @copy_docstring(State.bottom)
    def bottom(self):
        for var in self.bounds:
            self.bounds[var][LOW].bottom()
            self.bounds[var][UP].bottom()
        return self

    @copy_docstring(State.top)
    def top(self):
        for var in self.bounds:
            self.bounds[var][LOW].top()
            self.bounds[var][UP].top()
        return self

    def __repr__(self):
        items = sorted(self.bounds.items(), key=lambda x: x[0])
        return ", ".join(f"{k} -> (l:{l}, u:{u})" for k, (l, u) in items)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any(l.is_bottom() or u.is_bottom()
                   for (l, u) in self.bounds.values())

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all(l.is_top() and u.is_top()
                   for (l, u) in self.bounds.values())

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'NeurifyState') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'NeurifyState') -> 'NeurifyState':
        for var in self.bounds:
            other_bound = IntervalLattice(other.bounds[var][LOW].lower, other.bounds[var][UP].upper)
            self_bound = IntervalLattice(self.bounds[var][LOW].lower, self.bounds[var][UP].upper)
            self_bound.join(other_bound)
            self.bounds[var] = (
                IntervalLattice(self_bound.lower, self_bound.lower),
                IntervalLattice(self_bound.upper, self_bound.upper))

            self.poly[var] = (
                {'_': self.bounds[var][LOW].lower},
                {'_': self.bounds[var][UP].upper})
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'NeurifyState') -> 'NeurifyState':
        for var in self.bounds:
            other_bound = IntervalLattice(other.bounds[var][LOW].lower, other.bounds[var][UP].upper)
            self_bound = IntervalLattice(self.bounds[var][LOW].lower, self.bounds[var][UP].upper)
            self_bound.meet(other_bound)
            self.bounds[var] = (
                IntervalLattice(self_bound.lower, self_bound.lower),
                IntervalLattice(self_bound.upper, self_bound.upper))

            self.poly[var] = (
                {'_': self.bounds[var][LOW].lower},
                {'_': self.bounds[var][UP].upper})
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'NeurifyState') -> 'NeurifyState':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'NeurifyState':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'NeurifyState':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def __assume_helper(self, feature, lower, upper):
        self.bounds[feature.name] = (IntervalLattice(lower, lower), IntervalLattice(upper, upper))
        self.poly[feature.name] = (
            {'_': self.bounds[feature.name][LOW].lower},
            {'_': self.bounds[feature.name][UP].upper}
        )

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'NeurifyState':
        if self.is_bottom():
            return self
        if isinstance(condition, tuple):
            condition = list(condition)
        if isinstance(condition, list):
            for feature, (lower, upper) in condition:
                self.__assume_helper(feature, lower, upper)
            return self
        elif isinstance(condition, BinaryBooleanOperation):
            if condition.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self).assume(condition.right, bwd=bwd)
                return self.assume(condition.left, bwd=bwd).join(right)
            elif condition.operator == BinaryBooleanOperation.Operator.And:
                assert isinstance(condition.left, BinaryComparisonOperation)
                assert isinstance(condition.right, BinaryComparisonOperation)
                assert isinstance(condition.left.left, Literal)
                assert condition.left.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.left.right, VariableIdentifier)
                assert isinstance(condition.right.left, VariableIdentifier)
                assert condition.right.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.right.right, Literal)
                lower = eval(condition.left.left.val)
                upper = eval(condition.right.right.val)
                assert condition.left.right.name == condition.right.left.name
                self.__assume_helper(condition.left.right, lower, upper)
                return self
        elif isinstance(condition, BinaryComparisonOperation):
            if condition.operator == BinaryComparisonOperation.Operator.Gt:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = eval(condition.right.val)
                upper = 1
                self.__assume_helper(condition.left, lower, upper)
                return self
            elif condition.operator == BinaryComparisonOperation.Operator.LtE:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = 0
                upper = eval(condition.right.val)
                self.__assume_helper(condition.left, lower, upper)
                return self
        elif isinstance(condition, set):
            assert len(condition) == 1
            self.assume(condition.pop(), bwd=bwd)
            return self
        raise NotImplementedError(f"Assumption of {condition.__class__.__name__} is unsupported")

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'NeurifyState':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def forget(self, variables: List[VariableIdentifier]) -> 'NeurifyState':
        raise NotImplementedError(f"Call to _forget is unexpected!")

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'NeurifyState':
        if self.is_bottom():
            return self
        for lhs, expr in zip(left, right):
            name = str(lhs)
            rhs = texpr_to_dict(expr)
            _inf, inf = deepcopy(rhs), deepcopy(rhs)
            _sup, sup = deepcopy(rhs), deepcopy(rhs)
            self.poly[name] = (_inf, _sup)
            # while there is a non-input-variable in inf
            while any(variable in inf and variable not in self.inputs for variable in self.poly):
                for variable in self.poly:
                    if variable in inf and variable not in self.inputs:  # should be replaced
                        coeff = inf[variable]
                        if coeff > 0:
                            replacement = self.poly[variable][LOW]
                        elif coeff < 0:
                            replacement = self.poly[variable][UP]
                        else:  # coeff == 0
                            replacement = dict()
                            replacement['_'] = 0.0
                        del inf[variable]
                        for var, val in replacement.items():
                            if var in inf:
                                inf[var] += coeff * val
                            else:
                                inf[var] = coeff * val
            # while there is a non-input-variable in sup
            while any(variable in sup and variable not in self.inputs for variable in self.poly):
                for variable in self.poly:
                    if variable in sup and variable not in self.inputs:  # should be replaced
                        coeff = sup[variable]
                        if coeff > 0:
                            replacement = self.poly[variable][UP]
                        elif coeff < 0:
                            replacement = self.poly[variable][LOW]
                        else:  # coeff == 0
                            replacement = dict()
                            replacement['_'] = 0.0
                        del sup[variable]
                        for var, val in replacement.items():
                            if var in sup:
                                sup[var] += coeff * val
                            else:
                                sup[var] = coeff * val
            ext_bounds = {k: IntervalLattice(l.lower, u.upper) for k, (l, u) in self.bounds.items()}
            lower = evaluate(inf, ext_bounds)
            upper = evaluate(sup, ext_bounds)
            self.bounds[name] = (IntervalLattice(lower.lower, lower.upper),
                                 IntervalLattice(upper.lower, upper.upper))
        return self

    def set_flag(self, lower, upper, active, inactive):
        if upper <= 0 or inactive:
            self.flag = -1
        elif 0 <= lower or active:
            self.flag = 1
        else:
            self.flag = None

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'NeurifyState':
        if self.is_bottom():
            return self
        name = str(stmt)
        low_lattice, up_lattice = self.bounds[name]
        low_lower, low_upper = low_lattice.lower, low_lattice.upper
        up_lower, up_upper = up_lattice.lower, up_lattice.upper
        self.set_flag(low_lower, up_upper, active, inactive)
        if low_upper <= 0 or inactive:
            self.bounds[name] = (IntervalLattice(0, 0), self.bounds[name][UP])
            self.poly[name] = ({'_': 0.0}, self.poly[name][UP])
        elif 0 <= low_lower or active:
            if active and low_lower < 0:
                self.bounds[name] = (IntervalLattice(0, low_upper), self.bounds[name][UP])
                self.poly[name] = ({'_': 0.0}, self.poly[name][UP])
        else:
            m = low_upper / (low_upper - low_lower)
            sup = self.poly[name][LOW] # sup does side effects over self.poly[name][UP]
            for var, val in sup.items():
                sup[var] = m * val
            self.bounds[name] = (IntervalLattice(low_lower*m, low_upper*m), self.bounds[name][UP])

        if up_upper <= 0 or inactive:
            self.bounds[name] = (self.bounds[name][LOW], IntervalLattice(0, 0))
            self.poly[name] = (self.poly[name][LOW], {'_': 0.0})
        elif 0 <= up_lower or active:
            if active and up_lower < 0:
                self.bounds[name] = (self.bounds[name][LOW], IntervalLattice(0, up_upper))
        else:
            m = up_upper / (up_upper - up_lower)
            q = up_upper * (-up_lower) / (up_upper - up_lower)
            sup = self.poly[name][UP] # sup does side effects over self.poly[name][UP]
            for var, val in sup.items():
                sup[var] = m * val
            sup['_'] += q
            self.bounds[name] = (self.bounds[name][LOW], IntervalLattice(up_lower*m+q, up_upper*m+q))
        return self

    def outcome(self, outcomes: Set[VariableIdentifier]):
        found = None
        if self.is_bottom():
            found = '⊥'
        else:
            for chosen in outcomes:
                outcome = self.bounds[chosen.name][LOW]
                lower = outcome.lower
                unique = True
                remaining = outcomes - {chosen}
                for discarded in remaining:
                    alternative = self.bounds[discarded.name][UP]
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

    def get_bounds(self, var_name):
        low, up = self.bounds[var_name]
        return IntervalLattice(low.lower, up.upper)

    def resize_bounds(self, var_name, new_bounds):
        new_low_lower = new_bounds.lower
        new_low_upper = self.bounds[var_name][LOW].upper
        new_up_lower = self.bounds[var_name][UP].lower
        new_up_upper = new_bounds.upper

        if new_low_lower > new_low_upper:
            new_low_upper = new_low_lower
        if new_up_upper < new_up_lower:
            new_up_lower = new_up_upper

        self.bounds[var_name] = (
            IntervalLattice(new_low_lower, new_low_upper),
            IntervalLattice(new_up_lower, new_up_upper)
            )
