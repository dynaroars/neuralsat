"""
Bias Abstract Domain
====================

Disjunctive relational abstract domain to be used for **algorithmic bias analysis**.

:Authors: Caterina Urban
"""
from copy import deepcopy
from typing import Set, List

from apronpy.manager import PyManager
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier, Expression, BinaryComparisonOperation, \
    BinaryBooleanOperation, Lyra2APRON, NegationFreeExpression, Literal
from libra.core.utils import copy_docstring
from libra.abstract_domains.bounds_domain import BoundsDomain
from libra.abstract_domains.deeppoly_domain import IntervalLattice, texpr_to_dict, evaluate


class Box2State(State, BoundsDomain):
    """DeepPoly [Singh et al. POPL2019] state.

    .. document private methods
    .. automethod:: Box2State._assign
    .. automethod:: Box2State._assume
    .. automethod:: Box2State._output
    .. automethod:: Box2State._substitute

    """
    def __init__(self, inputs: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        self.inputs = {input.name for input in inputs}
        self.bounds = dict()
        for input in self.inputs:
            self.bounds[input] = IntervalLattice(0, 1)
        self.flag = None

    @copy_docstring(State.bottom)
    def bottom(self):
        for var in self.bounds:
            self.bounds[var].bottom()
        return self

    @copy_docstring(State.top)
    def top(self):
        for var in self.bounds:
            self.bounds[var].top()
        return self

    def __repr__(self):
        items = sorted(self.bounds.items(), key=lambda x: x[0])
        return ", ".join("{} -> {}".format(variable, value) for variable, value in items)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any(element.is_bottom() for element in self.bounds.values())

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all(element.is_top() for element in self.bounds.values())

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'Box2State') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'Box2State') -> 'Box2State':
        for var in self.bounds:
            self.bounds[var].join(other.bounds[var])
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'Box2State') -> 'Box2State':
        for var in self.bounds:
            self.bounds[var].meet(other.bounds[var])
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'Box2State') -> 'Box2State':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'Box2State':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'Box2State':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'Box2State':
        if self.is_bottom():
            return self
        if isinstance(condition, tuple):
            condition = list(condition)
        if isinstance(condition, list):
            for feature, (lower, upper) in condition:
                self.bounds[feature.name].meet(IntervalLattice(lower, upper))
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
                self.bounds[condition.left.right.name].meet(IntervalLattice(lower, upper))
                return self
        elif isinstance(condition, BinaryComparisonOperation):
            if condition.operator == BinaryComparisonOperation.Operator.Gt:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = eval(condition.right.val)
                upper = 1
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                return self
            elif condition.operator == BinaryComparisonOperation.Operator.LtE:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = 0
                upper = eval(condition.right.val)
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                return self
        # elif isinstance(condition, PyTcons1):
        #     abstract1 = self.domain(manager, self.environment, array=PyTcons1Array([condition]))
        #     self.state = self.state.meet(abstract1)
        #     return self
        elif isinstance(condition, set):
            assert len(condition) == 1
            self.assume(condition.pop(), bwd=bwd)
            return self
        raise NotImplementedError(f"Assumption of {condition.__class__.__name__} is unsupported!")

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'Box2State':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def forget(self, variables: List[VariableIdentifier]) -> 'Box2State':
        raise NotImplementedError(f"Call to _forget is unexpected!")

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'Box2State':
        if self.is_bottom():
            return self
        for lhs, expr in zip(left, right):
            name = str(lhs)
            rhs = texpr_to_dict(expr)
            inf = deepcopy(rhs)
            sup = deepcopy(rhs)

            lower = evaluate(inf, self.bounds)
            upper = evaluate(sup, self.bounds)
            self.bounds[name] = IntervalLattice(lower.lower, upper.upper)
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'Box2State':
        if self.is_bottom():
            return self
        name = str(stmt)
        lattice: IntervalLattice = self.bounds[name]
        lower, upper = lattice.lower, lattice.upper
        if upper <= 0 or inactive:
            # l_j = u_j = 0
            self.bounds[name] = IntervalLattice(0, 0)
            # 0 <= x_j <= 0
            self.flag = -1
        elif 0 <= lower or active:
            if active and lower < 0:
                self.bounds[name] = IntervalLattice(0, upper)
            self.flag = 1
        else:   # case (c) in Fig. 4, equation (4)
            _active, _inactive = deepcopy(self.bounds), deepcopy(self.bounds)
            _active[name] = _active[name].meet(IntervalLattice(0, upper))
            _inactive[name] = _inactive[name].meet(IntervalLattice(0, 0))

            if any(element.is_bottom() for element in _active.values()):
                self.flag = -1
            elif any(element.is_bottom() for element in _inactive.values()):
                self.flag = 1
            else:
                self.flag = None

            join = dict()
            for variable, itv in _active.items():
                join[variable] = itv.join(_inactive[variable])
            self.bounds[name] = join[name].meet(IntervalLattice(0, upper))
            self.flag = None
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

    def get_bounds(self, var_name):
        return self.bounds[var_name]

    def resize_bounds(self, var_name, new_bounds):
        self.bounds[var_name] = IntervalLattice(new_bounds.lower, new_bounds.upper)