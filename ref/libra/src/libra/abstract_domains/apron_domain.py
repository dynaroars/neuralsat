"""
APRON-based Abstract Domain
===========================

Abstract domain based on APRON to be used for **numerical analysis**.
The set of possible numerical values of a program variable in a program state
is represented by a conjunction of (more of less complex) linear constraints.

:Authors: Caterina Urban
"""
import sys
from abc import ABCMeta
from copy import deepcopy
from enum import IntEnum
from typing import Set, Type

from apronpy.abstract1 import Abstract1, PyAbstract1
from apronpy.environment import PyEnvironment
from apronpy.linexpr1 import PyLinexpr1
from apronpy.manager import PyManager
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier, Expression, BinaryBooleanOperation, \
    BinaryComparisonOperation, NegationFreeExpression, Lyra2APRON
from libra.core.utils import copy_docstring


class APRONState(State, metaclass=ABCMeta):
    """Analysis state based on APRON. An element of the abstract domain.

    Conjunction of constraints constraining the value of each variable.
    The value of all program variables is unconstrained by default.

    .. note:: Program variables storing collections are not supported yet.

    .. document private methods
    .. automethod:: APRONState._assign
    .. automethod:: APRONState._assume
    .. automethod:: APRONState._output
    .. automethod:: APRONState._substitute

    """

    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier],
                 domain: Type[PyAbstract1], precursory: State = None):
        super().__init__(precursory=precursory)
        r_vars = list()
        for variable in variables:
            r_vars.append(PyVar(variable.name))
        self.environment = PyEnvironment([], r_vars)
        self.domain = domain
        self.state = self.domain(manager, self.environment)

    @copy_docstring(State.bottom)
    def bottom(self, manager: PyManager = None):
        assert manager is not None
        self.state = self.domain.bottom(manager, self.environment)
        return self

    @copy_docstring(State.top)
    def top(self, manager: PyManager = None):
        assert manager is not None
        self.state = self.domain.top(manager, self.environment)
        return self

    def __repr__(self):
        if self.is_bottom():
            return "⊥"
        return '{}'.format(self.state)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return self.state.is_bottom()

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return self.state.is_top()

    def bound_variable(self, variable: PyVar):
        return self.state.bound_variable(variable)

    def bound_linexpr(self, linexpr: PyLinexpr1):
        return self.state.bound_linexpr(linexpr)

    def bound_texpr(self, texpr: PyTexpr1):
        return self.state.bound_texpr(texpr)

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'APRONState') -> bool:
        return self.state <= other.state

    @copy_docstring(State._join)
    def _join(self, other: 'APRONState') -> 'APRONState':
        self.state = self.state.join(other.state)
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'APRONState') -> 'APRONState':
        self.state = self.state.meet(other.state)
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'APRONState') -> 'APRONState':
        self.state = self.state.widening(other.state)
        return self

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'APRONState':
        if isinstance(left, VariableIdentifier):
            expr = self._lyra2apron.visit(right, self.environment)
            self.state = self.state.assign(PyVar(left.name), expr)
            return self
        elif isinstance(left, list):
            assert all(isinstance(lhs, VariableIdentifier) for lhs in left)
            assert isinstance(right, list)
            vars = [PyVar(lhs.name) for lhs in left]
            exprs = [self._lyra2apron.visit(rhs, self.environment) for rhs in right]
            self.state = self.state.assign(vars, exprs)
            return self
        raise NotImplementedError(f"Assignment to {left.__class__.__name__} is unsupported!")

    def assign(self, left: Set[Expression], right: Set[Expression]) -> 'APRONState':
        if isinstance(left, list) and isinstance(right, list):
            lhss = [lhs.pop() for lhs in left]
            rhss = [rhs.pop() for rhs in right]
            self._assign(lhss, rhss)
        else:
            self.big_join([deepcopy(self)._assign(lhs, rhs) for lhs in left for rhs in right])
        self.result = set()  # assignments have no result, only side-effects
        return self

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, manager: PyManager = None, bwd: bool = False) -> 'APRONState':
        normal = self._negation_free.visit(condition)
        if isinstance(normal, BinaryBooleanOperation):
            if normal.operator == BinaryBooleanOperation.Operator.And:
                right = deepcopy(self)._assume(normal.right, manager=manager, bwd=bwd)
                return self._assume(normal.left, manager=manager, bwd=bwd).meet(right)
            if normal.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self)._assume(normal.right, manager=manager, bwd=bwd)
                return self._assume(normal.left, manager=manager, bwd=bwd).join(right)
        elif isinstance(normal, BinaryComparisonOperation):
            cond = self._lyra2apron.visit(normal, self.environment)
            abstract1 = self.domain(manager, self.environment, array=PyTcons1Array([cond]))
            self.state = self.state.meet(abstract1)
            return self
        raise NotImplementedError(f"Assumption of {normal.__class__.__name__} is unsupported!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'APRONState':
        if isinstance(condition, PyTcons1):
            abstract1 = self.domain(manager, self.environment, array=PyTcons1Array([condition]))
            self.state = self.state.meet(abstract1)
            return self
        else:
            assert len(condition) == 1
            self._assume(condition.pop(), manager=manager, bwd=bwd)
            return self

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'APRONState':
        if isinstance(left, VariableIdentifier):
            expr = self._lyra2apron.visit(right, self.environment)
            self.state = self.state.substitute(PyVar(left.name), expr)
            return self
        raise NotImplementedError(f"Substitution of {left.__class__.__name__} is unsupported!")

    def outcome(self, outcomes: Set[VariableIdentifier]):
        found = None
        for chosen in outcomes:
            outcome = self.bound_variable(PyVar(chosen.name))
            inf = str(outcome.interval.contents.inf.contents)
            lower = eval(inf) if inf != '-1/0' else -sys.maxsize
            unique = True
            remaining = outcomes - {chosen}
            for discarded in remaining:
                alternative = self.bound_variable(PyVar(discarded.name))
                sup = str(alternative.interval.contents.sup.contents)
                upper = eval(sup) if sup != '1/0' else sys.maxsize
                if lower <= upper:
                    unique = False
                    break
            if unique:
                found = chosen
                break
        return found

    _negation_free = NegationFreeExpression()
    _lyra2apron = Lyra2APRON()
