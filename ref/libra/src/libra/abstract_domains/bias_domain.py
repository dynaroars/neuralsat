"""
Bias Abstract Domain
====================

Disjunctive relational abstract domain to be used for **algorithmic bias analysis**.

:Authors: Caterina Urban
"""
from copy import deepcopy
from itertools import chain
from typing import Set, List

from apronpy.environment import PyEnvironment
from apronpy.lincons1 import PyLincons1Array
from apronpy.manager import PyManager
from apronpy.polka import PyPolka
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.var import PyVar

from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier, Expression, BinaryComparisonOperation, \
    BinaryBooleanOperation, Lyra2APRON, \
    NegationFreeExpression
from libra.core.utils import copy_docstring


class BiasState(State):
    """Neural network analysis state.

    .. document private methods
    .. automethod:: BiasState._assign
    .. automethod:: BiasState._assume
    .. automethod:: BiasState._output
    .. automethod:: BiasState._substitute

    """

    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        r_vars = list()
        for variable in variables:
            r_vars.append(PyVar(variable.name))
        self.environment = PyEnvironment([], r_vars)
        self.polka = PyPolka(manager, self.environment)
        # self.mirror = PyPolkaMPQstrict(self.environment)

    @copy_docstring(State.bottom)
    def bottom(self, manager: PyManager = None):
        self.polka = PyPolka.bottom(manager, self.environment)
        return self

    @copy_docstring(State.top)
    def top(self, manager: PyManager = None):
        self.polka = PyPolka.top(manager, self.environment)
        return self

    def __repr__(self):
        if self.is_bottom():
            return '⊥'
        # return '{} \n∈ {}'.format(self.polka, self.mirror)
        return '{}'.format(self.polka)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return self.polka.is_bottom()

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return self.polka.is_top()

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'BiasState') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'BiasState') -> 'BiasState':
        self.polka = self.polka.join(other.polka)
        # self.mirror = self.mirror.join(other.mirror)
        return self

    @copy_docstring(State._meet)
    def _meet(self, other) -> 'BiasState':
        self.polka = self.polka.meet(other.polka)
        # self.mirror = self.mirror.meet(other.mirror)
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'BiasState') -> 'BiasState':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    def _assign(self, left: Expression, right: Expression) -> 'BiasState':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, manager: PyManager = None, bwd: bool = False) -> 'BiasState':
        assert manager is not None
        normal = self._negation_free.visit(condition)
        if isinstance(normal, BinaryBooleanOperation):
            if normal.operator == BinaryBooleanOperation.Operator.And:
                return self._assume(normal.left, manager=manager, bwd=False)._assume(normal.right, manager=manager, bwd=False)
            elif normal.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self)._assume(normal.right, manager=manager, bwd=False)
                return self._assume(normal.left, manager=manager, bwd=False).join(right)
        elif isinstance(normal, BinaryComparisonOperation):
            cond = self._lyra2apron.visit(normal, self.environment)
            abstract1 = PyPolka(manager, self.environment, array=PyTcons1Array([cond]))
            self.polka = self.polka.meet(abstract1)
            # if not bwd:
            #     self.mirror = self.mirror.meet(abstract1)
            return self
        raise NotImplementedError(f"Assumption of {normal.__class__.__name__} is unsupported!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'BiasState':
        assert manager is not None
        if isinstance(condition, PyTcons1):
            abstract1 = PyPolka(manager, self.environment, array=PyTcons1Array([condition]))
            self.polka = self.polka.meet(abstract1)
            return self
        else:
            assert len(condition) == 1
            self._assume(condition.pop(), manager=manager, bwd=bwd)
            return self

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'BiasState':
        if isinstance(left, VariableIdentifier):
            var = PyVar(left.name)
            expr = self._lyra2apron.visit(right, self.environment)
            self.polka = self.polka.substitute(var, expr)
            # self.mirror = self.mirror.substitute(var, expr)
            return self
        elif isinstance(left, list):
            assert all(isinstance(lhs, VariableIdentifier) for lhs in left)
            assert isinstance(right, list)
            vars = [PyVar(lhs.name) for lhs in left]
            exprs = [self._lyra2apron.visit(rhs, self.environment) for rhs in right]
            self.polka = self.polka.substitute(vars, exprs)
            # self.mirror = self.mirror.substitute(vars, exprs)
            return self
        raise NotImplementedError(f"Substitution of {left.__class__.__name__} is unsupported!")

    def substitute(self, left: Set[Expression], right: Set[Expression]) -> 'BiasState':
        if isinstance(left, list) and isinstance(right, list):
            lhss = [lhs.pop() for lhs in left]
            rhss = [rhs.pop() for rhs in right]
            self._substitute(lhss, rhss)
        else:
            assert len(left) == 1 and len(right) == 1
            self._substitute(left.pop(), right.pop())
        self.result = set()  # assignments have no result, only side-effects
        return self

    def forget(self, variables: List[VariableIdentifier]) -> 'BiasState':
        vars = [PyVar(var.name) for var in variables]
        self.polka = self.polka.forget(vars)
        # self.mirror = self.mirror.forget(vars)
        return self

    _negation_free = NegationFreeExpression()
    _lyra2apron = Lyra2APRON()
