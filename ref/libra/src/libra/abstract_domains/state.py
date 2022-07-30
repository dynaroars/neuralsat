"""
Abstract Domain
===============

Interface of an abstract domain.
Abstract domain elements support lattice operations and program statements.

:Author: Caterina Urban
"""


from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Set, Optional, List, Type, Dict, Any

from libra.abstract_domains.lattice import Lattice
from libra.core.expressions import Expression, VariableIdentifier
from libra.core.statements import ProgramPoint
from libra.core.utils import copy_docstring


class State(Lattice, metaclass=ABCMeta):
    """Analysis state. A mutable element of an abstract domain.

    .. warning::
        Lattice operations and statements modify the current state.
    """

    def __init__(self, precursory: 'State' = None):
        super().__init__()
        self._result = set()
        self._pp = None
        self._precursory = precursory

    @property
    def result(self):
        """Result of the previously analyzed statement."""
        return self._result

    @result.setter
    def result(self, result: Set[Expression]):
        self._result = result

    @property
    def pp(self):
        """Program point of the currently analyzed statement."""
        return self._pp

    @pp.setter
    def pp(self, pp: ProgramPoint):
        self._pp = pp

    @property
    def precursory(self):
        """Current precursory analysis state."""
        return self._precursory

    @precursory.setter
    def precursory(self, precursory: 'State'):
        self._precursory = precursory

    def __repr__(self):
        return ", ".join("{}".format(expression) for expression in self.result)

    @abstractmethod
    def _assign(self, left: Expression, right: Expression) -> 'State':
        """Assign an expression to another expression.

        .. warning::
            The current state could also be bottom or top.

        :param left: expression to be assigned to
        :param right: expression to assign
        :return: current state modified by the assignment

        """

    def assign(self, left: Set[Expression], right: Set[Expression]) -> 'State':
        """Assign an expression to another expression.

        :param left: set of expressions representing the expression to be assigned to
        :param right: set of expressions representing the expression to assign
        :return: current state modified by the assignment

        """
        self.big_join([deepcopy(self)._assign(lhs, rhs) for lhs in left for rhs in right])
        self.result = set()  # assignments have no result, only side-effects
        return self

    @abstractmethod
    def _assume(self, condition: Expression, bwd: bool = False) -> 'State':
        """Assume that some condition holds in the current state.

        .. warning::
            The current state could also be bottom or top.

        :param condition: expression representing the assumed condition
        :param bwd: whether the assumption happens in a backward analysis (default: False)
        :return: current state modified to satisfy the assumption

        """

    def assume(self, condition: Set[Expression], bwd: bool = False) -> 'State':
        """Assume that some condition holds in the current state.

        :param condition: set of expressions representing the assumed condition
        :param bwd: whether the assumption happens in a backward analysis (default: False)
        :return: current state modified to satisfy the assumption

        """
        self.big_join([deepcopy(self)._assume(expr, bwd=bwd) for expr in condition])
        return self

    def before(self, pp: ProgramPoint, precursory: Optional['State']) -> 'State':
        """Set the program point of the currently analyzed statement
        and the current precursory analysis state.

        :param pp: current program point
        :param precursory: current precursory analysis state
        :return: current state modified to set the current program point and precursory state
        """
        self.pp = pp
        self.precursory = precursory
        return self

    def filter(self, bwd: bool = False) -> 'State':
        """Assume that the current result holds in the current state.

        :param bwd: whether the filtering happens in a backward analysis (default: False)
        :return: current state modified to satisfy the current result

        """
        self.assume(self.result, bwd=bwd)
        self.result = set()  # filtering has no result, only side-effects
        return self

    @abstractmethod
    def _substitute(self, left: Expression, right: Expression) -> 'State':
        """Substitute an expression to another expression.

        .. warning::
            The current state could also be bottom or top.

        :param left: expression to be substituted
        :param right: expression to substitute
        :return: current state modified by the substitution

        """

    def substitute(self, left: Set[Expression], right: Set[Expression]) -> 'State':
        """Substitute an expression to another expression.

        :param left: set of expressions representing the expression to be substituted
        :param right: set of expressions representing the expression to substitute
        :return: current state modified by the substitution

        """
        self.big_join([deepcopy(self)._substitute(l, r) for l in left for r in right])
        self.result = set()  # assignments have no result, only side-effects
        return self


class EnvironmentMixin(State, metaclass=ABCMeta):
    """Mixin to add environment modification operations to another state."""

    @abstractmethod
    def add_variable(self, variable: VariableIdentifier):
        """Add a variable.

        :param variable: variable to be added
        :return: current state modified by the variable addition
        """

    @abstractmethod
    def forget_variable(self, variable: VariableIdentifier):
        """Forget the value of a variable.

        :param variable: variable whose value is to be forgotten
        :return: current state modified to have value top for the forgotten variable
        """

    @abstractmethod
    def remove_variable(self, variable: VariableIdentifier):
        """Remove a variable.

        :param variable: variable to be removed
        :return: current state modified by the variable removal
        """


class ProductState(State):
    """Product analysis state. A mutable element of a product abstract domain.
    
    .. warning::
        Lattice operations and statements modify the current state.
    """
    
    def __init__(self, states: List[Type[State]], arguments=None, precursory: State = None):
        super().__init__(precursory)
        if arguments is None:
            arguments: Dict[Type, Dict[str, Any]] = defaultdict(lambda: dict())
        self._states = [state(**arguments[state]) for state in states]

    @property
    def states(self):
        """Current list of states."""
        return self._states
    
    def __repr__(self):
        if self.is_bottom():
            return "⊥"
        return "\n".join(str(state) for state in self.states)

    @copy_docstring(State.bottom)
    def bottom(self) -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state.bottom()
        return self

    @copy_docstring(State.top)
    def top(self) -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state.top()
        return self

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any(state.is_bottom() for state in self.states)

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all(state.is_top() for state in self.states)

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'ProductState') -> bool:
        return all(state.less_equal(other.states[i]) for i, state in enumerate(self.states))

    @copy_docstring(State._join)
    def _join(self, other: 'ProductState') -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state.join(other.states[i])
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'ProductState'):
        for i, state in enumerate(self.states):
            self.states[i] = state.meet(other.states[i])
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'ProductState'):
        for i, state in enumerate(self.states):
            self.states[i] = state.widening(other.states[i])
        return self

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state._assign(left, right)
        return self

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state._assume(condition, bwd=bwd)
        return self

    @copy_docstring(State.before)
    def before(self, pp: ProgramPoint, precursory: Optional['State']) -> 'ProductState':
        super().before(pp, precursory)
        for i, state in enumerate(self.states):
            self.states[i] = state.before(pp, precursory)
        return self

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'ProductState':
        for i, state in enumerate(self.states):
            self.states[i] = state._substitute(left, right)
        return self
