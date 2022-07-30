"""
Forward Semantics
=================

Libra's internal forward semantics of statements.

:Authors: Caterina Urban
"""
from libra.semantics.semantics import Semantics, DefaultSemantics

from libra.abstract_domains.state import State
from libra.core.statements import Assignment, Call


class ForwardSemantics(Semantics):
    """Forward semantics of statements."""
    pass


class UserDefinedCallSemantics(ForwardSemantics):
    """Forward semantics of user-defined function/method calls."""

    # noinspection PyUnusedLocal
    def user_defined_call_semantics(self, stmt: Call, state: State):
        """Forward semantics of a user-defined function/method call.

        :param stmt: call statement to be executed
        :param state: state before executing the call statement
        :return: state modified by the call statement
        """
        error = f"Forward semantics for call statement {stmt} not yet implemented!"
        raise NotImplementedError(error)


class AssignmentSemantics(ForwardSemantics):
    """Forward semantics of assignments."""

    def assignment_semantics(self, stmt: Assignment, state: State) -> State:
        """Forward semantics of an assignment.

        :param stmt: assignment statement to be executed
        :param state: state before executing the assignment
        :return: state modified by the assignment
        """
        lhs = self.semantics(stmt.left, state).result    # lhs evaluation
        rhs = self.semantics(stmt.right, state).result   # rhs evaluation
        return state.assign(lhs, rhs)


# noinspection PyAbstractClass
class DefaultForwardSemantics(DefaultSemantics, UserDefinedCallSemantics, AssignmentSemantics):
    """Default forward semantics of statements."""
    pass
