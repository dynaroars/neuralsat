"""
Polyhedra Abstract Domain
=========================

Relational abstract domain to be used for **numerical analysis**.
The set of possible numerical values of a program variable in a program state
is represented by a conjunction of linear constraints.

:Authors: Caterina Urban
"""
from typing import Set

from apronpy.manager import PyManager
from apronpy.polka import PyPolka

from libra.abstract_domains.apron_domain import APRONState
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier


class PolyhedraState(APRONState):
    """Polyhedra analysis state based on APRON. An element of the polyhedra abstract domain.

    .. document private methods
    .. automethod:: PolyhedraState._assign
    .. automethod:: PolyhedraState._assume
    .. automethod:: PolyhedraState._output
    .. automethod:: PolyhedraState._substitute

    """

    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(manager, variables, PyPolka, precursory=precursory)
