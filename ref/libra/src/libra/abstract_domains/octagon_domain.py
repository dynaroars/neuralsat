"""
Octagon Abstract Domain
=======================

Relational abstract domain to be used for **numerical analysis**.
The set of possible numerical values of a program variable in a program state
is represented by a conjunction of linear constraints.

:Authors: Caterina Urban
"""
from typing import Set

from apronpy.manager import PyManager
from apronpy.oct import PyOct

from libra.abstract_domains.apron_domain import APRONState
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier


class OctagonState(APRONState):
    """Octagon analysis state based on APRON. An element of the octagon abstract domain.

    .. document private methods
    .. automethod:: OctagonState._assign
    .. automethod:: OctagonState._assume
    .. automethod:: OctagonState._output
    .. automethod:: OctagonState._substitute

    """

    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(manager, variables, PyOct, precursory=precursory)
