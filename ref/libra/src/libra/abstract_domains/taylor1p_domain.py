"""
Taylor1+ Abstract Domain
========================

:Authors: Caterina Urban
"""
from typing import Set

from apronpy.manager import PyManager
from apronpy.t1p import PyT1p

from libra.abstract_domains.apron_domain import APRONState
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier


class Taylor1pState(APRONState):
    """Taylor1+ analysis state based on APRON. An element of the taylor1+ abstract domain.

    .. document private methods
    .. automethod:: Taylor1pState._assign
    .. automethod:: Taylor1pState._assume
    .. automethod:: Taylor1pState._output
    .. automethod:: Taylor1pState._substitute

    """

    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(manager, variables, PyT1p, precursory=precursory)
