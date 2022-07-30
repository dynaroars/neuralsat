"""
Symbolic Constant Propagation
=============================

:Authors: Caterina Urban
"""
from abc import abstractmethod
from typing import Set, List, Dict, Tuple

from apronpy.box import PyBox
from apronpy.manager import PyManager
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.apron_domain import APRONState
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier


class SymbolicState(APRONState):
    """Interval+Symbolic.

    .. document private methods
    .. automethod:: SymbolicState._assign
    .. automethod:: SymbolicState._assume
    .. automethod:: SymbolicState._output
    .. automethod:: SymbolicState._substitute

    """
    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(manager, variables, PyBox, precursory=precursory)
        self.symbols: Dict[str, Tuple[PyVar, PyTexpr1]] = dict()
        self.flag = None

    @abstractmethod
    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'SymbolicState':
        """Affine layer.

        :param left: left-hand side
        :param right: right-hand side
        :return:
        """

    @abstractmethod
    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'SymbolicState':
        """ReLU activation function.

        :param stmt: variable concerned by the ReLU
        :param active: only consider the active case
        :param inactive: only consider the inactive case
        :return:
        """
