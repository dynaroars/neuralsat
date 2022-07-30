"""
Intervals
=========

:Authors: Caterina Urban
"""
from copy import deepcopy
from typing import Set, List

from apronpy.box import PyBox
from apronpy.coeff import PyMPQScalarCoeff
from apronpy.lincons0 import ConsTyp
from apronpy.manager import PyManager
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.apron_domain import APRONState
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier


rtype = TexprRtype.AP_RTYPE_REAL
rdir = TexprRdir.AP_RDIR_RND


class Box1State(APRONState):
    """Interval.

    .. document private methods
    .. automethod:: Box1State._assign
    .. automethod:: Box1State._assume
    .. automethod:: Box1State._output
    .. automethod:: Box1State._substitute

    """
    def __init__(self, manager: PyManager, variables: Set[VariableIdentifier], precursory: State = None):
        super().__init__(manager, variables, PyBox, precursory=precursory)
        self.flag = None

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'Box1State':
        self.state = self.state.assign(left, right)
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'Box1State':
        self.flag = None

        if active:      # we only do the active case (h >= 0)
            expr1 = PyTexpr1.var(self.environment, stmt)
            cond = PyTcons1.make(expr1, ConsTyp.AP_CONS_SUPEQ)
            abstract1 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond]))
            self.state = self.state.meet(abstract1)
            self.flag = 1
            return self

        if inactive:    # we only do the inactive case (h < 0)
            expr1 = PyTexpr1.var(self.environment, stmt)
            neg = PyTexpr1.unop(TexprOp.AP_TEXPR_NEG, expr1, rtype, rdir)
            cond = PyTcons1.make(neg, ConsTyp.AP_CONS_SUP)
            abstract1 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond]))
            zero = PyTexpr1.cst(self.environment, PyMPQScalarCoeff(0.0))
            self.state = self.state.meet(abstract1).assign(stmt, zero)
            self.flag = -1
            return self

        # we do both cases
        _active, _inactive = deepcopy(self.state), deepcopy(self.state)

        expr1 = PyTexpr1.var(self.environment, stmt)
        cond1 = PyTcons1.make(expr1, ConsTyp.AP_CONS_SUPEQ)
        abstract1 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond1]))
        _active = _active.meet(abstract1)

        expr2 = PyTexpr1.var(self.environment, stmt)
        neg = PyTexpr1.unop(TexprOp.AP_TEXPR_NEG, expr2, rtype, rdir)
        cond2 = PyTcons1.make(neg, ConsTyp.AP_CONS_SUP)
        abstract2 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond2]))
        zero = PyTexpr1.cst(self.environment, PyMPQScalarCoeff(0.0))
        _inactive = _inactive.meet(abstract2).assign(stmt, zero)

        if _active.is_bottom():
            self.flag = -1
        elif _inactive.is_bottom():
            self.flag = 1

        join = _active.join(_inactive)
        self.state = join
        return self
