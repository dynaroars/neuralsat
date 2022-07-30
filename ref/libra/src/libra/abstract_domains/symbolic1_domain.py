"""
Symbolic Constant Propagation (Variant 1)
=========================================

:Authors: Caterina Urban
"""
from copy import deepcopy
from typing import List

from apronpy.coeff import PyMPQScalarCoeff
from apronpy.lincons0 import ConsTyp
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from libra.abstract_domains.symbolic_domain import SymbolicState


rtype = TexprRtype.AP_RTYPE_REAL
rdir = TexprRdir.AP_RDIR_RND


class Symbolic1State(SymbolicState):
    """Interval+Symbolic (Variant 1).

    .. document private methods
    .. automethod:: Symbolic1State._assign
    .. automethod:: Symbolic1State._assume
    .. automethod:: Symbolic1State._output
    .. automethod:: Symbolic1State._substitute

    """

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'Symbolic1State':
        array = list()
        assignments = dict()
        for lhs, expr in zip(left, right):
            rhs = expr
            for sym, val in self.symbols.values():
                rhs = rhs.substitute(sym, val)
            assignments[str(lhs)] = (lhs, rhs)
            var = PyTexpr1.var(self.environment, lhs)
            binop = PyTexpr1.binop(TexprOp.AP_TEXPR_SUB, var, rhs, rtype, rdir)
            cond = PyTcons1.make(binop, ConsTyp.AP_CONS_EQ)
            array.append(cond)
        self.symbols = assignments
        self.state = self.state.meet(PyTcons1Array(array))
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'Symbolic1State':
        self.flag = None

        if active:      # we only do the active case (h >= 0)
            expr1 = PyTexpr1.var(self.environment, stmt)
            cond = PyTcons1.make(expr1, ConsTyp.AP_CONS_SUPEQ)
            abstract1 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond]))
            self.state = self.state.meet(abstract1)
            self.flag = 1
            return self

        if inactive:    # we only do the inactive case (h < 0)
            zero = PyTexpr1.cst(self.environment, PyMPQScalarCoeff(0.0))
            self.symbols[str(stmt)] = (stmt, zero)

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
            zero = PyTexpr1.cst(self.environment, PyMPQScalarCoeff(0.0))
            self.symbols[str(stmt)] = (stmt, zero)

            self.flag = -1
        elif _inactive.is_bottom():
            self.flag = 1
        else:
            del self.symbols[str(stmt)]

        join = _active.join(_inactive)
        self.state = join
        return self
