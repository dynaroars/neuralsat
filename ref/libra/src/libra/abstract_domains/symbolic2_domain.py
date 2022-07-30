"""
Symbolic Constant Propagation (Variant 2)
=========================================

:Authors: Caterina Urban
"""
from copy import deepcopy
from typing import List

from apronpy.coeff import PyMPQScalarCoeff
from apronpy.lincons0 import ConsTyp
from apronpy.scalar import PyMPQScalar
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.symbolic_domain import SymbolicState


rtype = TexprRtype.AP_RTYPE_REAL
rdir = TexprRdir.AP_RDIR_RND


def texpr_to_dict(texpr):

    def do(texpr0, env):
        if texpr0.discr == TexprDiscr.AP_TEXPR_CST:
            result = dict()
            t0 = '{}'.format(texpr0.val.cst)
            t1 = eval(t0)
            t2 = str(t1)
            t3 = float(t2)
            result['_'] = t3
            return result
        elif texpr0.discr == TexprDiscr.AP_TEXPR_DIM:
            result = dict()
            result['{}'.format(env.var_of_dim[texpr0.val.dim.value].decode('utf-8'))] = 1.0
            return result
        else:
            assert texpr0.discr == TexprDiscr.AP_TEXPR_NODE
            left = do(texpr0.val.node.contents.exprA.contents, env)
            op = texpr0.val.node.contents.op
            if texpr0.val.node.contents.exprB:
                right = do(texpr0.val.node.contents.exprB.contents, env)
            if op == TexprOp.AP_TEXPR_ADD:
                result = deepcopy(left)
                for var, val in right.items():
                    if var in result:
                        result[var] += val
                    else:
                        result[var] = val
                return result
            elif op == TexprOp.AP_TEXPR_SUB:
                result = deepcopy(left)
                for var, val in right.items():
                    if var in result:
                        result[var] -= val
                    else:
                        result[var] = -val
                return result
            elif op == TexprOp.AP_TEXPR_MUL:
                # print('multiplying')
                # print('left: ', left)
                # print('right: ', right)
                result = dict()
                if '_' in left and len(left) == 1:
                    for var, val in right.items():
                        result[var] = left['_'] * right[var]
                elif '_' in right and len(right) == 1:
                    for var, val in left.items():
                        result[var] = right['_'] * left[var]
                else:
                    assert False
                # print('result: ', result)
            elif op == TexprOp.AP_TEXPR_NEG:
                result = deepcopy(left)
                for var, val in result.items():
                    result[var] = -val
        return result

    texpr1 = texpr.texpr1.contents
    return do(texpr1.texpr0.contents, texpr1.env.contents)


def substitute_in_dict(todict, var, rhs):
    result = todict
    key = str(var)
    coeff = result[key]
    del result[key]
    for var, val in rhs.items():
        if var in result:
            result[var] += coeff * val
        else:
            result[var] = coeff * val
    return result


def dict_to_texpr(todict, env):
    texpr = PyTexpr1.cst(env, PyMPQScalarCoeff(PyMPQScalar(todict['_'])))
    for var, val in reversed(list(todict.items())):
        if var != '_':
            coeff = PyTexpr1.cst(env, PyMPQScalarCoeff(PyMPQScalar(val)))
            dim = PyTexpr1.var(env, PyVar(var))
            term = PyTexpr1.binop(TexprOp.AP_TEXPR_MUL, coeff, dim, TexprRtype.AP_RTYPE_REAL, TexprRdir.AP_RDIR_RND)
            texpr = PyTexpr1.binop(TexprOp.AP_TEXPR_ADD, term, texpr, TexprRtype.AP_RTYPE_REAL, TexprRdir.AP_RDIR_RND)
    return texpr


def print_state(environment, state):
    def var(dim):
        return environment.environment.contents.var_of_dim[dim].decode('utf-8')

    def itv(dim):
        bound = state.bound_variable(PyVar(var(dim)))
        interval = bound.interval.contents
        inf = '{}'.format(interval.inf.contents)
        lower = inf if inf != '-1/0' else '-inf'
        sup = '{}'.format(interval.sup.contents)
        upper = sup if sup != '1/0' else 'inf'
        return '[{}, {}]'.format(lower, upper)

    if state.is_bottom():
        return "⊥"
    env = environment.environment.contents
    result = ', '.join('{}: {}'.format(var(i), itv(i)) for i in range(env.intdim))
    result += ', '.join(
        '{} -> {}'.format(var(env.intdim + i), itv(env.intdim + i)) for i in range(env.realdim)
    )
    return result.replace('.0', '')


class Symbolic2State(SymbolicState):
    """Interval+Symbolic (Variant 2).

    .. document private methods
    .. automethod:: Symbolic2State._assign
    .. automethod:: Symbolic2State._assume
    .. automethod:: Symbolic2State._output
    .. automethod:: Symbolic2State._substitute

    """

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'Symbolic2State':
        array = list()
        assignments = dict()
        for lhs, expr in zip(left, right):
            rhs = texpr_to_dict(expr)
            for sym, val in self.symbols.values():
                rhs = substitute_in_dict(rhs, sym, val)
            assignments[str(lhs)] = (lhs, rhs)
            rhs = dict_to_texpr(rhs, self.environment)
            var = PyTexpr1.var(self.environment, lhs)
            binop = PyTexpr1.binop(TexprOp.AP_TEXPR_SUB, var, rhs, rtype, rdir)
            cond = PyTcons1.make(binop, ConsTyp.AP_CONS_EQ)
            array.append(cond)
        self.symbols = assignments
        self.state = self.state.meet(PyTcons1Array(array))
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'Symbolic2State':
        self.flag = None

        if active:      # we only do the active case (h >= 0)
            expr1 = PyTexpr1.var(self.environment, stmt)
            cond = PyTcons1.make(expr1, ConsTyp.AP_CONS_SUPEQ)
            abstract1 = self.domain(self.state.manager, self.environment, array=PyTcons1Array([cond]))
            self.state = self.state.meet(abstract1)
            self.flag = 1
            return self

        if inactive:    # we only do the inactive case (h < 0)
            zero = dict()
            zero['_'] = 0.0
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
            zero = dict()
            zero['_'] = 0.0
            self.symbols[str(stmt)] = (stmt, zero)

            self.flag = -1
        elif _inactive.is_bottom():
            self.flag = 1
        else:
            del self.symbols[str(stmt)]

        join = _active.join(_inactive)
        self.state = join
        return self
