"""
Bias Abstract Domain
====================

Disjunctive relational abstract domain to be used for **algorithmic bias analysis**.

:Authors: Caterina Urban
"""
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from itertools import chain
from math import inf
from typing import Set, List, Dict

from apronpy.coeff import PyMPQScalarCoeff
from apronpy.environment import PyEnvironment
from apronpy.lincons1 import PyLincons1Array
from apronpy.manager import PyManager
from apronpy.polka import PyPolka
from apronpy.scalar import PyMPQScalar
from apronpy.tcons1 import PyTcons1Array, PyTcons1
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier, Expression, BinaryComparisonOperation, \
    BinaryBooleanOperation, Lyra2APRON, \
    NegationFreeExpression, Literal
from libra.core.utils import copy_docstring
from libra.abstract_domains.bounds_domain import BoundsDomain


class IntervalLattice:

    class Kind(Enum):
        TOP = 3
        DEFAULT = 2
        BOTTOM = 1

    def __init__(self, lower=-inf, upper=inf):
        super().__init__()
        self._kind = IntervalLattice.Kind.DEFAULT
        if lower <= upper and lower != inf and upper != -inf:      # the interval is not empty
            self._lower = lower
            self._upper = upper
        else:                   # the interval is empty
            self.bottom()

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind: 'IntervalLattice.Kind'):
        self._kind = kind

    @property
    def lower(self):
        if self.is_bottom():
            return None
        return self._lower

    @property
    def upper(self):
        if self.is_bottom():
            return None
        return self._upper

    def __repr__(self):
        if self.is_bottom():
            return "⊥"
        return f"[{self.lower}, {self.upper}]"

    def bottom(self):
        self.kind = IntervalLattice.Kind.BOTTOM
        return self

    def top(self) -> 'IntervalLattice':
        self._replace(type(self)())
        return self

    def is_bottom(self) -> bool:
        return self.kind == IntervalLattice.Kind.BOTTOM

    def is_top(self) -> bool:
        return self.lower == -inf and self.upper == inf

    def _less_equal(self, other: 'IntervalLattice') -> bool:
        """``[a, b] ⊑ [c, d]`` if and only if ``c <= a`` and ``b <= d``."""
        return other.lower <= self.lower and self.upper <= other.upper

    def _join(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """``[a, b] ⊔ [c, d] = [min(a,c), max(b,d)]``."""
        lower = min(self.lower, other.lower)
        upper = max(self.upper, other.upper)
        return self._replace(type(self)(lower, upper))

    def join(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """Least upper bound between lattice elements."""
        if self.is_bottom() or other.is_top():
            return self._replace(other)
        elif other.is_bottom() or self.is_top():
            return self
        else:
            return self._join(other)

    def _meet(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """``[a, b] ⊓ [c, d] = [max(a,c), min(b,d)]``."""
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower <= upper:
            return self._replace(type(self)(lower, upper))
        return self.bottom()

    def meet(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """Greatest lower bound between lattice elements.

        :param other: other lattice element
        :return: current lattice element modified to be the greatest lower bound

        """
        if self.is_top() or other.is_bottom():
            return self._replace(other)
        elif other.is_top() or self.is_bottom():
            return self
        else:
            return self._meet(other)

    # arithmetic operations

    def _neg(self) -> 'IntervalLattice':
        """``- [a, b] = [-b, -a]``."""
        lower = 0 - self.upper
        upper = 0 - self.lower
        return self._replace(type(self)(lower, upper))

    def _add(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """``[a, b] + [c, d] = [a + c, b + d]``."""
        lower = 0 + self.lower + other.lower
        upper = 0 + self.upper + other.upper
        return self._replace(type(self)(lower, upper))

    def _sub(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """``[a, b] - [c, d] = [a - d, b - c]``."""
        lower = 0 + self.lower - other.upper
        upper = 0 + self.upper - other.lower
        return self._replace(type(self)(lower, upper))

    def _mult(self, other: 'IntervalLattice') -> 'IntervalLattice':
        """``[a, b] * [c, d] = [min(a*c, a*d, b*c, b*d), max(a*c, a*d, b*c, b*d)]``."""
        ac = 0 if self.lower == 0 or other.lower == 0 else 1 * self.lower * other.lower
        ad = 0 if self.lower == 0 or other.upper == 0 else 1 * self.lower * other.upper
        bc = 0 if self.upper == 0 or other.lower == 0 else 1 * self.upper * other.lower
        bd = 0 if self.upper == 0 or other.upper == 0 else 1 * self.upper * other.upper
        lower = min(ac, ad, bc, bd)
        upper = max(ac, ad, bc, bd)
        return self._replace(type(self)(lower, upper))

    def _replace(self, other: 'IntervalLattice') -> 'IntervalLattice':
        self.__dict__.update(other.__dict__)
        return self


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


def evaluate(dictionary, bounds):
    result = IntervalLattice(0, 0)
    for var, val in dictionary.items():
        coeff = IntervalLattice(val, val)
        if var != '_':
            result = result._add(coeff._mult(bounds[var]))
        else:
            result = result._add(coeff)
    return result

# def substitute_in_dict(todict, var, rhs):
#     result = todict
#     key = str(var)
#     coeff = result[key]
#     del result[key]
#     for var, val in rhs.items():
#         if var in result:
#             result[var] += coeff * val
#         else:
#             result[var] = coeff * val
#     return result


# def dict_to_texpr(todict, env):
#     texpr = PyTexpr1.cst(env, PyMPQScalarCoeff(PyMPQScalar(todict['_'])))
#     for var, val in reversed(list(todict.items())):
#         if var != '_':
#             coeff = PyTexpr1.cst(env, PyMPQScalarCoeff(PyMPQScalar(val)))
#             dim = PyTexpr1.var(env, PyVar(var))
#             term = PyTexpr1.binop(TexprOp.AP_TEXPR_MUL, coeff, dim, TexprRtype.AP_RTYPE_REAL, TexprRdir.AP_RDIR_RND)
#             texpr = PyTexpr1.binop(TexprOp.AP_TEXPR_ADD, term, texpr, TexprRtype.AP_RTYPE_REAL, TexprRdir.AP_RDIR_RND)
#     return texpr


class DeepPolyState(State, BoundsDomain):
    """DeepPoly [Singh et al. POPL2019] state.

    .. document private methods
    .. automethod:: DeepPolyState._assign
    .. automethod:: DeepPolyState._assume
    .. automethod:: DeepPolyState._output
    .. automethod:: DeepPolyState._substitute

    """
    def __init__(self, inputs: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        self.inputs = {input.name for input in inputs}
        # self.variables: Dict[str, VariableIdentifier] = dict()
        # for variable in variables:
        #     self.variables[variable.name] = variable
        self.bounds = dict()
        for input in self.inputs:
            self.bounds[input] = IntervalLattice(0, 1)
        self.poly = dict()
        for input in self.inputs:
            zero: Dict[str, float] = dict()
            zero['_'] = 0.0
            one: Dict[str, float] = dict()
            one['_'] = 1.0
            self.poly[input] = (zero, one)
        self.flag = None

    @copy_docstring(State.bottom)
    def bottom(self):
        for var in self.bounds:
            self.bounds[var].bottom()
        return self

    @copy_docstring(State.top)
    def top(self):
        for var in self.bounds:
            self.bounds[var].top()
        return self

    def __repr__(self):
        items = sorted(self.bounds.items(), key=lambda x: x[0])
        return ", ".join("{} -> {}".format(variable, value) for variable, value in items)

    @copy_docstring(State.is_bottom)
    def is_bottom(self) -> bool:
        return any(element.is_bottom() for element in self.bounds.values())

    @copy_docstring(State.is_top)
    def is_top(self) -> bool:
        return all(element.is_top() for element in self.bounds.values())

    @copy_docstring(State._less_equal)
    def _less_equal(self, other: 'DeepPolyState') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'DeepPolyState') -> 'DeepPolyState':
        for var in self.bounds:
            self.bounds[var].join(other.bounds[var])
            bounds = self.bounds[var]
            inf: Dict[str, float] = dict()
            inf['_'] = bounds.lower
            sup: Dict[str, float] = dict()
            sup['_'] = bounds.upper
            self.poly[var] = (inf, sup)
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'DeepPolyState') -> 'DeepPolyState':
        for var in self.bounds:
            self.bounds[var].meet(other.bounds[var])
            bounds = self.bounds[var]
            inf: Dict[str, float] = dict()
            inf['_'] = bounds.lower
            sup: Dict[str, float] = dict()
            sup['_'] = bounds.upper
            self.poly[var] = (inf, sup)
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'DeepPolyState') -> 'DeepPolyState':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'DeepPolyState':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'DeepPolyState':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'DeepPolyState':
        if self.is_bottom():
            return self
        if isinstance(condition, tuple):
            condition = list(condition)
        if isinstance(condition, list):
            for feature, (lower, upper) in condition:
                self.bounds[feature.name].meet(IntervalLattice(lower, upper))
                bounds = self.bounds[feature.name]
                inf: Dict[str, float] = dict()
                inf['_'] = bounds.lower
                sup: Dict[str, float] = dict()
                sup['_'] = bounds.upper
                self.poly[feature.name] = (inf, sup)
            return self
        elif isinstance(condition, BinaryBooleanOperation):
            if condition.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self).assume(condition.right, bwd=bwd)
                return self.assume(condition.left, bwd=bwd).join(right)
            elif condition.operator == BinaryBooleanOperation.Operator.And:
                assert isinstance(condition.left, BinaryComparisonOperation)
                assert isinstance(condition.right, BinaryComparisonOperation)
                assert isinstance(condition.left.left, Literal)
                assert condition.left.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.left.right, VariableIdentifier)
                assert isinstance(condition.right.left, VariableIdentifier)
                assert condition.right.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.right.right, Literal)
                lower = eval(condition.left.left.val)
                upper = eval(condition.right.right.val)
                assert condition.left.right.name == condition.right.left.name
                self.bounds[condition.left.right.name].meet(IntervalLattice(lower, upper))
                bounds = self.bounds[condition.left.right.name]
                inf: Dict[str, float] = dict()
                inf['_'] = bounds.lower
                sup: Dict[str, float] = dict()
                sup['_'] = bounds.upper
                self.poly[condition.right.left.name] = (inf, sup)
                return self
        elif isinstance(condition, BinaryComparisonOperation):
            if condition.operator == BinaryComparisonOperation.Operator.Gt:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = eval(condition.right.val)
                upper = 1
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                bounds = self.bounds[condition.left.name]
                inf: Dict[str, float] = dict()
                inf['_'] = bounds.lower
                sup: Dict[str, float] = dict()
                sup['_'] = bounds.upper
                self.poly[condition.left.name] = (inf, sup)
                return self
            elif condition.operator == BinaryComparisonOperation.Operator.LtE:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = 0
                upper = eval(condition.right.val)
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                bounds = self.bounds[condition.left.name]
                inf: Dict[str, float] = dict()
                inf['_'] = bounds.lower
                sup: Dict[str, float] = dict()
                sup['_'] = bounds.upper
                self.poly[condition.left.name] = (inf, sup)
                return self
        # elif isinstance(condition, PyTcons1):
        #     abstract1 = self.domain(manager, self.environment, array=PyTcons1Array([condition]))
        #     self.state = self.state.meet(abstract1)
        #     return self
        elif isinstance(condition, set):
            assert len(condition) == 1
            self.assume(condition.pop(), bwd=bwd)
            return self
        raise NotImplementedError(f"Assumption of {condition.__class__.__name__} is unsupported!")

    @copy_docstring(State._substitute)
    def _substitute(self, left: Expression, right: Expression) -> 'DeepPolyState':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def forget(self, variables: List[VariableIdentifier]) -> 'DeepPolyState':
        raise NotImplementedError(f"Call to _forget is unexpected!")

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'DeepPolyState':
        if self.is_bottom():
            return self
        for lhs, expr in zip(left, right):
            name = str(lhs)
            rhs = texpr_to_dict(expr)
            _inf, inf = deepcopy(rhs), deepcopy(rhs)
            _sup, sup = deepcopy(rhs), deepcopy(rhs)
            self.poly[name] = (_inf, _sup)
            while any(variable in inf and variable not in self.inputs for variable in self.poly):
                for variable in self.poly:
                    if variable in inf and variable not in self.inputs:  # should be replaced
                        coeff = inf[variable]
                        if coeff > 0:
                            replacement = self.poly[variable][0]
                        elif coeff < 0:
                            replacement = self.poly[variable][1]
                        else:  # coeff == 0
                            replacement = dict()
                            replacement['_'] = 0.0
                        del inf[variable]
                        for var, val in replacement.items():
                            if var in inf:
                                inf[var] += coeff * val
                            else:
                                inf[var] = coeff * val
            while any(variable in sup and variable not in self.inputs for variable in self.poly):
                for variable in self.poly:
                    if variable in sup and variable not in self.inputs:  # should be replaced
                        coeff = sup[variable]
                        if coeff > 0:
                            replacement = self.poly[variable][1]
                        elif coeff < 0:
                            replacement = self.poly[variable][0]
                        else:  # coeff == 0
                            replacement = dict()
                            replacement['_'] = 0.0
                        del sup[variable]
                        for var, val in replacement.items():
                            if var in sup:
                                sup[var] += coeff * val
                            else:
                                sup[var] = coeff * val
            lower = evaluate(inf, self.bounds)
            upper = evaluate(sup, self.bounds)
            self.bounds[name] = IntervalLattice(lower.lower, upper.upper)
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'DeepPolyState':
        if self.is_bottom():
            return self
        name = str(stmt)
        lattice: IntervalLattice = self.bounds[name]
        lower, upper = lattice.lower, lattice.upper
        if upper <= 0 or inactive:
            # l_j = u_j = 0
            self.bounds[name] = IntervalLattice(0, 0)
            # 0 <= x_j <= 0
            zero: Dict[str, float] = dict()
            zero['_'] = 0.0
            self.poly[name] = (zero, zero)
            self.flag = -1
        elif 0 <= lower or active:
            if active and lower < 0:
                self.bounds[name] = IntervalLattice(0, upper)
                sup = self.poly[name][1]
                zero: Dict[str, float] = dict()
                zero['_'] = 0.0
                self.poly[name] = (zero, sup)
            self.flag = 1
        elif upper <= -lower:   # case (b) in Fig. 4, equation (3)
            # l_j = 0 && u_j = u_i
            self.bounds[name] = IntervalLattice(0, upper)
            # 0 <= x_j <= u_i * (x_i - l_i) / (u_i - l_i)
            zero: Dict[str, float] = dict()
            zero['_'] = 0.0
            #
            m = upper / (upper - lower)
            if m > 0:
                sup = self.poly[name][1]
            elif m < 0:
                sup = self.poly[name][0]
            else:   # m == 0
                sup = dict()
                sup['_'] = 0.0
            for var, val in sup.items():
                sup[var] = m * val
            q = - upper * lower / (upper - lower)
            sup['_'] = sup['_'] + q
            #
            self.poly[name] = (zero, sup)
            self.flag = None
        else:   # case (c) in Fig. 4, equation (4)
            # l_j = l_i && u_j = u_i
            self.bounds[name] = IntervalLattice(lower, upper)
            # x_i <= x_j <= u_i * (x_i - l_i) / (u_i - l_i)
            inf = deepcopy(self.poly[name][0])
            #
            m = upper / (upper - lower)
            if m > 0:
                sup = self.poly[name][1]
            elif m < 0:
                sup = self.poly[name][0]
            else:  # m == 0
                sup = dict()
                sup['_'] = 0.0
            for var, val in sup.items():
                sup[var] = m * val
            q = - upper * lower / (upper - lower)
            sup['_'] = sup['_'] + q
            #
            self.poly[name] = (inf, sup)
            self.flag = None
        return self

    def outcome(self, outcomes: Set[VariableIdentifier]):
        found = None
        if self.is_bottom():
            found = '⊥'
        else:
            for chosen in outcomes:
                outcome = self.bounds[chosen.name]
                lower = outcome.lower
                unique = True
                remaining = outcomes - {chosen}
                for discarded in remaining:
                    alternative = self.bounds[discarded.name]
                    upper = alternative.upper
                    if lower <= upper:
                        unique = False
                        break
                if unique:
                    found = chosen
                    break
        return found

    _negation_free = NegationFreeExpression()
    _lyra2apron = Lyra2APRON()

    def get_bounds(self, var_name):
        return self.bounds[var_name]

    def resize_bounds(self, var_name, new_bounds):
        self.bounds[var_name] = IntervalLattice(new_bounds.lower, new_bounds.upper)