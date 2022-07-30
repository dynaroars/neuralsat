"""
Symbolic Constant Propagation (Variant 3)
=========================================

:Authors: Caterina Urban
"""
import time
from copy import deepcopy
from math import inf
from typing import Set, List, Dict, Tuple

from apronpy.manager import PyManager
from apronpy.texpr0 import TexprRtype, TexprRdir, TexprDiscr, TexprOp
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.abstract_domains.deeppoly_domain import IntervalLattice
from libra.abstract_domains.state import State
from libra.core.expressions import VariableIdentifier, Expression, BinaryBooleanOperation, \
    BinaryComparisonOperation, Literal, UnaryArithmeticOperation
from libra.core.utils import copy_docstring
from libra.abstract_domains.bounds_domain import BoundsDomain


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


class Symbolic3State(State, BoundsDomain):
    """Interval+Symbolic (Variant 3)

    .. document private methods
    .. automethod:: Symbolic3State._assign
    .. automethod:: Symbolic3State._assume
    .. automethod:: Symbolic3State._output
    .. automethod:: Symbolic3State._substitute

    """
    def __init__(self, inputs: Set[VariableIdentifier], precursory: State = None):
        super().__init__(precursory=precursory)
        self.inputs = {input.name for input in inputs}
        self.bounds: Dict[str, IntervalLattice] = dict()
        for input in self.inputs:
            self.bounds[input] = IntervalLattice(0, 1)
        self.symbols: Dict[str, Tuple[str, dict]] = dict()
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
    def _less_equal(self, other: 'Symbolic3State') -> bool:
        raise NotImplementedError(f"Call to _is_less_equal is unexpected!")

    @copy_docstring(State._join)
    def _join(self, other: 'Symbolic3State') -> 'Symbolic3State':
        for var in self.bounds:
            self.bounds[var].join(other.bounds[var])
        return self

    @copy_docstring(State._meet)
    def _meet(self, other: 'Symbolic3State') -> 'Symbolic3State':
        for var in self.bounds:
            self.bounds[var].meet(other.bounds[var])
        return self

    @copy_docstring(State._widening)
    def _widening(self, other: 'Symbolic3State') -> 'Symbolic3State':
        raise NotImplementedError(f"Call to _widening is unexpected!")

    @copy_docstring(State._assign)
    def _assign(self, left: Expression, right: Expression) -> 'Symbolic3State':
        raise NotImplementedError(f"Call to _assign is unexpected!")

    @copy_docstring(State._assume)
    def _assume(self, condition: Expression, bwd: bool = False) -> 'Symbolic3State':
        raise NotImplementedError(f"Call to _assume is unexpected!")

    def assume(self, condition, manager: PyManager = None, bwd: bool = False) -> 'Symbolic3State':
        if self.is_bottom():
            return self
        if isinstance(condition, tuple):
            condition = list(condition)
        if isinstance(condition, list):
            for feature, (lower, upper) in condition:
                self.bounds[feature.name].meet(IntervalLattice(lower, upper))
            return self
        elif isinstance(condition, BinaryBooleanOperation):
            if condition.operator == BinaryBooleanOperation.Operator.Or:
                right = deepcopy(self).assume(condition.right, bwd=bwd)
                return self.assume(condition.left, bwd=bwd).join(right)
            elif condition.operator == BinaryBooleanOperation.Operator.And:
                assert isinstance(condition.left, BinaryComparisonOperation)
                assert isinstance(condition.right, BinaryComparisonOperation)
                if isinstance(condition.left.left, Literal):
                    lower = eval(condition.left.left.val)
                else:
                    assert isinstance(condition.left.left, UnaryArithmeticOperation)
                    assert condition.left.left.operator == UnaryArithmeticOperation.Operator.Sub
                    assert isinstance(condition.left.left.expression, Literal)
                    lower = -eval(condition.left.left.expression.val)
                assert condition.left.operator == BinaryComparisonOperation.Operator.LtE
                assert isinstance(condition.left.right, VariableIdentifier)
                assert isinstance(condition.right.left, VariableIdentifier)
                assert condition.right.operator == BinaryComparisonOperation.Operator.LtE
                if isinstance(condition.right.right, Literal):
                    upper = eval(condition.right.right.val)
                else:
                    assert isinstance(condition.right.right, UnaryArithmeticOperation)
                    assert condition.right.right.operator == UnaryArithmeticOperation.Operator.Sub
                    assert isinstance(condition.right.right.expression, Literal)
                    upper = -eval(condition.right.right.expression.val)
                assert condition.left.right.name == condition.right.left.name
                self.bounds[condition.left.right.name].meet(IntervalLattice(lower, upper))
                return self
        elif isinstance(condition, BinaryComparisonOperation):
            if condition.operator == BinaryComparisonOperation.Operator.Gt:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = eval(condition.right.val)
                upper = inf
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
                return self
            elif condition.operator == BinaryComparisonOperation.Operator.LtE:
                assert isinstance(condition.left, VariableIdentifier)
                assert isinstance(condition.right, Literal)
                lower = -inf
                upper = eval(condition.right.val)
                self.bounds[condition.left.name].meet(IntervalLattice(lower, upper))
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
    def _substitute(self, left: Expression, right: Expression) -> 'Symbolic3State':
        raise NotImplementedError(f"Call to _substitute is unexpected!")

    def affine(self, left: List[PyVar], right: List[PyTexpr1]) -> 'Symbolic3State':
        if self.is_bottom():
            return self
        assignments = dict()
        for lhs, expr in zip(left, right):
            name = str(lhs)
            rhs = texpr_to_dict(expr)
            for sym, val in self.symbols.values():
                rhs = substitute_in_dict(rhs, sym, val)
            assignments[name] = (name, rhs)
            bound = evaluate(rhs, self.bounds)
            self.bounds[name] = IntervalLattice(bound.lower, bound.upper)
        self.symbols = assignments
        return self

    def relu(self, stmt: PyVar, active: bool = False, inactive: bool = False) -> 'Symbolic3State':
        if self.is_bottom():
            return self
        self.flag = None

        name = str(stmt)
        lattice: IntervalLattice = self.bounds[name]
        lower, upper = lattice.lower, lattice.upper
        if upper <= 0 or inactive:
            zero = dict()
            zero['_'] = 0.0
            self.symbols[name] = (name, zero)
            self.bounds[name] = IntervalLattice(0, 0)
            self.flag = -1
        elif 0 <= lower or active:
            if active and lower < 0:
                bounds = self.bounds[name]
                self.bounds[name] = bounds.meet(IntervalLattice(0, upper))
                del self.symbols[name]
            self.flag = 1
        else:
            _active, _inactive = deepcopy(self.bounds), deepcopy(self.bounds)
            _active[name] = _active[name].meet(IntervalLattice(0, upper))
            _inactive[name] = _inactive[name].meet(IntervalLattice(0, 0))

            if any(element.is_bottom() for element in _active.values()):
                zero = dict()
                zero['_'] = 0.0
                self.symbols[name] = (name, zero)
                self.flag = -1
            elif any(element.is_bottom() for element in _inactive.values()):
                self.flag = 1
            else:
                del self.symbols[name]
                self.flag = None

            join = dict()
            for variable, itv in _active.items():
                join[variable] = itv.join(_inactive[variable])
            self.bounds[name] = join[name].meet(IntervalLattice(0, upper))
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

    def get_bounds(self, var_name):
        return self.bounds[var_name]

    def resize_bounds(self, var_name, new_bounds):
        self.bounds[var_name] = IntervalLattice(new_bounds.lower, new_bounds.upper)
