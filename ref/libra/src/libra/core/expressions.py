"""
Expressions
===========

Libra's internal representation of Python expressions.

:Authors: Caterina Urban
"""

from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Set

from apronpy.coeff import PyMPQScalarCoeff, PyMPQIntervalCoeff
from apronpy.interval import PyMPQInterval
from apronpy.lincons0 import ConsTyp
from apronpy.tcons1 import PyTcons1
from apronpy.texpr0 import TexprOp, TexprRtype, TexprRdir
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar

from libra.core.utils import copy_docstring


class Expression(metaclass=ABCMeta):
    """Expression representation."""

    def __init__(self):
        """Expression construction."""

    @abstractmethod
    def __eq__(self, other: 'Expression'):
        """Expression equality.

        :param other: other expression to compare
        :return: whether the expression equality holds
        """

    @abstractmethod
    def __hash__(self):
        """Expression hash representation.

        :return: hash value representing the expression
        """

    def __ne__(self, other: 'Expression'):
        return not (self == other)

    @abstractmethod
    def __str__(self):
        """Expression string representation.

        :return: string representing the expression
        """

    def ids(self) -> Set['VariableIdentifier']:
        """Identifiers that appear in the expression.

        :return: set of identifiers that appear in the expression
        """
        ids = set()
        for expr in _walk(self):
            if isinstance(expr, VariableIdentifier):
                ids.add(expr)
        return ids


def _iter_child_exprs(expr: Expression):
    """
    Yield all direct child expressions of ``expr``,
    that is, all fields that are expressions
    and all items of fields that are lists of expressions.
    """
    for _, field in expr.__dict__.items():
        if isinstance(field, Expression):
            yield field
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, Expression):
                    yield item


def _walk(expr: Expression):
    """
    Recursively yield all expressions in an expression tree
    starting at ``expr`` (including ``expr`` itself),
    in no specified order.
    """
    from collections import deque
    todo = deque([expr])
    while todo:
        expr = todo.popleft()
        todo.extend(_iter_child_exprs(expr))
        yield expr


# noinspection PyPep8Naming
class ExpressionVisitor(metaclass=ABCMeta):
    """
    An expression visitor base class that walks the expression tree and calls a
    visitor function for every expression found.  This function may return a value
    which is forwarded by the `visit` method.

    Subclasses are meant to implement the visitor functions.
    The visitor function for an expression is ``'visit_'`` +
    class name of the expression.  So a `Literal` expression visit function would
    be `visit_Literal`.  If no visitor function exists for an expression
    a `NotImplementedError` is raised.

    Adapted from `ast.py`.
    """

    def visit(self, expr, *args, **kwargs):
        """Visit of an expression."""
        method = 'visit_' + expr.__class__.__name__
        if hasattr(self, method):
            return getattr(self, method)(expr, *args, **kwargs)
        error = f"Missing visitor for {expr.__class__.__name__} in {self.__class__.__qualname__}!"
        raise NotImplementedError(error)

    @abstractmethod
    def visit_Literal(self, expr: 'Literal'):
        """Visit of a literal expression."""

    @abstractmethod
    def visit_VariableIdentifier(self, expr: 'VariableIdentifier'):
        """Visit of a variable identifier."""

    @abstractmethod
    def visit_Input(self, expr: 'Input'):
        """Visit of an input call expression."""

    @abstractmethod
    def visit_UnaryArithmeticOperation(self, expr: 'UnaryArithmeticOperation'):
        """Visit of a unary arithmetic operation."""

    @abstractmethod
    def visit_UnaryBooleanOperation(self, expr: 'UnaryBooleanOperation'):
        """Visit of a unary boolean operation."""

    @abstractmethod
    def visit_BinaryArithmeticOperation(self, expr: 'BinaryArithmeticOperation'):
        """Visit of a binary arithmetic operation."""

    @abstractmethod
    def visit_BinaryBooleanOperation(self, expr: 'BinaryBooleanOperation'):
        """Visit of a binary boolean operation."""

    @abstractmethod
    def visit_BinaryComparisonOperation(self, expr: 'BinaryComparisonOperation'):
        """Visit of a binary comparison operation."""

    def generic_visit(self, expr, *args, **kwargs):
        raise ValueError(
            f"{self.__class__.__qualname__} does not support generic visit of expressions! "
            f"Define handling for a {expr.__class__.__name__} expression explicitly!")


class NegationFreeExpression(ExpressionVisitor):
    """Expression visitor that removes negations using De Morgan's law."""

    @copy_docstring(ExpressionVisitor.visit_Literal)
    def visit_Literal(self, expr: 'Literal', invert=False):
        return expr    # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_VariableIdentifier)
    def visit_VariableIdentifier(self, expr: 'VariableIdentifier', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_Input)
    def visit_Input(self, expr: 'Input', invert=False):
        return expr

    @copy_docstring(ExpressionVisitor.visit_UnaryArithmeticOperation)
    def visit_UnaryArithmeticOperation(self, expr: 'UnaryArithmeticOperation', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_UnaryBooleanOperation)
    def visit_UnaryBooleanOperation(self, expr: 'UnaryBooleanOperation', invert=False):
        if expr.operator == UnaryBooleanOperation.Operator.Neg:
            return self.visit(expr.expression, invert=not invert)
        raise ValueError(f"Unary boolean operator {expr.operator} is unsupported!")

    @copy_docstring(ExpressionVisitor.visit_BinaryArithmeticOperation)
    def visit_BinaryArithmeticOperation(self, expr: 'BinaryArithmeticOperation', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_BinaryBooleanOperation)
    def visit_BinaryBooleanOperation(self, expr: 'BinaryBooleanOperation', invert=False):
        left = self.visit(expr.left, invert)
        operator = expr.operator.reverse_operator() if invert else expr.operator
        right = self.visit(expr.right, invert)
        return BinaryBooleanOperation(left, operator, right)

    @copy_docstring(ExpressionVisitor.visit_BinaryComparisonOperation)
    def visit_BinaryComparisonOperation(self, expr: 'BinaryComparisonOperation', invert=False):
        left = expr.left
        operator = expr.operator.reverse_operator() if invert else expr.operator
        right = expr.right
        return BinaryComparisonOperation(left, operator, right)


class NegationFreeNormalExpression(ExpressionVisitor):
    """
    An expression visitor that:

    1. removes negations using De Morgan's law, and

    2. puts all boolean comparison operations with ``=``, ``!=``, ``<``, ``<=``, ``>``, and ``>=``
       in the normal form ``expr <= 0``.
    """

    @copy_docstring(ExpressionVisitor.visit_Literal)
    def visit_Literal(self, expr: 'Literal', invert=False):
        return expr    # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_VariableIdentifier)
    def visit_VariableIdentifier(self, expr: 'VariableIdentifier', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_Input)
    def visit_Input(self, expr: 'Input', invert=False):
        return expr

    @copy_docstring(ExpressionVisitor.visit_UnaryArithmeticOperation)
    def visit_UnaryArithmeticOperation(self, expr: 'UnaryArithmeticOperation', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_UnaryBooleanOperation)
    def visit_UnaryBooleanOperation(self, expr: 'UnaryBooleanOperation', invert=False):
        if expr.operator == UnaryBooleanOperation.Operator.Neg:
            return self.visit(expr.expression, invert=not invert)
        raise ValueError(f"Unary boolean operator {expr.operator} is unsupported!")

    @copy_docstring(ExpressionVisitor.visit_BinaryArithmeticOperation)
    def visit_BinaryArithmeticOperation(self, expr: 'BinaryArithmeticOperation', invert=False):
        return expr     # nothing to be done

    @copy_docstring(ExpressionVisitor.visit_BinaryBooleanOperation)
    def visit_BinaryBooleanOperation(self, expr: 'BinaryBooleanOperation', invert=False):
        left = self.visit(expr.left, invert)
        operator = expr.operator.reverse_operator() if invert else expr.operator
        right = self.visit(expr.right, invert)
        return BinaryBooleanOperation(left, operator, right)

    @copy_docstring(ExpressionVisitor.visit_BinaryComparisonOperation)
    def visit_BinaryComparisonOperation(self, expr: 'BinaryComparisonOperation', invert=False):
        left = expr.left
        operator = expr.operator.reverse_operator() if invert else expr.operator
        right = expr.right
        if operator == BinaryComparisonOperation.Operator.Eq:
            # left = right -> left - right <= 0 && right - left <= 0
            zero = Literal("0")
            minus = BinaryArithmeticOperation.Operator.Sub
            operator = BinaryComparisonOperation.Operator.LtE
            expr1 = BinaryArithmeticOperation(left, minus, right)
            expr1 = BinaryComparisonOperation(expr1, operator, zero)
            expr2 = BinaryArithmeticOperation(right, minus, left)
            expr2 = BinaryComparisonOperation(expr2, operator, zero)
            conjunction = BinaryBooleanOperation.Operator.And
            return BinaryBooleanOperation(expr1, conjunction, expr2)
        elif operator == BinaryComparisonOperation.Operator.NotEq:
            # left != right -> left - (right - 1) <= 0 || right - (left - 1) <= 0
            zero = Literal("0")
            one = Literal("1")
            minus = BinaryArithmeticOperation.Operator.Sub
            operator = BinaryComparisonOperation.Operator.LtE
            expr1 = BinaryArithmeticOperation(right, minus, one)
            expr1 = BinaryArithmeticOperation(left, minus, expr1)
            expr1 = BinaryComparisonOperation(expr1, operator, zero)
            expr2 = BinaryArithmeticOperation(left, minus, one)
            expr2 = BinaryArithmeticOperation(right, minus, expr2)
            expr2 = BinaryComparisonOperation(expr2, operator, zero)
            disjunction = BinaryBooleanOperation.Operator.Or
            return BinaryBooleanOperation(expr1, disjunction, expr2)
        elif operator == BinaryComparisonOperation.Operator.Lt:
            # left < right -> left - (right - 1) <= 0
            zero = Literal("0")
            one = Literal("1")
            minus = BinaryArithmeticOperation.Operator.Sub
            right = BinaryArithmeticOperation(right, minus, one)
            left = BinaryArithmeticOperation(left, minus, right)
            operator = BinaryComparisonOperation.Operator.LtE
            return BinaryComparisonOperation(left, operator, zero)
        elif operator == BinaryComparisonOperation.Operator.LtE:
            # left <= right -> left - right <= 0
            zero = Literal("0")
            minus = BinaryArithmeticOperation.Operator.Sub
            left = BinaryArithmeticOperation(left, minus, right)
            operator = BinaryComparisonOperation.Operator.LtE
            return BinaryComparisonOperation(left, operator, zero)
        elif operator == BinaryComparisonOperation.Operator.Gt:
            # left > right -> right - (left - 1) <= 0
            zero = Literal("0")
            one = Literal("1")
            minus = BinaryArithmeticOperation.Operator.Sub
            left = BinaryArithmeticOperation(left, minus, one)
            right = BinaryArithmeticOperation(right, minus, left)
            operator = BinaryComparisonOperation.Operator.LtE
            return BinaryComparisonOperation(right, operator, zero)
        elif operator == BinaryComparisonOperation.Operator.GtE:
            # left >= right -> right - left <= 0
            zero = Literal("0")
            minus = BinaryArithmeticOperation.Operator.Sub
            right = BinaryArithmeticOperation(right, minus, left)
            operator = BinaryComparisonOperation.Operator.LtE
            return BinaryComparisonOperation(right, operator, zero)
        elif operator == BinaryComparisonOperation.Operator.In:
            return BinaryComparisonOperation(left, operator, right)
        elif operator == BinaryComparisonOperation.Operator.NotIn:
            return BinaryComparisonOperation(left, operator, right)
        raise ValueError(f"Boolean comparison operator {expr} is unsupported!")


class Lyra2APRON(ExpressionVisitor):
    """Expression visitor that yields APRON tree expressions and constraints."""

    @copy_docstring(ExpressionVisitor.visit_Literal)
    def visit_Literal(self, expr: 'Literal', environment=None, usub=False) -> PyTexpr1:
        if usub:
            cst = PyMPQScalarCoeff(-float(expr.val))
        else:
            cst = PyMPQScalarCoeff(float(expr.val))
        return PyTexpr1.cst(environment, cst)

    @copy_docstring(ExpressionVisitor.visit_VariableIdentifier)
    def visit_VariableIdentifier(self, expr, environment=None, usub=False) -> PyTexpr1:
        assert not usub
        return PyTexpr1.var(environment, PyVar(expr.name))

    @copy_docstring(ExpressionVisitor.visit_Input)
    def visit_Input(self, expr, environment=None, usub=False) -> PyTexpr1:
        assert not usub
        return PyTexpr1.cst(environment, PyMPQIntervalCoeff(PyMPQInterval.top()))

    @copy_docstring(ExpressionVisitor.visit_UnaryArithmeticOperation)
    def visit_UnaryArithmeticOperation(self, expr, environment=None, usub=False) -> PyTexpr1:
        usub = expr.operator == UnaryArithmeticOperation.Operator.Sub
        return self.visit(expr.expression, environment, usub)

    @copy_docstring(ExpressionVisitor.visit_UnaryBooleanOperation)
    def visit_UnaryBooleanOperation(self, expr, environment=None, usub=False):
        raise ValueError(f"Conversion of {expr} to APRON is unsupported!")

    @copy_docstring(ExpressionVisitor.visit_BinaryArithmeticOperation)
    def visit_BinaryArithmeticOperation(self, expr, environment=None, usub=False) -> PyTexpr1:
        assert not usub
        expr1 = self.visit(expr.left, environment)
        expr2 = self.visit(expr.right, environment)
        op2op = {
            BinaryArithmeticOperation.Operator.Add: TexprOp.AP_TEXPR_ADD,
            BinaryArithmeticOperation.Operator.Sub: TexprOp.AP_TEXPR_SUB,
            BinaryArithmeticOperation.Operator.Mult: TexprOp.AP_TEXPR_MUL,
            BinaryArithmeticOperation.Operator.Div: TexprOp.AP_TEXPR_DIV
        }
        op = op2op[expr.operator]
        return PyTexpr1.binop(op, expr1, expr2, TexprRtype.AP_RTYPE_REAL, TexprRdir.AP_RDIR_RND)

    @copy_docstring(ExpressionVisitor.visit_BinaryBooleanOperation)
    def visit_BinaryBooleanOperation(self, expr, environment=None, usub=False):
        raise ValueError(f"Conversion of {expr} to APRON is unsupported!")

    @copy_docstring(ExpressionVisitor.visit_BinaryComparisonOperation)
    def visit_BinaryComparisonOperation(self, expr, environment=None, usub=False) -> PyTcons1:
        assert not usub
        left = expr.left
        right = expr.right
        sub = BinaryArithmeticOperation.Operator.Sub
        if expr.operator == BinaryComparisonOperation.Operator.GtE:
            expr = self.visit(BinaryArithmeticOperation(left, sub, right), environment)
            return PyTcons1.make(expr, ConsTyp.AP_CONS_SUPEQ)
        elif expr.operator == BinaryComparisonOperation.Operator.Gt:
            expr = self.visit(BinaryArithmeticOperation(left, sub, right), environment)
            return PyTcons1.make(expr, ConsTyp.AP_CONS_SUP)
        elif expr.operator == BinaryComparisonOperation.Operator.LtE:
            expr = self.visit(BinaryArithmeticOperation(right, sub, left), environment)
            return PyTcons1.make(expr, ConsTyp.AP_CONS_SUPEQ)
        elif expr.operator == BinaryComparisonOperation.Operator.Lt:
            expr = self.visit(BinaryArithmeticOperation(right, sub, left), environment)
            return PyTcons1.make(expr, ConsTyp.AP_CONS_SUP)


"""
Atomic Expressions
https://docs.python.org/3.4/reference/expressions.html#atoms
"""


class Literal(Expression):
    """Literal representation.

    https://docs.python.org/3.4/reference/expressions.html#literals
    """

    def __init__(self, val: str):
        """Literal construction.

        :param val: value of the literal
        """
        super().__init__()
        self._val = val

    @property
    def val(self):
        return self._val

    def __eq__(self, other: 'Literal'):
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)

    def __str__(self):
        return f"{self.val}"


class VariableIdentifier(Expression, metaclass=ABCMeta):
    """Variable identifier representation.

    https://docs.python.org/3.4/reference/expressions.html#atom-identifiers
    """

    def __init__(self, name: str):
        """Identifier construction.

        :param name: name of the identifier
        """
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name

    def __eq__(self, other: 'VariableIdentifier'):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "{0.name}".format(self)


"""
Primary Expressions
https://docs.python.org/3.4/reference/expressions.html#primaries
"""


class Call(Expression, metaclass=ABCMeta):
    """Call representation.

    https://docs.python.org/3.4/reference/expressions.html#calls
    """


class Input(Call):
    """Input call representation."""

    def __eq__(self, other: 'Input'):
        return True

    def __hash__(self):
        return hash('input()')

    def __str__(self):
        return "input()"


"""
Operation Expressions
"""


class Operation(Expression, metaclass=ABCMeta):
    """Operation representation."""


"""
Unary Operation Expressions
"""


class UnaryOperation(Operation):
    """Unary operation representation."""
    class Operator(IntEnum):
        """Unary operator representation."""

        @abstractmethod
        def __str__(self):
            """Unary operator string representation.

            :return: string representing the operator
            """

    def __init__(self, operator: Operator, expression: Expression):
        """Unary operation construction.

        :param operator: operator of the operation
        :param expression: expression of the operation
        """
        super().__init__()
        self._operator = operator
        self._expression = expression

    @property
    def operator(self):
        return self._operator

    @property
    def expression(self):
        return self._expression

    def __eq__(self, other: 'UnaryOperation'):
        operator = self.operator == other.operator
        expression = self.expression == other.expression
        return operator and expression

    def __hash__(self):
        return hash((self.operator, self.expression))

    def __str__(self):
        expr_string = str(self.expression)
        if isinstance(self.expression, Operation):
            expr_string = f"({expr_string})"
        return f"{str(self.operator)}{expr_string}"


class UnaryArithmeticOperation(UnaryOperation):
    """Unary arithmetic operation expression representation.

    https://docs.python.org/3.4/reference/expressions.html#unary-arithmetic-and-bitwise-operations
    """

    class Operator(UnaryOperation.Operator):
        """Unary arithmetic operator representation."""
        Add = 1
        Sub = -1

        def __str__(self):
            if self.value == 1:
                return "+"
            elif self.value == -1:
                return "-"

    def __init__(self, operator: Operator, expression: Expression):
        """Unary arithmetic operation expression representation.

        :param operator: operator of the operation
        :param expression: expression of the operation
        """
        super().__init__(operator, expression)


class UnaryBooleanOperation(UnaryOperation):
    """Unary boolean operation expression representation.

    https://docs.python.org/3.4/reference/expressions.html#boolean-operations
    """

    class Operator(UnaryOperation.Operator):
        """Unary boolean operator representation."""
        Neg = 1

        def __str__(self):
            if self.value == 1:
                return "not"

    def __init__(self, operator: Operator, expression: Expression):
        """Unary boolean operation expression representation.

        :param operator: operator of the operation
        :param expression: expression of the operation
        """
        super().__init__(operator, expression)


"""
Binary Operation Expressions
"""


class BinaryOperation(Operation):
    """Binary operation representation."""
    class Operator(IntEnum):
        """Binary operator representation."""

        @abstractmethod
        def __str__(self):
            """Binary operator string representation.

            :return: string representing the operator
            """

    def __init__(self, left: Expression, operator: Operator, right: Expression):
        """Binary operation construction.

        :param left: left expression of the operation
        :param operator: operator of the operation
        :param right: right expression of the operation
        """
        super().__init__()
        self._left = left
        self._operator = operator
        self._right = right

    @property
    def left(self):
        return self._left

    @property
    def operator(self):
        return self._operator

    @property
    def right(self):
        return self._right

    def __eq__(self, other: 'BinaryOperation'):
        left = self.left == other.left
        operator = self.operator == other.operator
        right = self.right == other.right
        return left and operator and right

    def __hash__(self):
        return hash((self.left, self.operator, self.right))

    def __str__(self):
        left_string = str(self.left)
        right_string = str(self.right)
        if isinstance(self.left, Operation):
            left_string = f"({left_string})"
        if isinstance(self.right, Operation):
            right_string = f"({right_string})"
        return f"{left_string} {str(self.operator)} {right_string}"


class BinaryArithmeticOperation(BinaryOperation):
    """Binary arithmetic operation expression representation.

    https://docs.python.org/3.4/reference/expressions.html#binary-arithmetic-operations
    """

    class Operator(BinaryOperation.Operator):
        """Binary arithmetic operator representation."""
        Add = 1
        Sub = 2
        Mult = 3
        Div = 4

        def __str__(self):
            if self.value == 1:
                return "+"
            elif self.value == 2:
                return "-"
            elif self.value == 3:
                return "*"
            elif self.value == 4:
                return "/"

    def __init__(self, left: Expression, operator: Operator, right: Expression):
        """Binary arithmetic operation expression representation.

        :param left: left expression of the operation
        :param operator: operator of the operation
        :param right: right expression of the operation
        """
        super().__init__(left, operator, right)


class BinaryBooleanOperation(BinaryOperation):
    """Binary boolean operation expression representation.

    https://docs.python.org/3.6/reference/expressions.html#boolean-operations
    """

    class Operator(BinaryOperation.Operator):
        """Binary arithmetic operator representation."""
        And = 1
        Or = 2

        def reverse_operator(self):
            """Returns the reverse operator of this operator."""
            if self.value == 1:
                return BinaryBooleanOperation.Operator.Or
            elif self.value == 2:
                return BinaryBooleanOperation.Operator.And

        def __str__(self):
            return self.name.lower()

    def __init__(self, left: Expression, operator: Operator, right: Expression):
        """Binary boolean operation expression representation.

        :param left: left expression of the operation
        :param operator: operator of the operation
        :param right: right expression of the operation
        """
        super().__init__(left, operator, right)


class BinaryComparisonOperation(BinaryOperation):
    """Binary comparison operation expression representation.

    https://docs.python.org/3.4/reference/expressions.html#comparisons
    """

    class Operator(BinaryOperation.Operator):
        """Binary comparison operator representation"""
        Eq = 1
        NotEq = 2
        Lt = 3
        LtE = 4
        Gt = 5
        GtE = 6

        def reverse_operator(self):
            """Returns the reverse operator of this operator."""
            if self.value == 1:
                return BinaryComparisonOperation.Operator.NotEq
            elif self.value == 2:
                return BinaryComparisonOperation.Operator.Eq
            elif self.value == 3:
                return BinaryComparisonOperation.Operator.GtE
            elif self.value == 4:
                return BinaryComparisonOperation.Operator.Gt
            elif self.value == 5:
                return BinaryComparisonOperation.Operator.LtE
            elif self.value == 6:
                return BinaryComparisonOperation.Operator.Lt

        def __str__(self):
            if self.value == 1:
                return "=="
            elif self.value == 2:
                return "!="
            elif self.value == 3:
                return "<"
            elif self.value == 4:
                return "<="
            elif self.value == 5:
                return ">"
            elif self.value == 6:
                return ">="

    def __init__(self, left: Expression, operator: Operator, right: Expression):
        """Binary comparison operation expression representation.

        :param left: left expression of the operation
        :param operator: operator of the operation
        :param right: right expression of the operation
        """
        super().__init__(left, operator, right)
