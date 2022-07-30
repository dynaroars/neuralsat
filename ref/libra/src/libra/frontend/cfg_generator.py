import ast
import optparse
import sys

from libra.core.cfg import *
from libra.core.expressions import Literal
from libra.core.statements import *


class LooseControlFlowGraph:
    class SpecialEdgeType(Enum):
        BREAK = 1
        CONTINUE = 2

    def __init__(self, nodes: Set[Node] = None, in_node: Node = None, out_node: Node = None,
                 edges: Set[Edge] = None,
                 loose_in_edges=None, loose_out_edges=None, both_loose_edges=None):
        """Loose control flow graph representation.

        This representation uses a complete (non-loose) control flow graph via aggregation
        and adds loose edges and
        some transformations methods to combine, prepend and append loose control flow graphs.
        This class
        intentionally does not provide access to the linked CFG.
        The completed CFG can be retrieved finally with
        `eject()`.

        :param nodes: optional set of nodes of the control flow graph
        :param in_node: optional entry node of the control flow graph
        :param out_node: optional exit node of the control flow graph
        :param edges: optional set of edges of the control flow graph
        :param loose_in_edges: optional set of loose edges
        that have no start yet and end inside this CFG
        :param loose_out_edges: optional set of loose edges
        that start inside this CFG and have no end yet
        :param both_loose_edges: optional set of loose edges, loose on both ends
        """
        assert not in_node or not (loose_in_edges or both_loose_edges)
        assert not out_node or not (loose_out_edges or both_loose_edges)
        assert all([e.source is None for e in loose_in_edges or []])
        assert all([e.target is None for e in loose_out_edges or []])
        assert all([e.source is None and e.target is None for e in both_loose_edges or []])

        self._cfg = ControlFlowGraph(nodes or set(), in_node, out_node, edges or set())
        self._loose_in_edges = loose_in_edges or set()
        self._loose_out_edges = loose_out_edges or set()
        self._both_loose_edges = both_loose_edges or set()
        self._special_edges = []

    @property
    def nodes(self) -> Dict[int, Node]:
        return self._cfg.nodes

    @property
    def in_node(self) -> Node:
        return self._cfg.in_node

    @in_node.setter
    def in_node(self, node):
        self._cfg._in_node = node

    @property
    def out_node(self) -> Node:
        return self._cfg.out_node

    @out_node.setter
    def out_node(self, node):
        self._cfg._out_node = node

    @property
    def edges(self) -> Dict[Tuple[Node, Node], Edge]:
        return self._cfg.edges

    @property
    def loose_in_edges(self) -> Set[Edge]:
        return self._loose_in_edges

    @property
    def loose_out_edges(self) -> Set[Edge]:
        return self._loose_out_edges

    @property
    def both_loose_edges(self) -> Set[Edge]:
        return self._both_loose_edges

    @property
    def special_edges(self) -> List[Tuple[Edge, SpecialEdgeType]]:
        return self._special_edges

    def loose(self):
        loose = len(self.loose_in_edges) or len(self.loose_out_edges) or len(self.both_loose_edges)
        return loose or len(self.special_edges)

    def add_node(self, node):
        self.nodes[node.identifier] = node

    def add_edge(self, edge):
        """Add a (loose/normal) edge to this loose CFG.
        """
        if not edge.source and not edge.target:
            self.both_loose_edges.add(edge)
            self._cfg._in_node = None
            self._cfg._out_node = None
        elif not edge.source:
            self.loose_in_edges.add(edge)
            self._cfg._in_node = None
        elif not edge.target:
            self.loose_out_edges.add(edge)
            self._cfg._out_node = None
        else:
            self.edges[edge.source, edge.target] = edge

    def combine(self, other):
        assert not (self.in_node and other.in_node)
        assert not (self.out_node and other.out_node)
        self.nodes.update(other.nodes)
        self.edges.update(other.edges)
        self.loose_in_edges.update(other.loose_in_edges)
        self.loose_out_edges.update(other.loose_out_edges)
        self.both_loose_edges.update(other.both_loose_edges)
        self.special_edges.extend(other.special_edges)
        self._cfg._in_node = other.in_node or self.in_node  # agree on in_node
        self._cfg._out_node = other.out_node or self.out_node  # agree on out_node
        return self

    def prepend(self, other):
        other.append(self)
        self.replace(other)

    def append(self, other):
        assert not (self.loose_out_edges and other.loose_in_edges)
        assert not self.both_loose_edges or (
            not other.loose_in_edges and not other.both_loose_edges)

        self.nodes.update(other.nodes)
        self.edges.update(other.edges)

        edge_added = False
        if self.loose_out_edges:
            edge_added = True
            for e in self.loose_out_edges:
                e._target = other.in_node
                # updated/created edge is not yet in edge dict -> add
                self.edges[(e.source, e.target)] = e
            # clear loose edge sets
            self._loose_out_edges = set()
        elif other.loose_in_edges:
            edge_added = True
            for e in other.loose_in_edges:
                e._source = self.out_node
                # updated/created edge is not yet in edge dict -> add
                self.edges[(e.source, e.target)] = e
            # clear loose edge set
            other._loose_in_edges = set()

        if self.both_loose_edges:
            edge_added = True
            for e in self.both_loose_edges:
                e._target = other.in_node
                self.add_edge(e)  # updated/created edge is not yet in edge dict -> add
            # clear loose edge set
            self._both_loose_edges = set()
        elif other.both_loose_edges:
            edge_added = True
            for e in other.both_loose_edges:
                e._source = self.out_node
                self.add_edge(e)  # updated/created edge is not yet in edge dict -> add
            # clear loose edge set
            other._both_loose_edges = set()
        if not edge_added:
            # neither of the CFGs has loose ends -> add unconditional edge
            e = Unconditional(self.out_node, other.in_node)
            # updated/created edge is not yet in edge dict -> add
            self.edges[(e.source, e.target)] = e

        # in any case, transfer loose_out_edges of other to self
        self.loose_out_edges.update(other.loose_out_edges)
        self.special_edges.extend(other.special_edges)
        self._cfg._out_node = other.out_node

        return self

    def eject(self) -> ControlFlowGraph:
        if self.loose():
            error = 'This control flow graph is still loose'
            error += ' and cannot eject a complete control flow graph!'
            raise TypeError(error)
        return self._cfg

    def replace(self, other):
        self.__dict__.update(other.__dict__)


def _dummy_node(id_gen):
    return Basic(id_gen.next)


def _dummy_cfg(id_gen):
    dummy = _dummy_node(id_gen)
    return LooseControlFlowGraph({dummy}, dummy, dummy, set())


class NodeFactory(metaclass=ABCMeta):
    """
    A helper class that encapsulates a partial CFG
    and possibly some statements not yet attached to CFG.

    Whenever the
    method `complete_basic_block()` is called,
    it is ensured that all unattached statements are properly attached to
    partial CFG. The partial CFG can be retrieved at any time by property `cfg`.
    """

    def __init__(self, id_gen):
        self._stmts = []
        self._cfg = None
        self._id_gen = id_gen

    @property
    def cfg(self):
        return self._cfg

    def prepend_cfg(self, other):
        if self._cfg is not None:
            self._cfg.prepend(other)
        else:
            self._cfg = other
        return self._cfg

    def append_cfg(self, other):
        if self._cfg is not None:
            if self._cfg.loose_out_edges and other.loose_in_edges:
                self._cfg.append(_dummy_cfg(self._id_gen))
            self._cfg.append(other)
        else:
            self._cfg = other
        return self._cfg

    def add_stmts(self, stmts):
        """
        Adds statements to the currently open block.
        :param stmts: a single statement or an iterable of statements
        :return:
        """
        if isinstance(stmts, List):
            self._stmts.extend(stmts)
        else:
            self._stmts.append(stmts)

    @abstractmethod
    def complete_basic_block(self):
        """"""

    def incomplete_block(self):
        return len(self._stmts) > 0


class BasicFactory(NodeFactory):

    def complete_basic_block(self):
        if self._stmts:
            block = Basic(self._id_gen.next, self._stmts)
            self.append_cfg(LooseControlFlowGraph({block}, block, block, set()))
            self._stmts = []


class FunctionFactory(NodeFactory):

    def complete_basic_block(self):
        if self._stmts:
            block = Function(self._id_gen.next, self._stmts)
            self.append_cfg(LooseControlFlowGraph({block}, block, block, set()))
            self._stmts = []


class ActivationFactory(NodeFactory):

    def complete_basic_block(self):
        if self._stmts:
            block = Activation(self._id_gen.next, self._stmts)
            self.append_cfg(LooseControlFlowGraph({block}, block, block, set()))
            self._stmts = []


# noinspection PyPep8Naming
class AST2CFG(ast.NodeVisitor):
    """AST visitor that generates a CFG."""

    class NodeIdentifierGenerator:
        """Helper class that generates an increasing sequence of node identifiers."""

        def __init__(self):
            self._next = 0

        @property
        def next(self):
            self._next += 1
            return self._next

    def __init__(self):
        super().__init__()
        self._id_gen = AST2CFG.NodeIdentifierGenerator()

    def visit(self, node, *args, **kwargs):
        """Visit an AST node.

        :param node: node to be visited
        :return: either a statement or a partial CFG, depending on the visited node

        :keyword arguments:
            * *types* -- dictionary mapping (variable) names to their corresponding (lyra) type
            * *typ* -- type of the current node
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *args, **kwargs)

    def generic_visit(self, node, *args, **kwargs):
        print(type(node).__name__)
        raise NotImplementedError(f"Visit of {node.__class__.__name__} is unsupported!")

    # Literals

    # noinspection PyUnusedLocal
    def visit_Num(self, node):
        """Visitor function for a number (integer, float, or complex).
        The n attribute stores the value, already converted to the relevant type."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        expr = Literal(str(node.n))
        return LiteralEvaluation(pp, expr)

    # Variables

    def visit_Name(self, node):
        """Visitor function for a variable name.
        The attribute id stores the name as a string.
        The attribute ctx is Store (to assign a new value to the variable),
        Load (to load the value of the variable), or Del (to delete the variable)."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        if isinstance(node.ctx, ast.Store):
            expr = VariableIdentifier(node.id)
            return VariableAccess(pp, expr)
        if isinstance(node.ctx, ast.Load):
            expr = VariableIdentifier(node.id)
            return VariableAccess(pp, expr)
        assert isinstance(node.ctx, ast.Del)
        raise NotImplementedError(f"Name deletion is unsupported!")

    # Expressions

    # noinspection PyUnusedLocal
    def visit_Expr(self, node):
        """Visitor function for an expression statement (whose return value is unused).
        The attribute value stored another AST node."""
        return self.visit(node.value)

    def visit_UnaryOp(self, node):
        """Visitor function for a unary operation.
        The attributes op and operand store the operator
        and any expression node, respectively."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        name = type(node.op).__name__.lower()
        argument = self.visit(node.operand)
        return Call(pp, name, [argument])

    def visit_BinOp(self, node):
        """Visitor function for a binary operation.
        The attributes op, left, and right store the operator
        and any expression nodes, respectively."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        name = type(node.op).__name__.lower()
        left = self.visit(node.left)
        right = self.visit(node.right)
        return Call(pp, name, [left, right])

    def visit_BoolOp(self, node):
        """Visitor function for a boolean operation.
        The attributes op and values store the operand
        and a list of any expression node representing the operand involed, respectively."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        name = type(node.op).__name__.lower()
        arguments = [self.visit(val) for val in node.values]
        return Call(pp, name, arguments)

    def visit_Compare(self, node):
        """Visitor function for a comparison operation.
        The attributes left, ops, and comparators store the first value in the comparison,
        the list of operators, and the list of compared values after the first."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        # assert isinstance(typ, BooleanLyraType)     # we expect typ to be a BooleanLyraType
        left = self.visit(node.left)
        name = type(node.ops[0]).__name__.lower()
        second = self.visit(node.comparators[0])
        result = Call(pp, name, [left, second])
        for op, comparator in zip(node.ops[1:], node.comparators[1:]):
            name = type(op).__name__.lower()
            right = self.visit(comparator)
            current = Call(pp, name, [second, right])
            result = Call(pp, 'and', [result, current])
            second = right
        return result

    def visit_Call(self, node):
        """Visitor function for a call.
        The attribute func stores the function being called (often a Name or Attribute object).
        The attribute args stores a list fo the arguments passed by position."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        if isinstance(node.func, ast.Name):
            name: str = node.func.id
            arguments = [self.visit(arg) for arg in node.args]
            return Call(pp, name, arguments)

    # Statements

    def visit_Assign(self, node):
        """Visitor function for an assignment.
        The attribute targets stores a list of targets of the assignment.
        The attribute value stores the assigned value."""
        pp = ProgramPoint(node.lineno, node.col_offset)
        assert len(node.targets) == 1
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        return Assignment(pp, target, value)

    # Control Flow

    def _visit_body(self, body):
        factory = FunctionFactory(self._id_gen)

        for child in body:
            if isinstance(child, ast.Assign):
                factory.add_stmts(self.visit(child))
            elif isinstance(child, ast.Expr):
                assert isinstance(child.value, ast.Call)
                call = child.value
                assert isinstance(call.func, ast.Name)
                name = call.func.id
                factory.complete_basic_block()
                if name == 'assume':
                    factory2 = BasicFactory(self._id_gen)
                elif name == 'ReLU':
                    factory2 = ActivationFactory(self._id_gen)
                else:
                    factory2 = BasicFactory(self._id_gen)
                factory2.add_stmts(self.visit(child))
                factory2.complete_basic_block()

                factory.append_cfg(factory2.cfg)
            else:
                error = "The statement {} is not yet translatable to CFG!".format(child)
                raise NotImplementedError(error)
        factory.complete_basic_block()

        return factory.cfg

    # noinspection PyUnusedLocal
    def visit_Module(self, node):
        """Visitor function for a Python module."""
        start = _dummy_cfg(self._id_gen)
        body = self._visit_body(node.body)
        end = _dummy_cfg(self._id_gen)
        return start.append(body).append(end) if body else start.append(end)


def ast_to_cfg(root):
    """Generate a CFG from an AST.

    :param root: root node of the AST
    :return: the CFG generated from the given AST
    """
    loose_cfg = AST2CFG().visit(root)
    return loose_cfg.eject()


def source_to_cfg(code: str):
    """Generate a CFG from a Python program.

    :param code: Python program
    :return: the CFG generated from the given Python program
    """
    root_node = ast.parse(code)
    return ast_to_cfg(root_node)
