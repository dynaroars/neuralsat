"""
Bias Analysis
=============

:Author: Caterina Urban
"""
import ast
import ctypes
import time
from enum import Enum
from queue import Queue
from typing import Set, Dict

from apronpy.box import PyBoxMPQManager
from apronpy.environment import PyEnvironment
from apronpy.manager import FunId, PyManager
from apronpy.polka import PyPolkaMPQstrictManager
from apronpy.texpr1 import PyTexpr1
from apronpy.var import PyVar
from pip._vendor.colorama import Style

from libra.abstract_domains.bias_domain import BiasState
from libra.abstract_domains.deeppoly_domain import DeepPolyState
from libra.abstract_domains.neurify_domain import NeurifyState
from libra.abstract_domains.product_domain import ProductState
from libra.abstract_domains.interval1_domain import Box1State
from libra.abstract_domains.interval2_domain import Box2State
from libra.abstract_domains.symbolic1_domain import Symbolic1State
from libra.abstract_domains.symbolic2_domain import Symbolic2State
from libra.abstract_domains.symbolic3_domain import Symbolic3State
from libra.core.cfg import Node, Function, Activation
from libra.core.expressions import VariableIdentifier
from libra.core.statements import Assignment, Lyra2APRON
from libra.engine.backward import BackwardInterpreter, BiasBackwardSemantics
from libra.engine.forward import ForwardInterpreter, ActivationPatternForwardSemantics
from libra.engine.runner import Runner
from libra.frontend.cfg_generator import ast_to_cfg


class AbstractDomain(Enum):
    BOXES1 = 0
    BOXES2 = 1
    SYMBOLIC1 = 2
    SYMBOLIC2 = 3
    SYMBOLIC3 = 4
    DEEPPOLY = 5
    NEURIFY = 6

    BOXES2_DEEPPOLY = 7
    BOXES2_NEURIFY = 8
    DEEPPOLY_NEURIFY = 9
    DEEPPOLY_SYMBOLIC3 = 10
    NEURIFY_SYMBOLIC3 = 11

    BOXES2_DEEPPOLY_NEURIFY = 12
    DEEPPOLY_NEURIFY_SYMBOLIC3 = 13


class BiasAnalysis(Runner):

    def __init__(self, spec, domain=AbstractDomain.SYMBOLIC3, steps=None, minL=None, startL=0.25, startU=2, maxU=None, cpu=None, analysis=True):
        super().__init__()
        self.spec = spec
        self.domain = domain
        self.steps = (1, 1) if steps is None else steps
        self.minL = startL if minL is None else minL
        self.startL = startL
        self.startU = startU
        self.maxU = startU if maxU is None else maxU
        self.analysis = analysis
        self.cpu = cpu

        self.inputs: Set[VariableIdentifier] = None                             # input variables
        self.outputs: Set[VariableIdentifier] = None                            # output variables
        self.activations: Set[Node] = None                                      # activations
        self.splits: Dict[int, Dict[VariableIdentifier, PyTexpr1]] = None       # partitioning information
        self.relus: Dict[VariableIdentifier, Node] = None                       # relus information
        self.man1: PyManager = PyBoxMPQManager()
        self.man2: PyManager = PyPolkaMPQstrictManager()
        min_int = (-ctypes.c_uint(-1).value) // 2
        self.man2.manager.contents.option.funopt[FunId.AP_FUNID_IS_BOTTOM].algorithm = min_int
        self.man2.manager.contents.option.funopt[FunId.AP_FUNID_IS_TOP].algorithm = min_int
        self.man2.manager.contents.option.funopt[FunId.AP_FUNID_MEET].algorithm = min_int
        self.man2.manager.contents.option.funopt[FunId.AP_FUNID_JOIN].algorithm = min_int
        self.man2.manager.contents.option.funopt[FunId.AP_FUNID_FORGET_ARRAY].algorithm = min_int

    def interpreter(self):
        precursory = ForwardInterpreter(self.cfg, self.man1, ActivationPatternForwardSemantics())
        return BackwardInterpreter(self.cfg, self.man2, self.domain, BiasBackwardSemantics(), self.spec,
                                   steps=self.steps,
                                   minL=self.minL, startL=self.startL, startU=self.startU, maxU=self.maxU,
                                   cpu=self.cpu, precursory=precursory)

    def state(self):
        self.inputs, variables, self.outputs = self.variables
        if self.domain == AbstractDomain.BOXES1:
            precursory = Box1State(self.man1, variables)
        elif self.domain == AbstractDomain.BOXES2:
            precursory = Box2State(self.inputs)
        elif self.domain == AbstractDomain.SYMBOLIC1:
            # generally faster than SYMBOLIC2 for small neural networks
            precursory = Symbolic1State(self.man1, variables)
        elif self.domain == AbstractDomain.SYMBOLIC2:
            # generally faster than SYMBOLIC1 for large neural networks
            precursory = Symbolic2State(self.man1, variables)
        elif self.domain == AbstractDomain.SYMBOLIC3:
            # without using APRON
            precursory = Symbolic3State(self.inputs)
        elif self.domain == AbstractDomain.DEEPPOLY:
            precursory = DeepPolyState(self.inputs)
        elif self.domain == AbstractDomain.NEURIFY:
            precursory = NeurifyState(self.inputs)
        elif self.domain == AbstractDomain.BOXES2_DEEPPOLY:
            precursory = ProductState(self.inputs, [Box2State(self.inputs), DeepPolyState(self.inputs)])
        elif self.domain == AbstractDomain.BOXES2_NEURIFY:
            precursory = ProductState(self.inputs, [Box2State(self.inputs), NeurifyState(self.inputs)])
        elif self.domain == AbstractDomain.DEEPPOLY_SYMBOLIC3:
            precursory = ProductState(self.inputs, [DeepPolyState(self.inputs), Symbolic3State(self.inputs)])
        elif self.domain == AbstractDomain.DEEPPOLY_NEURIFY:
            precursory = ProductState(self.inputs, [DeepPolyState(self.inputs), NeurifyState(self.inputs)])
        elif self.domain == AbstractDomain.NEURIFY_SYMBOLIC3:
            precursory = ProductState(self.inputs, [NeurifyState(self.inputs), Symbolic3State(self.inputs)])
        elif self.domain == AbstractDomain.BOXES2_DEEPPOLY_NEURIFY:
            precursory = ProductState(self.inputs, [Box2State(self.inputs), DeepPolyState(self.inputs), NeurifyState(self.inputs)])
        else:
            assert self.domain == AbstractDomain.DEEPPOLY_NEURIFY_SYMBOLIC3
            precursory = ProductState(self.inputs, [DeepPolyState(self.inputs), NeurifyState(self.inputs), Symbolic3State(self.inputs)])
        return BiasState(self.man2, variables, precursory=precursory)

    @property
    def variables(self):
        variables, assigned, outputs = set(), set(), set()
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current = worklist.get()
            for stmt in current.stmts:
                variables = variables.union(stmt.ids())
                if isinstance(stmt, Assignment):
                    assigned = assigned.union(stmt.left.ids())
                    outputs = outputs.union(stmt.left.ids())
            if isinstance(current, Activation):  # there is another layer
                outputs = set()
            for node in self.cfg.successors(current):
                worklist.put(node)
        return variables.difference(assigned), variables, outputs

    _lyra2apron = Lyra2APRON()

    def lyra2apron(self, environment):
        activations = set()
        splits: Dict[int, Dict[VariableIdentifier, PyTexpr1]] = dict()
        relus: Dict[VariableIdentifier, Node] = dict()
        layer: int = 1
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            # execute block
            if isinstance(current, Function):
                affine: Dict[VariableIdentifier, PyTexpr1] = dict()
                vars = list()
                exprs = list()
                for assignment in current.stmts:
                    variable, expression = self._lyra2apron.visit(assignment, environment)
                    affine[assignment.left.variable] = expression
                    vars.append(variable)
                    exprs.append(expression)
                splits[layer] = affine
                layer += 1
                newnode = Function(current.identifier, (vars, exprs))
                self.cfg.nodes[current.identifier] = newnode
            elif isinstance(current, Activation):
                activations.add(current)
                relus[current.stmts[0].arguments[0].variable] = current
                variable = self._lyra2apron.visit(current.stmts[0], environment)
                newnode = Activation(current.identifier, variable)
                self.cfg.nodes[current.identifier] = newnode
            # update worklist
            for node in self.cfg.successors(current):
                worklist.put(node)
        # delete last added layer because there are no ReLUs there
        del splits[layer - 1]
        return activations, splits, relus

    def main(self, path):
        self.path = path
        with open(self.path, 'r') as source:
            self.source = source.read()
            self.tree = ast.parse(self.source)
            self.cfg = ast_to_cfg(self.tree)
            # # rendering of the original CFG
            # renderer = CFGRenderer()
            # data = self.cfg
            # name = os.path.splitext(os.path.basename(self.path))[0]
            # label = f"CFG for {name}"
            # directory = os.path.dirname(self.path)
            # renderer.render(data, filename=name, label=label, directory=directory, view=True)
            # walking over the CFG to convert statements to APRON and to collect partitioning information
            _, variables, _ = self.variables
            r_vars = list()
            for variable in variables:
                r_vars.append(PyVar(variable.name))
            environment = PyEnvironment([], r_vars)
            self.activations, self.splits, self.relus = self.lyra2apron(environment)
        self.run()

    def run(self):
        start = time.time()
        log = self.interpreter().analyze(self.state(), inputs=self.inputs, outputs=self.outputs, activations=self.activations, analysis=self.analysis)
        end = time.time()
        print('Total Time: {}s'.format(end - start), Style.RESET_ALL)
        # print('{} {}s'.format(log, end - start))
