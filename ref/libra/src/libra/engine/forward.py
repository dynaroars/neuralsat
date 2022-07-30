"""
Forward Analysis Engine
=======================

:Author: Caterina Urban
"""
from copy import deepcopy
from queue import Queue
from pip._vendor.colorama import Fore, Style

from apronpy.manager import PyManager
 
from libra.core.statements import Call
from libra.engine.interpreter import Interpreter
from libra.semantics.forward import DefaultForwardSemantics
from libra.abstract_domains.state import State
from libra.core.cfg import Node, Function, Activation


class ForwardInterpreter(Interpreter):
    """Forward control flow graph interpreter."""

    def __init__(self, cfg, manager: PyManager, semantics, log=False):
        """Forward control flow graph interpreter construction.

        :param cfg: control flow graph to analyze
        :param semantics: semantics of statements in the control flow graph
        :param precursory: precursory control flow graph interpreter
        """
        super().__init__(cfg, semantics)
        self.manager = manager
        self._log = log

    def _state_log(self, state, outputs, full=False):
        """log of the state bounds (usually only Input/Output) of the state after a forward analysis step

        :param state: state of the analsis after a forward application
        :param outputs: set of outputs name
        :param full: True for full print or False for just Input/Output (Default False)
        """
        if self._log:
            input_color = Fore.YELLOW
            output_color = Fore.MAGENTA
            mid_color = Fore.LIGHTBLACK_EX
            error_color = Fore.RED
            outputs = {k.name for k in outputs}

            print("Forward Analysis (", Style.RESET_ALL, end='', sep='')
            print(input_color + "Input", Style.RESET_ALL, end='', sep='')
            print("|", Style.RESET_ALL, end='', sep='')
            if full:
                print(mid_color + "Hidden", Style.RESET_ALL, end='', sep='')
                print("|", Style.RESET_ALL, end='', sep='')

            print(output_color + "Output", Style.RESET_ALL, end='', sep='')
            print("): {", Style.RESET_ALL)

            if hasattr(state, "bounds") and isinstance(state.bounds, dict):
                inputs = [f"  {k} -> {state.bounds[k]}" for k in state.inputs]
                inputs.sort()
                print(input_color + "\n".join(inputs), Style.RESET_ALL)
                if full:
                    mid_states = [f"  {k} -> {state.bounds[k]}" for k in state.bounds.keys() - state.inputs - outputs]
                    mid_states.sort()
                    print(mid_color + "\n".join(mid_states), Style.RESET_ALL)
                outputs = [f"  {k} -> {state.bounds[k]}" for k in outputs]
                outputs.sort()
                print(output_color + "\n".join(outputs), Style.RESET_ALL)
            else:
                print(error_color + "Unable to show bounds on the param 'state'" +
                    "\n  > missing attribute 'state.bounds', or 'state.bounds' is not a dictionary" +
                    "\n  > next state logs will be hidden", Style.RESET_ALL)
                self._log = True

            print("}", Style.RESET_ALL)

    def analyze(self, initial, earlystop=True, forced_active=None, forced_inactive=None, outputs=None):
        """Forward analysis extracting abstract activation patterns.

        :param initial: initial state of the analysis
        :return: three sets: all activation nodes, always active nodes, always inactive nodes
        """
        worklist = Queue()
        worklist.put(self.cfg.in_node)
        state = deepcopy(initial)
        activated, deactivated = set(), set()

        while not worklist.empty():
            current: Node = worklist.get()  # retrieve the current node
            # execute block
            if isinstance(current, Function):
                state = state.affine(current.stmts[0], current.stmts[1])
            elif isinstance(current, Activation):
                if forced_active and current in forced_active:
                    state = state.relu(current.stmts, active=True)
                    activated.add(current)
                elif forced_inactive and current in forced_inactive:
                    state = state.relu(current.stmts, inactive=True)
                    deactivated.add(current)
                else:
                    state = state.relu(current.stmts)
                    if state.is_bottom():
                        deactivated.add(current)
                    if state.flag:
                        if state.flag > 0:
                            activated.add(current)
                        else:
                            deactivated.add(current)
            else:
                for stmt in reversed(current.stmts):
                    state = self.semantics.assume_call_semantics(stmt, state, self.manager)
            # update worklist
            for node in self.cfg.successors(current):
                worklist.put(self.cfg.nodes[node.identifier])

        self._state_log(state, outputs)
        found = state.outcome(outputs)

        return activated, deactivated, found


class ActivationPatternForwardSemantics(DefaultForwardSemantics):

    def assume_call_semantics(self, stmt: Call, state: State, manager: PyManager = None) -> State:
        argument = self.semantics(stmt.arguments[0], state).result
        state.assume(argument, manager=manager)
        state.result = set()
        return state
