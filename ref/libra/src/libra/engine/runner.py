"""
Program Analysis
================

:Author: Caterina Urban
"""

import ast
import os
import time
from abc import abstractmethod
from queue import Queue
from typing import Set

from libra.core.expressions import VariableIdentifier
from libra.core.statements import Assignment, VariableAccess
from libra.engine.result import AnalysisResult
from libra.frontend.cfg_generator import ast_to_cfg


class Runner:
    """Analysis runner."""

    def __init__(self):
        self._path = None
        self._source = None
        self._tree = None
        self._cfg = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        self._cfg = cfg

    @abstractmethod
    def interpreter(self):
        """Control flow graph interpreter."""

    @abstractmethod
    def state(self):
        """Initial analysis state."""

    @property
    def variables(self) -> Set[VariableIdentifier]:
        variables = set()
        visited, worklist = set(), Queue()
        worklist.put(self.cfg.in_node)
        while not worklist.empty():
            current = worklist.get()
            if current.identifier not in visited:
                visited.add(current.identifier)
                for stmt in current.stmts:
                    if isinstance(stmt, Assignment) and isinstance(stmt.left, VariableAccess):
                        variable = stmt.left.variable
                        variables.add(variable)
                for node in self.cfg.successors(current):
                    worklist.put(node)
        return variables

    def main(self, path):
        self.path = path
        with open(self.path, 'r') as source:
            self.source = source.read()
            self.tree = ast.parse(self.source)
            self.cfg = ast_to_cfg(self.tree)
        return self.run()

    def run(self) -> AnalysisResult:
        start = time.time()
        result = self.interpreter().analyze(self.state())
        end = time.time()
        print('Time: {}s'.format(end - start))
        return result

