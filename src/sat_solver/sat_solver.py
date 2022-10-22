from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def __init__(self):
        """
        Initializes the solver.
        """
        pass

    def create_new_decision_level(self):
        """
        Creates a new decision level.
        """
        pass

    def backtrack(self, level: int):
        """
        Backtracks to the specified level.
        """
        pass

    def propagate(self):
        """
        Propagates constraints.
        """
        pass

    @abstractmethod
    def get_assignment(self) -> dict:
        """
        :return: a {literal: int -> value: bool} dictionary containing the current assignment.
        """
        pass

    @abstractmethod
    def solve(self) -> bool:
        """
        :return: True if SAT, False otherwise.
        """
        pass