
from abc import ABCMeta, abstractmethod

class BoundsDomain(metaclass=ABCMeta):

    @abstractmethod
    def get_bounds(self, var_name):
        """[summary]

        Args:
            var_name ([type]): [description]
        """

    @abstractmethod
    def resize_bounds(self, new_bounds):
        """[summary]

        Args:
            new_bounds ([type]): [description]
        """
