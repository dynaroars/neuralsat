from collections.abc import Sequence

__all__ = ['_totuple']


def _totuple(x) -> tuple:
    if isinstance(x, Sequence):
        return tuple(x)
    return x,
