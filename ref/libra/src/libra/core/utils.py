
def copy_docstring(fromfunc):
    """Decorator to copy the docstring of ``fromfunc``.

    It appends an existing docstring to it.
    """
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__:
            func.__doc__ = sourcedoc + func.__doc__
        else:
            func.__doc__ = sourcedoc
        return func
    return _decorator
