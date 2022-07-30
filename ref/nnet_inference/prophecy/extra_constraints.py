__all__ = [
    'ClassificationTargetConstraint',
    'SampleEnvelopmentConstraint'
]


class Constraint(object):
    pass


class ClassificationTargetConstraint(Constraint):
    def __init__(self, target):
        self.target = target


class SampleEnvelopmentConstraint(Constraint):
    def __init__(self, x):
        self.x = x
