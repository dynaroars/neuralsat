from collections import UserList

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from .prophecy import ProphecyAnalyzer
from .symbolic import *
from .extra_constraints import *

__all__ = ['NNInferAnalyzer']


class _CvxpyConstraintList(UserList, list):
    def __iadd__(self, other):
        if isinstance(other, list):
            self.extend(other)
        else:
            self.append(other)
        return self


def _cvxpy_symbolic_evaluate(eq_lower, eq_upper,
                             input_lower, input_upper):
    output_shape = eq_lower.shape[1:]
    eq_lower = eq_lower.flatten(1)
    eq_upper = eq_upper.flatten(1)
    input_lower = cp.reshape(input_lower, (np.prod(input_lower.shape), 1))
    input_upper = cp.reshape(input_upper, (np.prod(input_upper.shape), 1))
    output_lower = \
        cp.multiply(pos(eq_upper[:-1]).numpy(), input_lower) + cp.multiply(neg(eq_lower[:-1]).numpy(), input_upper)
    output_upper = \
        cp.multiply(pos(eq_upper[:-1]).numpy(), input_upper) + cp.multiply(neg(eq_lower[:-1]).numpy(), input_lower)
    output_lower = cp.sum(output_lower, 0) + eq_lower[-1].numpy()
    output_upper = cp.sum(output_upper, 0) + eq_upper[-1].numpy()
    return cp.reshape(output_lower, output_shape), cp.reshape(output_upper, output_shape)


# noinspection DuplicatedCode
class NNInferAnalyzer(ProphecyAnalyzer):
    """
    Implement the algorithm NNInfer presented in the paper
    "Finding Input Characterizations for Output Properties in ReLU Neural Networks".

    Ref: https://arxiv.org/abs/2003.04273
    """

    def compute_underapprox_box(self, decision_signatures, label=None, extra_constraints=()):

        """Use cvxpy to compute maximum under-approximate box."""
        # decision variables
        # input_high = cp.Variable(self.input_shape, name='x_h')
        # input_low = cp.Variable(self.input_shape, name='x_l')
        input_high = np.array(
            [cp.Variable(name=f'x_{i}_h')
             for i in range(self.n_inputs)], dtype=object).reshape(self.input_shape)
        input_low = np.array(
            [cp.Variable(name=f'x_{i}_l')
             for i in range(self.n_inputs)], dtype=object).reshape(self.input_shape)

        constraints = _CvxpyConstraintList()

        # add min <= low <= high <= max constraint
        input_high_flatten = input_high.flatten()
        input_low_flatten = input_low.flatten()
        for i in range(self.n_inputs):
            constraints += input_low_flatten[i] >= self.input_lower.flatten()[i]
            constraints += input_high_flatten[i] <= self.input_upper.flatten()[i]
            constraints += input_high_flatten[i] - input_low_flatten[i] >= 0

        # add concrete data envelopment constraint
        for constraint in extra_constraints:
            if isinstance(constraint, SampleEnvelopmentConstraint):
                for i in range(self.n_inputs):
                    constraints += input_low.flatten()[i] <= constraint.x.flatten()[i].item()
                    constraints += input_high.flatten()[i] >= constraint.x.flatten()[i].item()

        # add decision signature constraint
        eq_lower = torch.zeros(self.n_inputs + 1, *self.input_shape, dtype=torch.float64)
        eq_lower.flatten(1).fill_diagonal_(1)
        eq_upper = eq_lower.clone()
        output_low, output_high = input_low, input_high
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                eq_lower, eq_upper = linear_eq_forward(eq_lower, eq_upper, layer.weight, layer.bias)
            elif isinstance(layer, nn.Conv2d):
                eq_lower, eq_upper = conv2d_eq_forward(eq_lower, eq_upper,
                                                       layer.weight, layer.bias,
                                                       layer.stride, layer.padding, layer.dilation)
            elif isinstance(layer, nn.Flatten):
                eq_lower, eq_upper = flatten_eq_forward(eq_lower, eq_upper,
                                                        layer.start_dim, layer.end_dim)
            elif isinstance(layer, nn.ReLU):
                # add signature constraint
                if layer_id not in decision_signatures or decision_signatures[layer_id] is None:
                    break
                for i, sig in enumerate(decision_signatures[layer_id].flatten()):
                    if sig is not None:
                        if sig:
                            constraints += output_low.flatten()[i] >= 0
                        else:
                            constraints += output_high.flatten()[i] <= 0
                eq_lower, eq_upper = relu_eq_forward(eq_lower, eq_upper, decision_signatures[layer_id])
            output_low, output_high = numpy_evaluate(eq_lower, eq_upper, input_low, input_high)

            # add output constraint (if any)
            if layer_id == len(self.layers) - 1 and label is not None:
                for i in range(self.n_outputs):
                    if i != label:
                        constraints += output_low[label] >= output_high[i]

        # objective
        # objective = cp.Maximize(cp.sum(cp.log(input_high - input_low)))
        objective = cp.Maximize(sum(cp.log(input_high.flatten()[i] - input_low.flatten()[i])
                                    for i in range(self.n_inputs)))
        prob = cp.Problem(objective, constraints)

        # print(prob.constraints)
        # print(prob.objective)

        # solve
        prob.solve(solver='ECOS', verbose=self.verbose)
        # print(prob.objective.value())
        assert prob.status == cp.OPTIMAL
        input_high = np.vectorize(lambda _: _.value)(input_high).astype(np.float64)
        input_low = np.vectorize(lambda _: _.value)(input_low).astype(np.float64)
        return input_low, input_high
