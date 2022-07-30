from collections import OrderedDict

import numpy as np
import pulp
import torch
import torch.nn as nn

try:
    from ._import_maraboupy import import_marabou

    import_marabou()

    from maraboupy import MarabouCore as mc
    from maraboupy.Marabou import createOptions
except:
    pass

from .symbolic import *
from .utils import _totuple
from .extra_constraints import *

__all__ = ['ProphecyAnalyzer']

_C_LARGE = 999999


# noinspection DuplicatedCode
class ProphecyAnalyzer:
    """
    Implement the algorithm Prophecy presented in the paper
    "Property Inference for Deep Neural Networks".

    Ref: https://arxiv.org/abs/1904.13215
    """

    def __init__(self,
                 net,
                 input_shape,
                 input_lower=None,
                 input_upper=None,
                 eps=1e-5,
                 verbose=False):
        self.net = net.double()
        self.input_shape = _totuple(input_shape)
        self.n_inputs = np.prod(self.input_shape)
        self.input_lower = input_lower
        self.input_upper = input_upper
        if self.input_lower is None:
            self.input_lower = torch.empty(self.input_shape, dtype=torch.float64).fill_(-torch.inf)
        if self.input_upper is None:
            self.input_upper = torch.empty(self.input_shape, dtype=torch.float64).fill_(torch.inf)
        self.eps = eps
        self.verbose = verbose

        # setup variable indices
        _cnt = self.n_inputs
        _features_shape = self.input_shape
        _features_numel = self.n_inputs
        self.variables = [np.arange(0, self.n_inputs).reshape(self.input_shape)]
        for layer in self.layers:
            _requires_variables = True
            # update features shape and numel
            if isinstance(layer, nn.Linear):
                _features_shape = (layer.out_features,)
                _features_numel = layer.out_features
            elif isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                out_shape = [(_features_shape[i + 1] + 2 * layer.padding[i] - (
                        layer.dilation[i] * (layer.kernel_size[i] - 1) + 1)) // layer.stride[i] + 1
                             for i in range(2)]
                _features_shape = (layer.out_channels, *out_shape)
                _features_numel = np.prod(_features_shape)
            elif isinstance(layer, nn.Flatten):
                _features_shape = (_features_numel,)
                _requires_variables = False
            elif isinstance(layer, nn.ReLU):
                pass
            else:
                raise NotImplementedError(f'Unsupported layer {type(layer)}.')
            # add variables
            if _requires_variables:
                self.variables.append(np.arange(_cnt, _cnt + _features_numel).reshape(_features_shape))
                _cnt += _features_numel
            else:
                self.variables.append(self.variables[-1].reshape(_features_shape))
        self.n_variables = _cnt

        self.input_variables = self.variables[0]
        self.output_variables = self.variables[-1]
        self.n_outputs = self.output_variables.size

    @torch.no_grad()
    def forward(self, x):
        """Forward through network to get output and decision signatures"""
        decision_signatures = OrderedDict()
        x = x.unsqueeze(0)
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):  # save decision signatures of relu layers
                decision = x.gt(0).long()
                x = x * decision
                decision_signatures[layer_id] = decision.squeeze(0).cpu().numpy().astype(object)
            else:
                x = layer.forward(x)
        return x.squeeze(0), decision_signatures

    def decision_procedure(self, decision_signatures, output_cls=None):
        """Call a decision procedure (i.e. Marabou) to check whether implication holds."""
        # setup input query
        ipq = mc.InputQuery()
        ipq.setNumberOfVariables(self.n_variables + 1 if output_cls is not None
                                 else self.n_variables)

        # set lower and upper bounds of inputs
        input_lower = torch.clamp(self.input_lower, -_C_LARGE, _C_LARGE).flatten()
        input_upper = torch.clamp(self.input_upper, -_C_LARGE, _C_LARGE).flatten()
        for i, v in enumerate(self.input_variables.flatten()):
            ipq.setLowerBound(v, input_lower[i].item())
            ipq.setUpperBound(v, input_upper[i].item())

        # add network constraints
        for layer_id, layer in enumerate(self.layers):
            prev_layer_id = layer_id - 1
            layer_variables = self.variables[layer_id + 1]
            prev_layer_variables = self.variables[prev_layer_id + 1]
            if isinstance(layer, nn.Linear):
                for j, v in enumerate(layer_variables):
                    eq = mc.Equation()
                    # print(f'{v} = ', end='')
                    for i, pv in enumerate(prev_layer_variables):
                        eq.addAddend(layer.weight[j, i].item(), pv)
                        # print(f'{layer.weight[j, i].item()}*{pv} + ', end='')
                    eq.addAddend(-1, v)
                    # print(f'{layer.bias[j].item() if layer.bias is not None else 0} ')
                    eq.setScalar(layer.bias[j].item() if layer.bias is not None else 0)
                    ipq.addEquation(eq)
            elif isinstance(layer, nn.Conv2d):
                stride_h, stride_w = layer.stride
                pad_h, pad_w = layer.padding
                dil_h, dil_w = layer.dilation
                kernel_h, kernel_w = layer.kernel_size

                n_in_channels, in_h, in_w = prev_layer_variables.shape
                n_out_channels, out_h, out_w = layer_variables.shape

                n_weight_grps = layer.groups
                in_c_per_weight_grp = layer.weight.shape[1]
                out_c_per_weight_grp = n_out_channels // n_weight_grps

                for c_out in range(n_out_channels):
                    weight_grp = c_out // out_c_per_weight_grp
                    for i in range(out_h):
                        for j in range(out_w):
                            eq = mc.Equation()
                            v = layer_variables[c_out, i, j]
                            for di in range(kernel_h):
                                for dj in range(kernel_w):
                                    for c in range(in_c_per_weight_grp):
                                        pi = stride_h * i - pad_h + dil_h * di
                                        pj = stride_w * j - pad_w + dil_w * dj
                                        if 0 <= pi < in_h and 0 <= pj < in_w:
                                            c_in = weight_grp * in_c_per_weight_grp + c
                                            pv = prev_layer_variables[c_in, pi, pj]
                                            eq.addAddend(layer.weight[c_out, c, di, dj].item(), pv)
                            eq.addAddend(-1, v)
                            eq.setScalar(layer.bias[c_out].item() if layer.bias is not None else 0)
                            ipq.addEquation(eq)
            elif isinstance(layer, nn.MaxPool2d):
                stride_h, stride_w = layer.stride
                pad_h, pad_w = layer.padding
                dil_h, dil_w = layer.dilation
                kernel_h, kernel_w = layer.kernel_size

                n_in_channels, in_h, in_w = prev_layer_variables.shape
                _, out_h, out_w = layer_variables.shape

                for c in range(n_in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            v = layer_variables[c, i, j]
                            pvs = set()
                            for di in range(kernel_h):
                                for dj in range(kernel_w):
                                    pi = stride_h * i - pad_h + dil_h * di
                                    pj = stride_w * j - pad_w + dil_w * dj
                                    if 0 <= pi < in_h and 0 <= pj < in_w:
                                        pvs.add(prev_layer_variables[c, pi, pj])
                            mc.addMaxConstraint(ipq, pvs, v)
            elif isinstance(layer, nn.Flatten):
                pass
            elif isinstance(layer, nn.ReLU):
                layer_variables = layer_variables.flatten()
                prev_layer_variables = prev_layer_variables.flatten()
                # add relu constraints
                for pv, v in zip(prev_layer_variables, layer_variables):
                    mc.addReluConstraint(ipq, pv, v)
                    # print(f'{v} = relu({pv})')
                # add decision signature constraints
                if layer_id not in decision_signatures:
                    for v in layer_variables:
                        ipq.setLowerBound(v, 0)
                        ipq.setUpperBound(v, _C_LARGE)
                        # print(v, 'unconstrained')
                else:
                    for v, sig in zip(layer_variables, decision_signatures[layer_id].flatten()):
                        if sig is None:  # ignore:
                            ipq.setLowerBound(v, 0)
                            ipq.setUpperBound(v, _C_LARGE)
                            # print(f'{v} unconstrained')
                        elif sig:  # constrain > 0
                            ipq.setLowerBound(v, self.eps)
                            ipq.setUpperBound(v, _C_LARGE)
                            # print(f'{v} > 0')
                        else:  # constrain <= 0
                            ipq.setLowerBound(v, 0)
                            ipq.setUpperBound(v, 0)
                            # print(f'{v} == 0')

        # add output constraints
        if output_cls is not None:
            label_v = self.output_variables[output_cls]
            # get max of other classes
            mc.addMaxConstraint(ipq,
                                set(v for v in self.output_variables if v != label_v),
                                self.n_variables)
            # constrain label variable to be less than that max
            eq = mc.Equation(mc.Equation.LE)
            eq.addAddend(1, label_v)
            eq.addAddend(-1, self.n_variables)
            eq.setScalar(self.eps)
            ipq.addEquation(eq)
            # print(f'{label_v} < max({[v for v in self.output_variables if v != label_v]})')

        options = createOptions(verbosity=False)
        res, vars, stats = mc.solve(ipq, options)
        if res == 'ERROR':
            raise RuntimeError('Unknown MarabouError encountered.')
        return res == 'sat'

    def iterative_relaxation(self, decision_signatures, label):
        """Perform iterative relaxation"""
        relaxed_decision_signatures = decision_signatures.copy()

        sat = self.decision_procedure(relaxed_decision_signatures, label)
        if sat:
            return relaxed_decision_signatures, True
        # TODO: remove this
        # return relaxed_decision_signatures, False
        for layer_id in reversed(decision_signatures.keys()):
            unconstrained_layer = relaxed_decision_signatures.pop(layer_id)  # remove 1 layer constrain
            sat = self.decision_procedure(relaxed_decision_signatures, label)
            if sat:  # critical layer
                # add back layer constrain
                relaxed_decision_signatures.update({layer_id: unconstrained_layer})
                critical_layer = unconstrained_layer.reshape(-1)
                for i in range(critical_layer.size):
                    unconstrained_neuron = critical_layer[i]
                    critical_layer[i] = None  # remove 1 neuron constrain
                    sat = self.decision_procedure(relaxed_decision_signatures, label)
                    if sat:  # critical neuron
                        # add back neuron constrain
                        critical_layer[i] = unconstrained_neuron
                return relaxed_decision_signatures, False
        return relaxed_decision_signatures, False

    def compute_underapprox_box(self, decision_signatures, label=None, extra_constraints=None):
        """Use PuLP to compute maximum under-approximate box."""
        prob = pulp.LpProblem('under-approx_box', pulp.LpMaximize)
        # decision variables
        input_high = np.array(
            [pulp.LpVariable(f'x_{i}_h',
                             lowBound=self.input_lower.flatten()[i],
                             upBound=self.input_upper.flatten()[i],
                             cat=pulp.LpContinuous)
             for i in range(self.n_inputs)], dtype=object).reshape(self.input_shape)
        input_low = np.array(
            [pulp.LpVariable(f'x_{i}_l',
                             lowBound=self.input_lower.flatten()[i],
                             upBound=self.input_upper.flatten()[i],
                             cat=pulp.LpContinuous)
             for i in range(self.n_inputs)], dtype=object).reshape(self.input_shape)

        # objective
        prob.setObjective(sum(input_high.flatten()[i] - input_low.flatten()[i]
                              for i in range(self.n_inputs)))
        # prob += input_high[0] - input_low[0] <= 5

        # add concrete data envelopment constraint
        for constraint in extra_constraints:
            if isinstance(constraint, SampleEnvelopmentConstraint):
                for i in range(self.n_inputs):
                    prob += input_low.flatten()[i] <= constraint.x.flatten()[i].item()
                    prob += input_high.flatten()[i] >= constraint.x.flatten()[i].item()

        # add high >= low constraint
        for i in range(self.n_inputs):
            prob += input_high.flatten()[i] - input_low.flatten()[i] >= 0

        # add decision signature constraint
        eq_lower = torch.zeros(self.n_inputs + 1, *self.input_shape, dtype=torch.float64)
        eq_lower.flatten(1).fill_diagonal_(1)
        eq_upper = eq_lower.clone()
        output_low, output_high = input_low, input_high
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                eq_lower, eq_upper = linear_eq_forward(eq_lower, eq_upper,
                                                       layer.weight, layer.bias)
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
                            prob += output_low.flatten()[i] >= 0
                        else:
                            prob += output_high.flatten()[i] <= 0
                eq_lower, eq_upper = relu_eq_forward(eq_lower, eq_upper, decision_signatures[layer_id])
            output_low, output_high = numpy_evaluate(eq_lower, eq_upper, input_low, input_high)

            # add output constraint (if any)
            if layer_id == len(self.layers) - 1 and label is not None:
                for i in range(self.n_outputs):
                    if i != label:
                        prob += output_low[label] >= output_high[i]

        # print(prob.constraints)
        # print(prob.objective)

        # solve
        prob.solve(pulp.PULP_CBC_CMD(msg=self.verbose))
        # print(prob.objective.value())
        assert prob.status == pulp.LpStatusOptimal
        input_high = np.vectorize(lambda _: _.value())(input_high).astype(np.float64)
        input_low = np.vectorize(lambda _: _.value())(input_low).astype(np.float64)
        return input_low, input_high

    def infer(self, x, envelop=False):
        if x.shape[1:] == self.input_shape:
            assert x.size(0) == 1
            x = x.squeeze(0)

        # get decision signatures and output class
        y, decision_signatures = self.forward(x)
        label = y.argmax().item()

        # relax input property
        if self.verbose:
            print('label:', label)
            print('decision signatures:', decision_signatures)
        relaxed_decision_signatures, with_output_constraint = self.iterative_relaxation(
            decision_signatures, label
        )
        if self.verbose:
            print('relaxed:', relaxed_decision_signatures, with_output_constraint)

        # compute under-approximate input box
        input_low, input_high = self.compute_underapprox_box(
            decision_signatures=relaxed_decision_signatures,
            label=label if with_output_constraint else None,
            extra_constraints=[SampleEnvelopmentConstraint(x)] if envelop else [],
        )
        return input_low, input_high

    @property
    def layers(self):
        return self.net.layers
