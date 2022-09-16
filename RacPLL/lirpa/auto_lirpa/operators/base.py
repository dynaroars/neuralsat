import torch.nn as nn
import torch
import time
import os

from auto_lirpa.perturbations import *

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

class Interval(tuple):

    # Subclassing tuple object so that all previous code can be reused.
    def __new__(self, lb=None, ub=None, nominal=None, lower_offset=None, upper_offset=None, ptb=None):
        # print('__new__ method called')
        return tuple.__new__(Interval, (lb, ub))

    def __init__(self, lb, ub, nominal=None, lower_offset=None, upper_offset=None, ptb=None):
        # print('__init__ method called')
        self.nominal = nominal
        self.lower_offset = lower_offset
        self.upper_offset = upper_offset

        if ptb is None:
            self.ptb = None
        else:
            if not isinstance(ptb, Perturbation):
                raise ValueError("ptb must be a Perturbation object or None. Got type {}".format(type(ptb)))
            else:
                self.ptb = ptb


    @staticmethod
    def get_perturbation(interval):
        if isinstance(interval, Interval) and interval.ptb is not None:
            if isinstance(interval.ptb, PerturbationLpNorm):
                return interval.ptb.norm, interval.ptb.eps
            # elif isinstance(interval.ptb, PerturbationSynonym):
            #     return np.inf, 1.0
            # elif isinstance(interval.ptb, PerturbationL0Norm):
            #     return 0, interval.ptb.eps, interval.ptb.ratio
            # elif interval.ptb is None:
            #     raise RuntimeError("get_perturbation() encountered an interval that is not perturbed.")
            else:
                raise RuntimeError("get_perturbation() does not know how to handle {}".format(type(interval.ptb)))
        else:
            # Tuple object. Assuming L infinity norm lower and upper bounds.
            return np.inf, np.nan

    @staticmethod
    def use_relative_bounds(*intervals):
        using = True
        for interval in intervals:
            using = using and (
                isinstance(interval, Interval) and 
                interval.nominal is not None and 
                interval.lower_offset is not None and interval.upper_offset is not None)
        return using


    @staticmethod
    def make_interval(lb, ub, other=None, nominal=None, use_relative=False):
        if isinstance(other, Interval):
            return Interval(lb, ub, ptb=other.ptb)
        else:
            if use_relative:
                if nominal is None:
                    return Interval(
                        None, None, (lb + ub) / 2, (lb - ub) / 2, (ub - lb) / 2)
                else:
                    return Interval(None, None, nominal, lb - nominal, ub - nominal)
            else:
                return (lb, ub)

        
    def __str__(self):
        return "({}, {}) with ptb={}".format(self[0], self[1], self.ptb)

    def __repr__(self):
        return "Interval(lb={}, ub={}, ptb={})".format(self[0], self[1], self.ptb)

class Bound(nn.Module):

    r"""
    Base class for supporting the bound computation of an operator.

    Args:
        input_name (list): The name of input nodes.
        name (str): The name of this node.
        ori_name (str): Name in the original model.
        attr (dict): Attributes of the operator.
        inputs (list): A list of input nodes.
        output_index (int): The index in the output if the operator has multiple outputs. Usually output_index=0.
        options (dict): Bound options.
        device (str or torch.device): Device of the bounded module.
    """

    def __init__(self, input_name, name, ori_name, attr={}, inputs=[], output_index=0, options={}, device=None):
        super().__init__()

        self.input_name, self.name, self.ori_name, self.attr, self.inputs, self.output_index, self.options, self.device = \
            input_name, name, ori_name, attr, inputs, output_index, options, device

        self.from_input = False
        self.use_default_ibp = False 

        self.perturbed = False
        self.forward_value = None
        # If set to true, the backward bound output of this node is 0.
        self.zero_backward_coeffs_l = False
        self.zero_backward_coeffs_u = False
        # If set to true, the A matrix accumulated on this node is 0.
        self.zero_lA_mtx = False
        self.zero_uA_mtx = False
        
    """save input and output shapes uniformly by the decorator"""
    @staticmethod
    def save_io_shape(func):
        def wrapper(self, *args, **kwargs):
            if len(args) > 0:
                # x should always be the first input
                self.input_shape = args[0].shape  

            output = func(self, *args, **kwargs)

            if isinstance(output, torch.Tensor):
                self.output_shape = output.shape
            return output

        return wrapper

    def infer_batch_dim(self, batch_size, *x):
        # Default implementation assuming the batch dimension is always at 0.
        # Do not use it if the operator can alter the shape
        assert x[0] in [0, -1]
        return x[0]


    """Check if the i-th input is with perturbation or not."""
    def is_input_perturbed(self, i=0):
        return self.inputs[i].perturbed


    def interval_propagate(self, *v):
        r"""
        Function for interval bound propagation (IBP) computation.

        There is a default function `self.default_interval_propagate(*v)` in the base class, 
        which can be used if the operator is *monotonic*. To use it, set `self.use_default_ibp = True`
        in the `__init__` function, and the implementation of this function can be skipped.

        Args: 
            v: A list of the interval bound of input nodes. 
            Generally, for each element `v[i]`, `v[i][0]` is the lower interval bound,
            and `v[i][1]` is the upper interval bound.

        Returns:
            bound: The interval bound of this node, in a same format as v[i].
        """        
        if self.use_default_ibp:
            return self.default_interval_propagate(*v)
        raise NotImplementedError  

    """For unary monotonous functions or functions for altering shapes only but not values"""
    def default_interval_propagate(self, *v):
        if len(v) == 0:
            return Interval.make_interval(self.forward(), self.forward())
        elif len(v) == 1:
            if Interval.use_relative_bounds(v[0]):
                return Interval(
                    None, None,
                    self.forward(v[0].nominal), 
                    self.forward(v[0].lower_offset), 
                    self.forward(v[0].upper_offset)
                )
            else:
                return Interval.make_interval(self.forward(v[0][0]), self.forward(v[0][1]), v[0])
        else:
            raise NotImplementedError('default_interval_propagate only supports no more than 1 input node')


    @staticmethod
    @torch.jit.script
    def clamp_mutiply(A, pos, neg):
        Apos = A.clamp(min=0)
        Aneg = A.clamp(max=0)
        return pos.contiguous() * Apos + neg.contiguous() * Aneg, Apos, Aneg


    """Some operations are non-deterministic and deterministic mode will fail. So we temporary disable it."""
    def non_deter_wrapper(self, op, *args, **kwargs):
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(False)
        ret = op(*args, **kwargs)
        if self.options.get('deterministic', False):
            torch.use_deterministic_algorithms(True)
        return ret

    def non_deter_scatter_add(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.scatter_add, *args, **kwargs)

    def non_deter_index_select(self, *args, **kwargs):
        return self.non_deter_wrapper(torch.index_select, *args, **kwargs)
