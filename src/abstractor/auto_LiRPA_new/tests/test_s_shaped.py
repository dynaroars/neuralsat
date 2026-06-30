# pylint: disable=wrong-import-position
"""Test S-shaped activation functions."""
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class test_model(nn.Module):
    def __init__(self, act_func):
        super().__init__()
        self.act_func = act_func

    def forward(self, x):
        return self.act_func(x)

def sigmoid(x):
    return torch.sigmoid(x)

def sin(x):
    return torch.sin(x)


def verify_bounds(model, input_lb, input_ub, lb, ub):
    """
    Empirically verify that the model's output bounds are correct given input bounds.

    Args:
        model: The neural network model.
        input_lb: Lower bound of the input.
        input_ub: Upper bound of the input.
        lb: Computed lower bound of the output.
        ub: Computed upper bound of the output.
    """
    n_samples = 100000
    atol = 1e-5
    inputs = torch.rand(n_samples, *input_lb.shape[1:]) * (input_ub - input_lb) + input_lb
    outputs = model(inputs)
    empirical_lb = outputs.min(dim=0).values
    empirical_ub = outputs.max(dim=0).values
    if not (empirical_lb - lb >= -atol).all():
        max_violation = (lb - empirical_lb).max().item()
        raise AssertionError(f"Lower bound violated. Max violation: {max_violation}")
    if not (empirical_ub - ub <= atol).all():
        max_violation = (empirical_ub - ub).max().item()
        raise AssertionError(f"Upper bound violated. Max violation: {max_violation}")
    print("Bounds verified successfully.")


class TestSShaped(TestCase):
    def __init__(self, methodName='runTest', generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(
            methodName, seed=1, ref_name='s_shape_test_data',
            generate=generate,
            device=device, dtype=dtype)

    def _run_bound_test(self, model, input_lb, input_ub, methods):
        """Helper to compute and verify bounds."""
        model = model.to(device=self.default_device, dtype=self.default_dtype)
        lirpa_model = BoundedModule(model, torch.empty_like(input_lb), device=self.default_device)
        ptb = PerturbationLpNorm(x_L=input_lb, x_U=input_ub)
        ptb_data = BoundedTensor(input_lb, ptb)

        for method in methods:
            lb, ub = lirpa_model.compute_bounds(x=(ptb_data,), method=method)
            verify_bounds(model, input_lb, input_ub, lb, ub)
            self.result.append((lb, ub))

    def test(self):
        self.result = []
        methods = ['CROWN', 'CROWN-OPTIMIZED']

        # ----- Test BoundSin -----
        model_sin = test_model(sin)
        start, end = -10, 10
        n_intervals = end - start - 1

        # Inputs as multiples of pi
        input_lb = torch.linspace(start, end - 1, n_intervals) * torch.pi
        input_ub = torch.linspace(start + 1, end, n_intervals) * torch.pi
        input_lb, input_ub = input_lb.unsqueeze(0), input_ub.unsqueeze(0)

        self._run_bound_test(model_sin, input_lb, input_ub, methods)

        # Inputs as multiples of pi / 2
        self._run_bound_test(model_sin, input_lb / 2, input_ub / 2, methods)

        # ----- Test BoundSigmoid -----
        model_sigmoid = test_model(sigmoid)
        input_lb = torch.tensor([[-2., -0.1]], device=self.default_device, dtype=self.default_dtype)
        input_ub = torch.tensor([[0.1, 2.]], device=self.default_device, dtype=self.default_dtype)

        self._run_bound_test(model_sigmoid, input_lb, input_ub, methods)

        # Check reference results
        self.check()


if __name__ == '__main__':
    # Change to generate=True when generating reference results
    testcase = TestSShaped(generate=False)
    testcase.setUp()
    testcase.test()
