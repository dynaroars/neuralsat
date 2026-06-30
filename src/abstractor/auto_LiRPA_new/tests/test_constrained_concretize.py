"""Test optimized bounds in simple_verification."""
import torch
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class ConstrainedConcretizeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.tensor([[1., -1.], [2., -1.]])
        self.w2 = torch.tensor([[1., -1.]])

    def forward(self, x):
        z1 = x.matmul(self.w1.t())
        hz1 = torch.nn.functional.relu(z1)
        z2 = hz1.matmul(self.w2.t())
        return z2

class TestConstrainedConcretize(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, 1, "test_constrained_concretize", generate, device=device, dtype=dtype)

    def test(self):
        model = ConstrainedConcretizeModel().to(self.default_device).to(self.default_dtype)
        # Input x.
        x = torch.tensor([[1., 1.]], dtype=self.default_dtype, device=self.default_device)
        # Lower and upper bounds of x.
        lower = torch.tensor([[-1., -2.]], dtype=self.default_dtype, device=self.default_device)
        upper = torch.tensor([[2., 1.]], dtype=self.default_dtype, device=self.default_device)

        # Wrap model with auto_LiRPA for bound computation.
        # The second parameter is for constructing the trace of the computational graph,
        # and its content is not important.

        lirpa_model = BoundedModule(model, torch.empty_like(x))
        pred = lirpa_model(x)
        print(f'Model prediction: {pred.item()}')

        # Compute bounds using LiRPA using the given lower and upper bounds.
        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper)
        bounded_x = BoundedTensor(x, ptb)

        # Compute bounds.
        lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds: lower={lb.item()}, upper={ub.item()}')

        # Add a new constraint of :
        #    1*x_0 + 1*x_1 + 2 <= 0
        constraint_a = torch.tensor([[[1.0, 1.0]]], dtype=self.default_dtype, device=self.default_device)
        constraint_b = torch.tensor([[2.0]], dtype=self.default_dtype, device=self.default_device)
        constraints = (constraint_a, constraint_b)

        norm = float("inf")
        ptb = PerturbationLpNorm(norm = norm, x_L=lower, x_U=upper, constraints=constraints)
        bounded_x = BoundedTensor(x, ptb)
        # Compute bounds.
        constrained_lb, constrained_ub = lirpa_model.compute_bounds(x=(bounded_x,), method='CROWN')
        print(f'CROWN bounds (with constraints): lower={constrained_lb.item()}, upper={constrained_ub.item()}')

        self.result = (lb, ub, constrained_lb, constrained_ub)
        self.check()

if __name__ == '__main__':
    testcase = TestConstrainedConcretize(generate=True)
    testcase.setUp()
    testcase.test()