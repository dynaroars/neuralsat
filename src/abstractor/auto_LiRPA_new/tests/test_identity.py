"""Test a model with an nn.Identity layer only"""
import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE 

class TestIdentity(TestCase):
    def __init__(self, methodName='runTest', device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, device=device, dtype=dtype)

    def test(self):
        model = nn.Sequential(nn.Identity())
        x = torch.randn(2, 10, device=self.default_device,
                        dtype=self.default_dtype)
        y = model(x)
        eps = 0.1
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(x, ptb)
        model = BoundedModule(model, x, device=self.default_device)
        y_l, y_u = model.compute_bounds()
        self.assertEqual(torch.Tensor(x), y)
        self.assertEqual(y_l, x - eps)
        self.assertEqual(y_u, x + eps)


if __name__ == '__main__':
    testcase = TestIdentity()
    testcase.test()
