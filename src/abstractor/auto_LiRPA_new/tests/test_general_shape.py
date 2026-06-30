""" Test inputs of general shapes (especially for matmul)"""
import torch
import torch.nn as nn
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.operators import BoundMatMul
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

BATCH_SIZE = 2

class GeneralShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_1 = nn.Parameter(torch.randn(3, 4))
        self.weight_2 = nn.Parameter(torch.randn(4, 3))
        self.weight_3 = nn.Parameter(torch.randn(3, 4))
        self.weight_4 = nn.Parameter(torch.randn(4, 4, 3))
        self.weight_5 = nn.Parameter(torch.randn(6, 3, 4))
        self.weight_6 = nn.Parameter(torch.randn(3, 5))
        self.relu = nn.ReLU()
        
    def forward(self, x, w):
        # Basic MatMul (B, 3) @ (3, 4) -> (B, 4)
        y1 = x.matmul(self.weight_1)

        # BoundUnsqueeze and BoundTile
        y2 = self.relu(y1)
        y2 = y2.unsqueeze(1).repeat(1, 5, 1)   # (B, 5, 4)
        y2 = y2.matmul(self.weight_2)   # (B, 5, 4) @ (4, 3) -> (B, 5, 3)

        # More dimensions on x
        y3 = self.relu(y2)
        y3 = y3.unsqueeze(1).repeat(1, 4, 1, 1)     # (B, 4, 5, 3)
        y3 = y3.matmul(self.weight_3)   # (B, 4, 5, 3) @ (3, 4) -> (B, 4, 5, 4)

        # More dimensions on weight
        y4 = self.relu(y3)
        y4 = y4.matmul(self.weight_4)   # (B, 4, 5, 4) @ (4, 4, 3) -> (B, 4, 5, 3)

        # Automatically broadcast x
        y5 = self.relu(y4)
        y5 = y5.unsqueeze(2)   # (B, 4, 1, 5, 3)
        y5 = y5.matmul(self.weight_5)   # (B, 4, 1, 5, 3) @ (6, 3, 4) -> (B, 4, 6, 5, 4)

        # Multiply with a weight with batch dimension
        y6 = self.relu(y5)
        y6 = y6.matmul(w)   # (B, 4, 6, 5, 4) @ (B, 4, 6, 4, 3) -> (B, 4, 6, 5, 3)

        # Swap x and weight
        y7 = self.relu(y6)
        y7 = self.weight_6.matmul(y7)   # (3, 5) @ (B, 4, 6, 5, 3) -> (B, 4, 6, 3, 3)

        return y7

class TestGeneralShape(TestCase):
    def __init__(self, methodName='runTest', seed=1, generate=False,
                 device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, seed, 'test_general_shape_data', generate, device=device, dtype=dtype)
        self.rtol = 1e-4

    def test(self):
        model = GeneralShapeModel().to(device=self.default_device, dtype=self.default_dtype)
        input = torch.randn(
            (BATCH_SIZE, 3), device=self.default_device, dtype=self.default_dtype)
        eps = 100
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(input, ptb)
        # w is an unperturbed input, but still have batch dimension
        w = torch.randn((BATCH_SIZE, 4, 6, 4, 3),
                        device=self.default_device, dtype=self.default_dtype)
        lirpa_model = BoundedModule(model, (x, w), device=self.default_device)

        lb, ub = lirpa_model.compute_bounds((x, w), method="backward")

        # # Test by sampling
        # sample_ptb = torch.rand(BATCH_SIZE, *input.shape[1:]) * 2 * eps - eps
        # sample_inputs = input[0] + sample_ptb
        # sample_output = model(sample_inputs, w)
        # assert (sample_output <= ub).all()
        # assert (sample_output >= lb).all()

        self.result = []
        for node in lirpa_model.nodes():
            if type(node) == BoundMatMul:
                self.result.append((node.lower, node.upper))
        self.result.append((lb, ub))

        self.check()

if __name__ == '__main__':
    testcase = TestGeneralShape(generate=False)
    testcase.setUp()
    testcase.test()