import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE


def ff(num_conv=2, num_mlp_only=None, pooling=False, activation="ReLU",
       hidden_size=256, input_ch=1, input_dim=28, num_classes=10, pool_kernel=3, pool_stride=1, pool_padding=1):
    activation = eval(f"nn.{activation}()")
    layers = []
    if num_conv:
        layers.append(nn.Conv2d(input_ch, 4, 3, stride=1, padding=1))
        layers.append(activation)
        num_channels = 4
        if pooling:
            layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
        if num_conv >= 2:
            layers.append(nn.Conv2d(4, 8, 3, stride=1, padding=1))
            layers.append(nn.ReLU())
            if pooling:
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
            num_channels = 8
        for _ in range(num_conv - 2):
            layers.append(nn.Conv2d(8, 8, 3, stride=1, padding=1))
            layers.append(nn.ReLU())
            if pooling:
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding))
        layers.append(nn.Flatten(1))

        # Calculate output size after pooling operations
        if pooling and num_conv > 0:
            pooled_dim = input_dim
            for _ in range(num_conv):
                pooled_dim = (pooled_dim + 2 * pool_padding - pool_kernel) // pool_stride + 1
            linear_input_size = num_channels * (pooled_dim ** 2)
        else:
            linear_input_size = num_channels * (input_dim ** 2)

        layers.append(nn.Linear(linear_input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_classes))
    else:
        layers.append(nn.Flatten(1))
        cur = input_ch * (input_dim ** 2)
        for _ in range(num_mlp_only - 1):
            layers.append(nn.Linear(cur, hidden_size))
            layers.append(activation)
            cur = hidden_size
        layers.append(nn.Linear(hidden_size, num_classes))
    return nn.Sequential(*layers)


def synthetic_net(input_ch, input_dim, **kwargs):
    return ff(input_ch=input_ch, input_dim=input_dim, num_classes=2, **kwargs)


def synthetic_4c2f_pool(input_ch, input_dim, **kwargs):
    return synthetic_net(input_ch, input_dim, num_conv=4, pooling=True, **kwargs)


class TestAvgPool(TestCase):
    def __init__(self, methodName='runTest', generate=False, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName,
            seed=1234, ref_name='avgpool_test_data',
            generate=generate, device=device, dtype=dtype)

    def test(self):
        test_configs = [
            {'input_ch': 1, 'input_dim': 5, 'hidden_size': 8, 'pool_kernel': 3, 'pool_stride': 1, 'pool_padding': 1},
            {'input_ch': 1, 'input_dim': 32, 'hidden_size': 16, 'pool_kernel': 2, 'pool_stride': 2, 'pool_padding': 0}
        ]

        self.result = []

        for config in test_configs:
            print(f"Testing config: {config}")

            model_ori = synthetic_4c2f_pool(**config)
            model_ori = model_ori.eval().to(self.default_device).to(self.default_dtype)

            x = torch.randn(8, config['input_ch'], config['input_dim'], config['input_dim'])

            ptb = PerturbationLpNorm(norm=np.inf, eps=100)
            x_bounded = BoundedTensor(x, ptb)

            print(f"  Testing with default conv_mode (patches)")
            model = BoundedModule(model_ori, x, device=self.default_device)

            lb_patches, ub_patches = model.compute_bounds(x=(x_bounded,), method='backward')
            print(f"    Patches mode - LB: {lb_patches}")
            print(f"    Patches mode - UB: {ub_patches}")

            self.result += [lb_patches, ub_patches]

            print(f"  Testing with conv_mode='matrix'")
            model_matrix = BoundedModule(model_ori, x, bound_opts={'conv_mode': 'matrix'})

            lb_matrix, ub_matrix = model_matrix.compute_bounds(x=(x_bounded,), method='backward')
            print(f"    Matrix mode - LB: {lb_matrix}")
            print(f"    Matrix mode - UB: {ub_matrix}")

            self.result += [lb_matrix, ub_matrix]

            lb_diff = torch.abs(lb_patches - lb_matrix).max().item()
            ub_diff = torch.abs(ub_patches - ub_matrix).max().item()
            print(f"    Max difference in LB between patches and matrix: {lb_diff}")
            print(f"    Max difference in UB between patches and matrix: {ub_diff}")

            assert torch.allclose(lb_patches, lb_matrix, atol=1e-6), f"Lower bounds not equivalent between patches and matrix modes"
            assert torch.allclose(ub_patches, ub_matrix, atol=1e-6), f"Upper bounds not equivalent between patches and matrix modes"
            print(f"    Matrix and patches modes produce equivalent results")
            print()

        self.check()


if __name__ == '__main__':
    testcase = TestAvgPool(generate=False)
    testcase.test()