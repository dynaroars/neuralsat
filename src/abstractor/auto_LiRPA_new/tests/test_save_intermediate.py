import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from testcase import _to, TestCase, DEFAULT_DEVICE, DEFAULT_DTYPE

class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 500),
            nn.Linear(500, 200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class TestSave(TestCase):
    def __init__(self, methodName='runTest', device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        super().__init__(methodName, device=device, dtype=dtype)

    def test(self, gen_ref=False):
        image = torch.randn(1, 3, 32, 32)
        image = image.to(device=self.default_device,
                         dtype=self.default_dtype) / 255.0
        model = test_model().to(device=self.default_device, dtype=self.default_dtype)

        bounded_model = BoundedModule(
            model, image, bound_opts={
                'optimize_bound_args': {'iteration': 2},
            }, device=self.default_device)

        ptb = PerturbationLpNorm(eps=3/255)
        x = BoundedTensor(image, ptb)
        bounded_model.compute_bounds(x=(x,), method='CROWN-Optimized')
        if self.default_dtype == torch.float32:
            data_path = 'data/'
        elif self.default_dtype == torch.float64:
            data_path = 'data_64/'
        data_path += 'test_save_data'

        save_dict = bounded_model.save_intermediate(
            save_path=data_path if gen_ref else None)

        if gen_ref:
            torch.save(save_dict, data_path)
            return

        ref_dict = torch.load(data_path)
        ref_dict = _to(
            ref_dict, device=self.default_device, dtype=self.default_dtype)


        for node in ref_dict.keys():
            assert torch.allclose(ref_dict[node][0], save_dict[node][0], atol=1e-5)
            assert torch.allclose(ref_dict[node][1], save_dict[node][1], atol=1e-5)


if __name__ == '__main__':
    testcase = TestSave()
    testcase.test()
