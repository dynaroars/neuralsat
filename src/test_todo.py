"""Regression tests for every instance marked passed in /todo.md.

Run all:  python -m unittest test_todo -v
Run one:   python -m unittest test_todo.TestTodoPassed.test_cgan2026 -v

Requires vnncomp2026_benchmarks and GPU (same entry point as main.py).
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from helper.misc.result import ReturnStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / 'vnncomp2026_benchmarks' / 'benchmarks'
MAIN_PY = REPO_ROOT / 'neuralsat' / 'src' / 'main.py'
PYTHON = REPO_ROOT / 'venv' / 'bin' / 'python3'

# (test_name, onnx_rel, vnnlib_rel, timeout_s, expected_status)
# Keep in sync with todo.md lines marked [x] under "# Instances".
TODO_PASSED = [
    (
        'cgan2026',
        'cgan2026/1.0/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx',
        'cgan2026/1.0/vnnlib/cGAN_imgSz32_nCh_3_small_transformer_prop_1_input_eps_0.200_output_eps_0.025.vnnlib',
        120,
        ReturnStatus.SAT,
    ),
    (
        'challenging_certified_training_2026',
        'challenging_certified_training_2026/1.0/onnx/cifar10_eps8_wide_cnn7.onnx',
        'challenging_certified_training_2026/1.0/vnnlib/cifar10_eps8_wide_cnn7/cifar10_eps8_wide_cnn7_idx9436_sample0.vnnlib',
        120,
        ReturnStatus.SAT,
    ),
    (
        'cifar100_9460_oom',
        'cifar100_2024/1.0/onnx/CIFAR100_resnet_large.onnx',
        'cifar100_2024/1.0/vnnlib/CIFAR100_resnet_large_prop_idx_9460_sidx_4347_eps_0.0039.vnnlib',
        60,
        ReturnStatus.UNKNOWN,
    ),
    (
        'cifar100_1063_sat',
        'cifar100_2024/1.0/onnx/CIFAR100_resnet_large.onnx',
        'cifar100_2024/1.0/vnnlib/CIFAR100_resnet_large_prop_idx_1063_sidx_7948_eps_0.0039.vnnlib',
        120,
        ReturnStatus.SAT,
    ),
    (
        'ml4acopf_linear_nonresidual_prop13',
        'ml4acopf_2024/1.0/onnx/14_ieee_ml4acopf-linear-nonresidual.onnx',
        'ml4acopf_2024/1.0/vnnlib/14_ieee_prop13.vnnlib',
        120,
        ReturnStatus.UNSAT,
    ),
    (
        'ml4acopf_full_prop7',
        'ml4acopf_2024/1.0/onnx/14_ieee_ml4acopf.onnx',
        'ml4acopf_2024/1.0/vnnlib/14_ieee_prop7.vnnlib',
        120,
        ReturnStatus.UNSAT,
    ),
    (
        'ml4acopf_linear_residual_prop8',
        'ml4acopf_2024/1.0/onnx/14_ieee_ml4acopf-linear-residual.onnx',
        'ml4acopf_2024/1.0/vnnlib/14_ieee_prop8.vnnlib',
        120,
        ReturnStatus.UNSAT,
    ),
    (
        'ml4acopf_linear_residual_prop13',
        'ml4acopf_2024/1.0/onnx/14_ieee_ml4acopf-linear-residual.onnx',
        'ml4acopf_2024/1.0/vnnlib/14_ieee_prop13.vnnlib',
        120,
        ReturnStatus.UNSAT,
    ),
    (
        'safenlp_2024',
        'safenlp_2024/1.0/onnx/ruarobot/perturbations_0.onnx',
        'safenlp_2024/1.0/vnnlib/ruarobot/hyperrectangle_4454.vnnlib',
        60,
        ReturnStatus.SAT,
    ),
    (
        'soundnessbench_property_000',
        'soundnessbench_2026/1.0/onnx/model.onnx',
        'soundnessbench_2026/1.0/vnnlib/properties/property_000.vnnlib',
        60,
        ReturnStatus.SAT,
    ),
    (
        'soundnessbench_property_021',
        'soundnessbench_2026/1.0/onnx/model.onnx',
        'soundnessbench_2026/1.0/vnnlib/properties/property_021.vnnlib',
        60,
        ReturnStatus.SAT,
    ),
    (
        'vggnet16_spec5_scale',
        'vggnet16_2022/1.0/onnx/vgg16-7.onnx',
        'vggnet16_2022/1.0/vnnlib/spec5_scale.vnnlib',
        120,
        ReturnStatus.UNSAT,
    ),
    (
        'vggnet16_spec12_borzoi',
        'vggnet16_2022/1.0/onnx/vgg16-7.onnx',
        'vggnet16_2022/1.0/vnnlib/spec12_borzoi.vnnlib',
        120,
        ReturnStatus.UNKNOWN,
    ),
    (
        'yolo_prop_000037',
        'yolo_2023/1.0/onnx/TinyYOLO.onnx',
        'yolo_2023/1.0/vnnlib/TinyYOLO_prop_000037_eps_1_255.vnnlib',
        120,
        ReturnStatus.TIMEOUT,
    ),
]


def _run_main(net: Path, spec: Path, timeout: float, result_file: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault('GRB_LICENSE_FILE', str(Path.home() / 'gurobi.lic'))
    env.pop('NEURALSAT_DEBUG', None)
    proc = subprocess.run(
        [
            str(PYTHON),
            str(MAIN_PY),
            '--net', str(net),
            '--spec', str(spec),
            '--timeout', str(timeout),
            '--verbosity', '0',
            '--result_file', str(result_file),
        ],
        cwd=str(MAIN_PY.parent),
        env=env,
        capture_output=True,
        text=True,
    )
    if result_file.is_file():
        status = result_file.read_text().splitlines()[0].strip()
    else:
        status = 'missing'
    return proc.returncode, status


def _assert_todo_instance(test_case, name, onnx_rel, vnnlib_rel, timeout, expected):
    net = BENCH_ROOT / onnx_rel
    spec = BENCH_ROOT / vnnlib_rel
    if not net.is_file() or not spec.is_file():
        test_case.skipTest(f'missing benchmark files for {name}')
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        result_path = Path(f.name)
    try:
        exit_code, status = _run_main(net, spec, timeout, result_path)
    finally:
        result_path.unlink(missing_ok=True)
    test_case.assertEqual(
        status,
        expected,
        f'{name}: expected {expected}, got {status} (exit={exit_code})',
    )
    test_case.assertEqual(exit_code, 0, f'{name}: main.py exited with {exit_code}')


class TestTodoPassed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not BENCH_ROOT.is_dir():
            raise unittest.SkipTest(f'benchmarks not found: {BENCH_ROOT}')
        if not PYTHON.is_file():
            raise unittest.SkipTest(f'venv python not found: {PYTHON}')


def _make_todo_test(name, onnx_rel, vnnlib_rel, timeout, expected):
    def test_method(self):
        _assert_todo_instance(self, name, onnx_rel, vnnlib_rel, timeout, expected)

    test_method.__doc__ = f'todo.md [x] {name} -> {expected}'
    return test_method


for _name, _onnx, _spec, _timeout, _expected in TODO_PASSED:
    setattr(TestTodoPassed, f'test_{_name}', _make_todo_test(_name, _onnx, _spec, _timeout, _expected))


if __name__ == '__main__':
    unittest.main()
