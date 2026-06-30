"""onnx2pytorch wrappers for Quantized ORT-optimized ONNX graphs."""

from __future__ import annotations

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

from helper.network.onnx2pytorch_lookup.operations.quantized_ort_helpers import (
    ort_global_average_pool,
    ort_layer_norm,
    ort_mlas_erf,
    ort_reduce,
    ort_softmax,
    ort_tanh,
    qgemm_forward,
    qlinear_add_forward,
    qlinear_matmul_forward,
    qlinear_softmax_forward,
    qlinearconv_forward,
    quant_dtype,
    quantize_linear_forward,
    to_int_list,
    torch_dequantize,
)


_QLINEARCONV_GRAD_MODE = "auto"
_QLINEARCONV_GRAD_MODES = {"ort", "exact_ste", "surrogate", "measure", "auto"}
_QLINEARCONV_SURROGATE_PATTERNS = ("QLinearConv_",)
_QLINEARCONV_EXACT_PATTERNS = (
    "/encoder/conv1/",
    "/encoder/conv2/",
    "/video_backbone/stem/",
    "/layer1/layer1.0/conv2/",
    "/layer1/layer1.1/conv2/",
    "/layer2/layer2.0/conv2/",
    "/layer2/layer2.0/downsample/",
    "/layer3/layer3.0/downsample/",
)
_QLINEARMATMUL_MODE = "auto"
_QLINEARMATMUL_MODES = {"ort", "exact_ste", "torch", "measure", "auto"}
_QLINEARMATMUL_TORCH_PATTERNS = (
    "/encoder/layers.0/self_attn/MatMul_output_0",
    "/encoder/layers.0/self_attn/MatMul_1_output_0",
    "/encoder/layers.1/self_attn/MatMul_output_0",
    "/encoder/layers.1/self_attn/MatMul_1_output_0",
    "/encoder/layers.2/self_attn/MatMul_output_0",
    "/encoder/layers.2/self_attn/MatMul_1_output_0",
    "/encoder/layers.3/self_attn/MatMul_output_0",
    "/encoder/layers.3/self_attn/MatMul_1_output_0",
)
_QLINEARMATMUL_EXACT_PATTERNS = ()


def _attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def _string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value if str(item)]


def _configured_patterns(quirks, name, defaults):
    return list(defaults) + _string_list(quirks.get(name))


def _matches_any_pattern(name, patterns):
    return any(pattern in name for pattern in patterns)


def _configure_qlinearconv_auto_policy(module, op_name, quirks):
    module.debug_name = op_name
    surrogate_patterns = _configured_patterns(
        quirks, "qlinearconv_surrogate_patterns",
        _QLINEARCONV_SURROGATE_PATTERNS)
    exact_patterns = _configured_patterns(
        quirks, "qlinearconv_exact_patterns", _QLINEARCONV_EXACT_PATTERNS)
    if _matches_any_pattern(op_name, surrogate_patterns):
        module.force_exact_in_pgd = False
    if _matches_any_pattern(op_name, exact_patterns):
        module.force_exact_in_pgd = True


def _configure_qlinearmatmul_auto_policy(module, op_name, quirks):
    module.debug_name = op_name
    torch_patterns = _configured_patterns(
        quirks, "qlinearmatmul_torch_patterns",
        _QLINEARMATMUL_TORCH_PATTERNS)
    exact_patterns = _configured_patterns(
        quirks, "qlinearmatmul_exact_patterns",
        _QLINEARMATMUL_EXACT_PATTERNS)
    if _matches_any_pattern(op_name, torch_patterns):
        module.force_torch_in_pgd = True
    if _matches_any_pattern(op_name, exact_patterns):
        module.force_torch_in_pgd = False


class QuantizeLinear(nn.Module):
    """ONNX QuantizeLinear module for onnx2pytorch conversion."""

    def forward(self, x, scale, zero_point):
        return quantize_linear_forward(x, scale, zero_point)


class BroadcastAdd(nn.Module):
    """Broadcast-safe ONNX Add replacement for optimized quantized graphs."""

    def forward(self, *inputs):
        output = inputs[0]
        for value in inputs[1:]:
            output = output + value
        return output


class Slice(nn.Module):
    """ONNX Slice with opset-13 input-form starts/ends/axes/steps."""

    def forward(self, data, starts, ends, axes=None, steps=None):
        starts = to_int_list(starts)
        ends = to_int_list(ends)
        axes = to_int_list(axes)
        steps = to_int_list(steps)
        if axes is None:
            axes = list(range(len(starts)))
        if steps is None:
            steps = [1] * len(starts)

        slices = [slice(None)] * data.ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            if axis < 0:
                axis += data.ndim
            dim_size = data.shape[axis]
            if end >= np.iinfo(np.int64).max // 2:
                end = None
            elif end <= np.iinfo(np.int64).min // 2:
                end = None
            else:
                end = max(end + dim_size, 0) if end < 0 else min(end, dim_size)
            if start <= np.iinfo(np.int64).min // 2:
                start = None
            elif start < 0:
                start = max(start + dim_size, 0)
            slices[axis] = slice(start, end, step)
        return data[tuple(slices)]


class Reduce(nn.Module):
    """ORT ReduceMean/ReduceSum module with analytic gradients."""

    def __init__(self, op_type, axes=None, keepdims=1):
        super().__init__()
        self.op_type = op_type
        self.axes = None if axes is None else [int(v) for v in axes]
        self.keepdims = bool(keepdims)

    def forward(self, data, axes=None):
        axes = to_int_list(axes) if axes is not None else self.axes
        return ort_reduce(self.op_type, data, axes, self.keepdims)


class Erf(nn.Module):
    """ORT Erf module."""

    def forward(self, x):
        return ort_mlas_erf(x)


class Tanh(nn.Module):
    """ORT Tanh module."""

    def forward(self, x):
        return ort_tanh(x)


class Softmax(nn.Module):
    """ORT Softmax module."""

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = int(axis)

    def forward(self, x):
        return ort_softmax(x, self.axis)


class GlobalAveragePool(nn.Module):
    """ORT GlobalAveragePool module."""

    def forward(self, x):
        return ort_global_average_pool(x)


class DequantizeLinear(nn.Module):
    """ONNX DequantizeLinear module for onnx2pytorch conversion."""

    def __init__(self, axis=1):
        super().__init__()
        self.axis = int(axis)

    def forward(self, x, scale, zero_point=None):
        if zero_point is None:
            zero_point = torch.as_tensor(0, dtype=x.dtype, device=x.device)
        return torch_dequantize(x, scale, zero_point, self.axis)


class QLinearAdd(nn.Module):
    """Torch QLinearAdd module with exact forward and surrogate gradients."""

    def forward(self, a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero):
        return qlinear_add_forward(
            a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero)


class QLinearMatMul(nn.Module):
    """ONNX QLinearMatMul module with exact forward and surrogate gradients."""

    def __init__(self, mode=None):
        super().__init__()
        self.mode = (mode or _QLINEARMATMUL_MODE).lower()
        if self.mode not in _QLINEARMATMUL_MODES:
            raise ValueError(
                f"Unsupported QLinearMatMul mode: {self.mode}. "
                f"Expected one of {sorted(_QLINEARMATMUL_MODES)}.")
        self.force_torch_in_pgd = False
        self.debug_name = None
        self.last_mismatch = None

    def forward(self, a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero):
        return qlinear_matmul_forward(
            self, a, a_scale, a_zero, b, b_scale, b_zero,
            y_scale, y_zero, quant_dtype(y_zero))


class QGemm(nn.Module):
    """ORT QGemm module for quantized and float-output forms."""

    def __init__(self, alpha=1.0, trans_a=0, trans_b=0, has_bias=True):
        super().__init__()
        self.alpha = float(alpha)
        self.trans_a = int(trans_a)
        self.trans_b = int(trans_b)
        self.has_bias = bool(has_bias)

    def forward(self, *inputs):
        return qgemm_forward(self, *inputs)


class QLinearConv(nn.Module):
    """ONNX QLinearConv module with exact forward and surrogate gradients."""

    def __init__(
            self, strides=None, pads=None, dilations=None, group=1,
            channels_last=0, has_bias=True, grad_mode=None):
        super().__init__()
        self.strides = None if strides is None else tuple(int(v) for v in strides)
        self.pads = None if pads is None else tuple(int(v) for v in pads)
        self.dilations = (
            None if dilations is None else tuple(int(v) for v in dilations))
        self.group = int(group)
        self.channels_last = bool(channels_last)
        self.has_bias = bool(has_bias)
        self.grad_mode = (grad_mode or _QLINEARCONV_GRAD_MODE).lower()
        if self.grad_mode not in _QLINEARCONV_GRAD_MODES:
            raise ValueError(
                f"Unsupported QLinearConv grad mode: {self.grad_mode}. "
                f"Expected one of {sorted(_QLINEARCONV_GRAD_MODES)}.")
        self.force_exact_in_pgd = False
        self.debug_name = None
        self.last_mismatch = None

    def forward(
            self, x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero,
            bias=None):
        return qlinearconv_forward(
            self, x, x_scale, x_zero, w, w_scale, w_zero,
            y_scale, y_zero, bias)


class QLinearSoftmax(nn.Module):
    """ORT QLinearSoftmax module with exact forward and surrogate gradients."""

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = int(axis)

    def forward(self, x, x_scale, x_zero, y_scale, y_zero):
        return qlinear_softmax_forward(
            x, x_scale, x_zero, y_scale, y_zero,
            quant_dtype(y_zero), self.axis)


class BiasGelu(nn.Module):
    """BiasGelu module using the native PyTorch forward path."""

    def forward(self, x, bias):
        return F.gelu(x + bias)


class LayerNormalization(nn.Module):
    """ORT LayerNormalization module."""

    def __init__(self, axis=-1, epsilon=1e-5, simplified=False):
        super().__init__()
        self.axis = int(axis)
        self.epsilon = float(epsilon)
        self.simplified = bool(simplified)

    def forward(self, x, weight, bias=None):
        return ort_layer_norm(
            x, weight, bias, self.axis, self.epsilon, self.simplified)


class SkipLayerNormalization(nn.Module):
    """ORT SkipLayerNormalization module."""

    def __init__(self, axis=-1, epsilon=1e-5):
        super().__init__()
        self.axis = int(axis)
        self.epsilon = float(epsilon)

    def forward(self, x, skip, weight, bias=None):
        return ort_layer_norm(
            x + skip, weight, bias, self.axis, self.epsilon,
            simplified=False)


def _make_op(node, quirks):
    op_type = node.op_type
    if op_type == "Add":
        return BroadcastAdd()
    if op_type == "Slice":
        return Slice()
    if op_type == "Erf":
        return Erf()
    if op_type == "Tanh":
        return Tanh()
    if op_type == "Softmax":
        return Softmax(axis=_attr(node, "axis", -1))
    if op_type in ("ReduceMean", "ReduceSum"):
        return Reduce(
            op_type,
            axes=_attr(node, "axes", None),
            keepdims=_attr(node, "keepdims", 1))
    if op_type == "GlobalAveragePool":
        return GlobalAveragePool()
    if op_type == "QuantizeLinear":
        return QuantizeLinear()
    if op_type == "DequantizeLinear":
        return DequantizeLinear(axis=_attr(node, "axis", 1))
    if op_type == "QLinearAdd":
        return QLinearAdd()
    if op_type == "QLinearMatMul":
        return QLinearMatMul(mode=quirks.get("qlinearmatmul_mode"))
    if op_type == "QGemm":
        has_bias = len(node.input) > 6 and node.input[6] != ""
        _remove_empty_inputs(node)
        return QGemm(
            alpha=_attr(node, "alpha", 1.0),
            trans_a=_attr(node, "transA", 0),
            trans_b=_attr(node, "transB", 0),
            has_bias=has_bias)
    if op_type == "QLinearConv":
        has_bias = len(node.input) > 8 and node.input[8] != ""
        _remove_empty_inputs(node)
        return QLinearConv(
            strides=_attr(node, "strides", None),
            pads=_attr(node, "pads", None),
            dilations=_attr(node, "dilations", None),
            group=_attr(node, "group", 1),
            channels_last=_attr(node, "channels_last", 0),
            has_bias=has_bias,
            grad_mode=quirks.get("qlinearconv_grad_mode"))
    if op_type == "QLinearSoftmax":
        return QLinearSoftmax(axis=_attr(node, "axis", -1))
    if op_type == "BiasGelu":
        return BiasGelu()
    if op_type == "LayerNormalization":
        return LayerNormalization(
            axis=_attr(node, "axis", -1),
            epsilon=_attr(node, "epsilon", 1e-5),
            simplified=False)
    if op_type == "SimplifiedLayerNormalization":
        return LayerNormalization(
            axis=_attr(node, "axis", -1),
            epsilon=_attr(node, "epsilon", 1e-5),
            simplified=True)
    if op_type == "SkipLayerNormalization":
        return SkipLayerNormalization(
            axis=_attr(node, "axis", -1),
            epsilon=_attr(node, "epsilon", 1e-5))
    return None


def make_quantized_op(node, quirks=None):
    """Return a Quantized-compatible op for ``node``, or ``None``."""
    quirks = quirks if isinstance(quirks, dict) else {}
    op = _make_op(node, quirks)
    if op is None:
        return None
    op_name = f"{node.op_type}_{node.output[0]}"
    if isinstance(op, QLinearConv):
        _configure_qlinearconv_auto_policy(op, op_name, quirks)
    if isinstance(op, QLinearMatMul):
        _configure_qlinearmatmul_auto_policy(op, op_name, quirks)
    return op


def _remove_empty_inputs(node):
    if "" not in node.input:
        return
    kept = [name for name in node.input if name]
    del node.input[:]
    node.input.extend(kept)
