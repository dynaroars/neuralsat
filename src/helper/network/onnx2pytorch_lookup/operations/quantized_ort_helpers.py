"""ONNX Runtime exact forward helpers for fused quantized custom ops."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper


_ORT_QUANTIZE_SESSIONS = {}
_ORT_QLINEAR_MATMUL_SESSIONS = {}
_ORT_QLINEAR_CONV_SESSIONS = {}
_ORT_QGEMM_SESSIONS = {}
_ORT_QLINEAR_SOFTMAX_SESSIONS = {}
_ORT_ERF_SESSION = None
_ORT_TANH_SESSION = None
_ORT_SOFTMAX_SESSIONS = {}
_ORT_REDUCE_SESSIONS = {}
_ORT_GLOBAL_AVG_POOL_SESSION = None
_ORT_LAYER_NORM_SESSIONS = {}

_QUANT_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.int8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
)


def as_float(x):
    return x.to(dtype=torch.get_default_dtype())


def reshape_quant_param(param, target, axis):
    if not isinstance(param, torch.Tensor):
        param = torch.as_tensor(param, device=target.device)
    if param.numel() == 1:
        return as_float(param).to(device=target.device).reshape(())
    if axis < 0:
        axis += target.ndim
    shape = [1] * target.ndim
    shape[axis] = param.numel()
    return as_float(param).to(device=target.device).reshape(shape)


def reshape_channel_param(param, target):
    if not isinstance(param, torch.Tensor):
        param = torch.as_tensor(param, device=target.device)
    param = as_float(param).to(device=target.device)
    if param.numel() == 1:
        return param.reshape(())
    return param.reshape(1, -1, *([1] * (target.ndim - 2)))


def torch_dequantize(x, scale, zero_point, axis):
    scale = reshape_quant_param(scale, x, axis)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.as_tensor(0, device=x.device, dtype=x.dtype)
    zero_point = reshape_quant_param(zero_point, x, axis)
    return (as_float(x) - zero_point) * scale


def quantize_linear_torch_forward(x, scale, zero_point, qmin, qmax):
    dtype = x.dtype if torch.is_floating_point(x) else torch.get_default_dtype()
    scale = as_float(scale).to(device=x.device, dtype=dtype)
    zero_point = zero_point.to(device=x.device, dtype=dtype)
    q = torch.round(x.to(dtype=dtype) / scale) + zero_point
    return torch.clamp(
        q,
        min=qmin.to(device=x.device, dtype=dtype),
        max=qmax.to(device=x.device, dtype=dtype))


def to_int_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, int):
        value = [value]
    return [int(v) for v in value]


def quant_dtype(zero_point):
    return dtype_from_zero_point(zero_point)


def quant_bound_tensors(zero_point, reference):
    qmin, qmax = quant_bounds(quant_dtype(zero_point))
    dtype = (
        reference.dtype if torch.is_floating_point(reference)
        else torch.get_default_dtype())
    return (
        torch.as_tensor(qmin, device=reference.device, dtype=dtype),
        torch.as_tensor(qmax, device=reference.device, dtype=dtype),
    )


def as_param_float(value, reference):
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=reference.device)
    return as_float(value).to(device=reference.device, dtype=torch.float32)


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def empty_bias(reference):
    return torch.empty(0, dtype=reference.dtype, device=reference.device)


def pad_conv_input(x, pads):
    if not pads or not any(pads):
        return x
    dims = len(pads) // 2
    pad = []
    for left, right in reversed(list(zip(pads[:dims], pads[dims:]))):
        pad.extend([left, right])
    return F.pad(x, pad)


def quant_bounds(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return 0.0, 255.0
    if dtype == np.dtype(np.int8):
        return -128.0, 127.0
    if dtype == np.dtype(np.uint16):
        return 0.0, 65535.0
    if dtype == np.dtype(np.int16):
        return -32768.0, 32767.0
    raise RuntimeError(f"Unsupported quantized dtype: {dtype}.")


def dtype_from_quant_bounds(qmin, qmax):
    qmin_value = float(_first_item(qmin))
    qmax_value = float(_first_item(qmax))
    if qmin_value == 0.0 and qmax_value == 255.0:
        return np.dtype(np.uint8)
    if qmin_value == -128.0 and qmax_value == 127.0:
        return np.dtype(np.int8)
    if qmin_value == 0.0 and qmax_value == 65535.0:
        return np.dtype(np.uint16)
    if qmin_value == -32768.0 and qmax_value == 32767.0:
        return np.dtype(np.int16)
    raise RuntimeError(
        f"Cannot infer quantized dtype from bounds ({qmin_value}, {qmax_value}).")


def dtype_from_zero_point(zero_point, qmin=None, qmax=None):
    if zero_point is not None:
        dtype = np.dtype(_to_numpy(zero_point).dtype)
        if dtype in _QUANT_DTYPES:
            return dtype
    if qmin is not None and qmax is not None:
        return dtype_from_quant_bounds(qmin, qmax)
    if zero_point is not None and np.any(_to_numpy(zero_point) < 0):
        return np.dtype(np.int8)
    return np.dtype(np.uint8)


def onnx_tensor_type(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return TensorProto.FLOAT
    if dtype == np.dtype(np.int64):
        return TensorProto.INT64
    if dtype == np.dtype(np.int32):
        return TensorProto.INT32
    if dtype == np.dtype(np.uint8):
        return TensorProto.UINT8
    if dtype == np.dtype(np.int8):
        return TensorProto.INT8
    if dtype == np.dtype(np.uint16):
        return TensorProto.UINT16
    if dtype == np.dtype(np.int16):
        return TensorProto.INT16
    raise RuntimeError(f"Unsupported ONNX tensor dtype: {dtype}.")


def as_quant_numpy(value, dtype):
    dtype = np.dtype(dtype)
    array = _to_numpy(value)
    if array.dtype == dtype:
        return array
    if dtype not in _QUANT_DTYPES:
        return np.asarray(array.astype(dtype, copy=False), dtype=dtype)
    qmin, qmax = quant_bounds(dtype)
    return np.asarray(np.rint(array).clip(qmin, qmax).astype(dtype), dtype=dtype)


def ort_quantize_forward(x, scale, zero_point, qmin, qmax):
    y_dtype = dtype_from_zero_point(zero_point, qmin, qmax)
    session = _ort_quantize_session(y_dtype)
    y = session.run(None, {
        "x": _as_float_numpy(x),
        "scale": _as_float_numpy(scale),
        "zero_point": as_quant_numpy(zero_point, y_dtype),
    })[0]
    return _float_tensor_like(y, x)


def ort_qlinear_matmul_forward(a, a_scale, a_zero, b, b_scale, b_zero,
                               y_scale, y_zero, qmin, qmax):
    a_dtype = _dtype_from_value_or_zero_point(a, a_zero)
    b_dtype = _dtype_from_value_or_zero_point(
        b, b_zero, prefer_signed_float_weight=True)
    y_dtype = dtype_from_zero_point(y_zero, qmin, qmax)
    session = _ort_qlinear_matmul_session(a_dtype, b_dtype, y_dtype)
    y = session.run(None, {
        "a": as_quant_numpy(a, a_dtype),
        "a_scale": _as_float_numpy(a_scale),
        "a_zero": as_quant_numpy(a_zero, a_dtype),
        "b": as_quant_numpy(b, b_dtype),
        "b_scale": _as_float_numpy(b_scale),
        "b_zero": as_quant_numpy(b_zero, b_dtype),
        "y_scale": _as_float_numpy(y_scale),
        "y_zero": as_quant_numpy(y_zero, y_dtype),
    })[0]
    return _float_tensor_like(y, a)


def ort_qgemm_forward(a, a_scale, a_zero, b, b_scale, b_zero, bias,
                      y_scale, y_zero, qmin, qmax, alpha, trans_a, trans_b):
    a_dtype = _dtype_from_value_or_zero_point(a, a_zero)
    b_dtype = _dtype_from_value_or_zero_point(
        b, b_zero, prefer_signed_float_weight=True)
    y_dtype = dtype_from_zero_point(y_zero, qmin, qmax)
    bias_dtype = _bias_dtype(bias)
    session = _ort_qgemm_session(
        a_dtype, b_dtype, y_dtype, bias_dtype,
        _as_float_scalar(alpha), _as_int(trans_a), _as_int(trans_b))
    feed = {
        "a": as_quant_numpy(a, a_dtype),
        "a_scale": _as_float_numpy(a_scale),
        "a_zero": as_quant_numpy(a_zero, a_dtype),
        "b": as_quant_numpy(b, b_dtype),
        "b_scale": _as_float_numpy(b_scale),
        "b_zero": as_quant_numpy(b_zero, b_dtype),
        "y_scale": _as_float_numpy(y_scale),
        "y_zero": as_quant_numpy(y_zero, y_dtype),
    }
    if bias_dtype is not None:
        feed["bias"] = _as_bias_numpy(bias, bias_dtype)
    y = session.run(None, feed)[0]
    return _float_tensor_like(y, a)


def ort_qlinear_conv_forward(x, x_scale, x_zero, w, w_scale, w_zero,
                             y_scale, y_zero, bias, qmin, qmax,
                             strides, pads, dilations, group, channels_last):
    x_dtype = _dtype_from_value_or_zero_point(x, x_zero)
    w_dtype = _dtype_from_value_or_zero_point(
        w, w_zero, prefer_signed_float_weight=True)
    y_dtype = dtype_from_zero_point(y_zero, qmin, qmax)
    bias_dtype = _bias_dtype(bias)
    strides = _as_int_tuple(strides)
    pads = _as_int_tuple(pads)
    dilations = _as_int_tuple(dilations)
    group = _as_int(group)
    channels_last = bool(_as_int(channels_last))
    x_for_ort = _to_channels_first(x, _rank(w)) if channels_last else x
    session = _ort_qlinear_conv_session(
        x_dtype, w_dtype, y_dtype, bias_dtype, _rank(w),
        strides, pads, dilations, group, False)
    feed = {
        "x": as_quant_numpy(x_for_ort, x_dtype),
        "x_scale": _as_float_numpy(x_scale),
        "x_zero": as_quant_numpy(x_zero, x_dtype),
        "w": as_quant_numpy(w, w_dtype),
        "w_scale": _as_float_numpy(w_scale),
        "w_zero": as_quant_numpy(w_zero, w_dtype),
        "y_scale": _as_float_numpy(y_scale),
        "y_zero": as_quant_numpy(y_zero, y_dtype),
    }
    if bias_dtype is not None:
        feed["bias"] = _as_bias_numpy(bias, bias_dtype)
    y = session.run(None, feed)[0]
    if channels_last:
        y = _to_channels_last(y)
    return _float_tensor_like(y, x)


def ort_qlinear_softmax_forward(x, x_scale, x_zero, y_scale, y_zero,
                                qmin, qmax, axis):
    x_dtype = _dtype_from_value_or_zero_point(x, x_zero)
    y_dtype = dtype_from_zero_point(y_zero, qmin, qmax)
    axis = _as_int(axis)
    session = _ort_qlinear_softmax_session(x_dtype, y_dtype, axis)
    y = session.run(None, {
        "x": as_quant_numpy(x, x_dtype),
        "x_scale": _as_float_numpy(x_scale),
        "x_zero": as_quant_numpy(x_zero, x_dtype),
        "y_scale": _as_float_numpy(y_scale),
        "y_zero": as_quant_numpy(y_zero, y_dtype),
    })[0]
    return _float_tensor_like(y, x)


class _ORTQuantizeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        ctx.save_for_backward(scale)
        return ort_quantize_forward(x, scale, zero_point, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        (scale,) = ctx.saved_tensors
        return (
            grad_output / scale.to(device=grad_output.device),
            None, None, None, None)


def ort_quantize(x, scale, zero_point, zero_dtype):
    qmin, qmax = quant_bounds(np.dtype(zero_dtype))
    dtype = as_float(x).dtype
    qmin = torch.as_tensor(qmin, device=x.device, dtype=dtype)
    qmax = torch.as_tensor(qmax, device=x.device, dtype=dtype)
    scale = as_float(scale).to(device=x.device)
    zero_point = zero_point.to(device=x.device)
    if x.requires_grad and torch.is_grad_enabled():
        return _ORTQuantizeOp.apply(x, scale, zero_point, qmin, qmax)
    return ort_quantize_forward(x, scale, zero_point, qmin, qmax).to(
        dtype=torch.get_default_dtype())


def _ort_erf_session():
    global _ORT_ERF_SESSION  # pylint: disable=global-statement
    if _ORT_ERF_SESSION is None:
        import onnxruntime as ort
        graph = helper.make_graph(
            [helper.make_node("Erf", ["x"], ["y"], name="erf")],
            "quantized_ort_erf",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_ERF_SESSION = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_ERF_SESSION


class _ORTErfOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = _ort_erf_session().run(None, {"x": x.detach().cpu().numpy()})[0]
        return torch.from_numpy(y).to(device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (2.0 / math.sqrt(math.pi)) * torch.exp(-x.square())


def ort_mlas_erf(x):
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        return torch.erf(x)
    return _ORTErfOp.apply(x)


def _ort_tanh_session():
    global _ORT_TANH_SESSION  # pylint: disable=global-statement
    if _ORT_TANH_SESSION is None:
        import onnxruntime as ort
        graph = helper.make_graph(
            [helper.make_node("Tanh", ["x"], ["y"], name="tanh")],
            "quantized_ort_tanh",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_TANH_SESSION = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_TANH_SESSION


class _ORTTanhOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _ort_tanh_session().run(None, {"x": x.detach().cpu().numpy()})[0]
        y = torch.from_numpy(y).to(device=x.device)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        return grad_output * (1.0 - y.square())


def ort_tanh(x):
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        return torch.tanh(x)
    return _ORTTanhOp.apply(x)


def _ort_softmax_session(axis):
    axis = int(axis)
    if axis not in _ORT_SOFTMAX_SESSIONS:
        import onnxruntime as ort
        graph = helper.make_graph(
            [helper.make_node(
                "Softmax", ["x"], ["y"], name="softmax", axis=axis)],
            "quantized_ort_softmax",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_SOFTMAX_SESSIONS[axis] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_SOFTMAX_SESSIONS[axis]


class _ORTSoftmaxOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis):
        y = _ort_softmax_session(axis).run(
            None, {"x": x.detach().cpu().numpy()})[0]
        y = torch.from_numpy(y).to(device=x.device)
        ctx.save_for_backward(y)
        ctx.axis = int(axis)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        dot = (grad_output * y).sum(dim=ctx.axis, keepdim=True)
        return y * (grad_output - dot), None


def ort_softmax(x, axis):
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        return F.softmax(x, dim=int(axis))
    return _ORTSoftmaxOp.apply(x, int(axis))


def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    axes = tuple(to_int_list(axes))
    return tuple(axis if axis >= 0 else axis + ndim for axis in axes)


def _ort_reduce_session(op_type, axes, keepdims):
    key = (op_type, tuple(axes) if axes is not None else None, bool(keepdims))
    if key not in _ORT_REDUCE_SESSIONS:
        import onnxruntime as ort
        inputs = ["x"]
        graph_inputs = [helper.make_tensor_value_info(
            "x", TensorProto.FLOAT, None)]
        initializers = []
        attrs = {"keepdims": int(keepdims)}
        if axes is not None and op_type == "ReduceSum":
            inputs.append("axes")
            initializers.append(numpy_helper.from_array(
                np.asarray(axes, dtype=np.int64), "axes"))
            attrs["noop_with_empty_axes"] = 0
        elif axes is not None:
            attrs["axes"] = list(axes)
        graph = helper.make_graph(
            [helper.make_node(op_type, inputs, ["y"], name="reduce", **attrs)],
            "quantized_ort_reduce",
            graph_inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            initializers)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_REDUCE_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_REDUCE_SESSIONS[key]


class _ORTReduceOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op_type, axes, keepdims):
        normalized_axes = _normalize_axes(axes, x.ndim)
        session = _ort_reduce_session(op_type, normalized_axes, keepdims)
        y = session.run(None, {"x": x.detach().cpu().numpy()})[0]
        ctx.op_type = op_type
        ctx.axes = normalized_axes
        ctx.keepdims = bool(keepdims)
        ctx.input_shape = tuple(x.shape)
        return torch.from_numpy(y).to(device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output
        if not ctx.keepdims:
            for axis in sorted(ctx.axes):
                grad = grad.unsqueeze(axis)
        grad = grad.expand(ctx.input_shape)
        if ctx.op_type == "ReduceMean":
            size = math.prod(ctx.input_shape[axis] for axis in ctx.axes)
            grad = grad / float(size)
        return grad, None, None, None


def ort_reduce(op_type, x, axes, keepdims):
    normalized_axes = _normalize_axes(axes, x.ndim)
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        dim = tuple(normalized_axes)
        if op_type == "ReduceMean":
            return torch.mean(x, dim=dim, keepdim=bool(keepdims))
        return torch.sum(x, dim=dim, keepdim=bool(keepdims))
    return _ORTReduceOp.apply(x, op_type, normalized_axes, bool(keepdims))


def _torch_global_average_pool(x):
    dims = tuple(range(2, x.ndim))
    return torch.mean(x, dim=dims, keepdim=True)


def _ort_global_average_pool_session():
    global _ORT_GLOBAL_AVG_POOL_SESSION  # pylint: disable=global-statement
    if _ORT_GLOBAL_AVG_POOL_SESSION is None:
        import onnxruntime as ort
        graph = helper.make_graph(
            [helper.make_node(
                "GlobalAveragePool", ["x"], ["y"],
                name="global_average_pool")],
            "quantized_ort_global_average_pool",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_GLOBAL_AVG_POOL_SESSION = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_GLOBAL_AVG_POOL_SESSION


class _ORTGlobalAveragePoolOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = _ort_global_average_pool_session().run(
            None, {"x": x.detach().cpu().numpy()})[0]
        ctx.input_shape = tuple(x.shape)
        return torch.from_numpy(y).to(device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        spatial_shape = ctx.input_shape[2:]
        grad = grad_output.expand(ctx.input_shape)
        return grad / float(math.prod(spatial_shape))


def ort_global_average_pool(x):
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        return _torch_global_average_pool(x)
    return _ORTGlobalAveragePoolOp.apply(x)


def torch_layer_norm(x, weight, bias=None, axis=-1, epsilon=1e-5,
                     simplified=False):
    dims = tuple(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    centered = x if simplified else x - x.mean(dim=dims, keepdim=True)
    variance = (centered * centered).mean(dim=dims, keepdim=True)
    y = centered * torch.reciprocal(torch.sqrt(variance + epsilon))
    y = y * weight
    if bias is not None:
        y = y + bias
    return y


def _ort_layer_norm_session(axis, epsilon, has_bias, simplified):
    key = (axis, float(epsilon), has_bias, simplified)
    if key not in _ORT_LAYER_NORM_SESSIONS:
        import onnxruntime as ort
        op_type = (
            "SimplifiedLayerNormalization" if simplified
            else "LayerNormalization")
        inputs = ["x", "weight"]
        graph_inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("weight", TensorProto.FLOAT, None),
        ]
        if has_bias:
            inputs.append("bias")
            graph_inputs.append(helper.make_tensor_value_info(
                "bias", TensorProto.FLOAT, None))
        node = helper.make_node(
            op_type, inputs, ["y"], name="layer_norm",
            axis=axis, epsilon=float(epsilon), stash_type=1)
        graph = helper.make_graph(
            [node],
            "quantized_ort_layer_norm",
            graph_inputs,
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
            [])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 14),
                helper.make_opsetid("com.microsoft", 1),
            ])
        _ORT_LAYER_NORM_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_LAYER_NORM_SESSIONS[key]


def _grad_for_needed_inputs(y, candidates, grad_output):
    targets = []
    target_indices = []
    for index, (value, needed) in enumerate(candidates):
        if needed:
            targets.append(value)
            target_indices.append(index)
    grads = [None] * len(candidates)
    if not targets:
        return grads
    computed = torch.autograd.grad(
        y, tuple(targets), grad_output, allow_unused=True)
    for index, grad in zip(target_indices, computed):
        grads[index] = grad
    return grads


class _ORTLayerNormOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, axis, epsilon, simplified, has_bias):
        ctx.axis = axis
        ctx.epsilon = epsilon
        ctx.simplified = simplified
        ctx.has_bias = has_bias
        ctx.save_for_backward(x, weight, bias)
        feed = {
            "x": x.detach().cpu().numpy(),
            "weight": weight.detach().cpu().numpy(),
        }
        if has_bias:
            feed["bias"] = bias.detach().cpu().numpy()
        y = _ort_layer_norm_session(
            axis, epsilon, has_bias, simplified).run(None, feed)[0]
        return torch.from_numpy(y).to(device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        with torch.enable_grad():
            x_var = x.detach().requires_grad_(ctx.needs_input_grad[0])
            weight_var = weight.detach().requires_grad_(ctx.needs_input_grad[1])
            bias_var = (
                bias.detach().requires_grad_(ctx.needs_input_grad[2])
                if ctx.has_bias else None)
            y = torch_layer_norm(
                x_var, weight_var, bias_var,
                ctx.axis, ctx.epsilon, ctx.simplified)
            grad_x, grad_weight, grad_bias = _grad_for_needed_inputs(
                y,
                [
                    (x_var, ctx.needs_input_grad[0]),
                    (weight_var, ctx.needs_input_grad[1]),
                    (bias_var, ctx.has_bias and ctx.needs_input_grad[2]),
                ],
                grad_output)
        return grad_x, grad_weight, grad_bias, None, None, None, None


def ort_layer_norm(x, weight, bias=None, axis=-1, epsilon=1e-5,
                   simplified=False):
    if torch.jit.is_tracing() or x.dtype != torch.float32:
        return torch_layer_norm(x, weight, bias, axis, epsilon, simplified)
    if bias is None:
        bias = torch.empty(0, dtype=x.dtype, device=x.device)
        has_bias = False
    else:
        has_bias = True
    return _ORTLayerNormOp.apply(
        x, weight, bias, axis, epsilon, simplified, has_bias)


def qlinear_matmul_selected_mode(module):
    if module.mode == "auto":
        return "torch" if module.force_torch_in_pgd else "exact_ste"
    return module.mode


def qlinear_matmul_torch_forward(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero):
    dtype = (
        a.dtype if torch.is_floating_point(a)
        else torch.get_default_dtype())
    af = round_ste(as_float(a).to(dtype=torch.float32))
    bf = round_ste(as_float(b).to(device=a.device, dtype=torch.float32))
    af = af - as_param_float(a_zero, a)
    bf = bf - as_param_float(b_zero, a)
    y_real = torch.matmul(af, bf)
    y_real = y_real * (
        as_param_float(a_scale, a) * as_param_float(b_scale, a))
    qmin, qmax = quant_bound_tensors(y_zero, a)
    y = torch_quantize_ste(y_real, y_scale, y_zero, qmin, qmax)
    return y.to(dtype=dtype)


def qlinear_matmul_record_mismatch(exact, approx):
    with torch.no_grad():
        diff = (exact.float() - approx.float()).abs()
        nonzero = diff != 0
        return {
            "shape": tuple(exact.shape),
            "max": float(diff.max().detach().cpu()),
            "mean": float(diff.mean().detach().cpu()),
            "frac": float(nonzero.float().mean().detach().cpu()),
            "num_mismatch": int(nonzero.sum().detach().cpu()),
            "numel": diff.numel(),
        }


def qlinear_matmul_forward(module, a, a_scale, a_zero, b, b_scale, b_zero,
                           y_scale, y_zero, y_dtype):
    mode = qlinear_matmul_selected_mode(module)
    if mode == "torch":
        return qlinear_matmul_torch_forward(
            a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero)

    qmin, qmax = quant_bounds(np.dtype(y_dtype))
    dtype = as_float(a).dtype
    qmin = torch.as_tensor(qmin, device=a.device, dtype=dtype)
    qmax = torch.as_tensor(qmax, device=a.device, dtype=dtype)
    exact = ort_qlinear_matmul_forward(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero,
        qmin, qmax)
    needs_grad = (
        (a.requires_grad or b.requires_grad) and torch.is_grad_enabled())
    if mode == "ort" or (not needs_grad and mode != "measure"):
        return exact

    approx = qlinear_matmul_torch_forward(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero)
    if mode == "measure":
        module.last_mismatch = qlinear_matmul_record_mismatch(exact, approx)
    return exact.detach() + approx - approx.detach()


def qlinear_softmax_forward(x, x_scale, x_zero, y_scale, y_zero,
                            y_dtype, axis):
    qmin, qmax = quant_bounds(np.dtype(y_dtype))
    dtype = as_float(x).dtype
    qmin = torch.as_tensor(qmin, device=x.device, dtype=dtype)
    qmax = torch.as_tensor(qmax, device=x.device, dtype=dtype)
    exact = ort_qlinear_softmax_forward(
        x, x_scale, x_zero, y_scale, y_zero, qmin, qmax, int(axis))
    if not x.requires_grad or not torch.is_grad_enabled():
        return exact
    dtype = torch.get_default_dtype()
    x_float = (
        x.to(dtype=dtype) - x_zero.to(device=x.device, dtype=dtype)
    ) * x_scale.to(device=x.device, dtype=dtype)
    y = F.softmax(x_float, dim=int(axis))
    approx = ort_quantize(y, y_scale, y_zero, y_dtype)
    return exact.detach() + approx - approx.detach()


def qlinear_add_exact_torch(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero,
        qmin, qmax):
    dtype = a.dtype if torch.is_floating_point(a) else torch.get_default_dtype()
    af = torch.round(as_float(a).to(dtype=torch.float32))
    bf = torch.round(as_float(b).to(device=a.device, dtype=torch.float32))
    y_scale = as_param_float(y_scale, a)
    y = (
        (af - as_param_float(a_zero, a))
        * (as_param_float(a_scale, a) / y_scale)
        + (bf - as_param_float(b_zero, a))
        * (as_param_float(b_scale, a) / y_scale)
        + as_param_float(y_zero, a)
    )
    y = torch.round(y)
    y = torch.clamp(
        y,
        min=qmin.to(device=a.device, dtype=torch.float32),
        max=qmax.to(device=a.device, dtype=torch.float32))
    return y.to(dtype=dtype)


def qlinear_add_surrogate_torch(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero,
        qmin, qmax):
    dtype = a.dtype if torch.is_floating_point(a) else torch.get_default_dtype()
    af = as_float(a).to(dtype=torch.float32)
    bf = as_float(b).to(device=a.device, dtype=torch.float32)
    y_scale = as_param_float(y_scale, a)
    y = (
        (af - as_param_float(a_zero, a))
        * (as_param_float(a_scale, a) / y_scale)
        + (bf - as_param_float(b_zero, a))
        * (as_param_float(b_scale, a) / y_scale)
        + as_param_float(y_zero, a)
    )
    y = torch.clamp(
        y,
        min=qmin.to(device=a.device, dtype=torch.float32),
        max=qmax.to(device=a.device, dtype=torch.float32))
    y = y + (torch.round(y) - y).detach()
    return y.to(dtype=dtype)


def qlinear_add_forward(a, a_scale, a_zero, b, b_scale, b_zero,
                        y_scale, y_zero):
    qmin, qmax = quant_bound_tensors(y_zero, a)
    exact = qlinear_add_exact_torch(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero,
        qmin, qmax)
    if not (a.requires_grad or b.requires_grad):
        return exact
    approx = qlinear_add_surrogate_torch(
        a, a_scale, a_zero, b, b_scale, b_zero, y_scale, y_zero,
        qmin, qmax)
    return exact.detach() + approx - approx.detach()


def torch_quantize_ste(y_real, y_scale, y_zero, qmin, qmax):
    y_scale = as_param_float(y_scale, y_real)
    y_zero = as_param_float(y_zero, y_real)
    qmin = as_param_float(qmin, y_real)
    qmax = as_param_float(qmax, y_real)
    q = y_real.to(dtype=torch.float32) / y_scale + y_zero
    q = torch.minimum(torch.maximum(q, qmin), qmax)
    return q + (torch.round(q) - q).detach()


def torch_quantize_linear_ste(x, scale, zero_point, qmin, qmax):
    exact = quantize_linear_torch_forward(x, scale, zero_point, qmin, qmax)
    if not x.requires_grad or not torch.is_grad_enabled():
        return exact
    dtype = x.dtype if torch.is_floating_point(x) else torch.get_default_dtype()
    scale = scale.to(device=x.device, dtype=dtype)
    zero_point = zero_point.to(device=x.device, dtype=dtype)
    surrogate = x.to(dtype=dtype) / scale + zero_point
    return exact.detach() + surrogate - surrogate.detach()


def quantize_linear_forward(x, scale, zero_point):
    x = as_float(x)
    scale = as_float(scale).to(device=x.device)
    zero_point = zero_point.to(device=x.device)
    qmin, qmax = quant_bound_tensors(zero_point, x)
    return torch_quantize_linear_ste(x, scale, zero_point, qmin, qmax)


def qgemm_forward(module, *inputs):
    if len(inputs) >= 8:
        return _qgemm_forward_quantized(module, *inputs)
    return qgemm_forward_float(module, *inputs)


def _qgemm_forward_quantized(
        module, a, a_scale, a_zero, b, b_scale, b_zero, *tail):
    if module.has_bias:
        bias, y_scale, y_zero = tail
    else:
        bias = empty_bias(a)
        y_scale, y_zero = tail
    qmin, qmax = quant_bound_tensors(y_zero, a)
    exact = ort_qgemm_forward(
        a, a_scale, a_zero, b, b_scale, b_zero, bias,
        y_scale, y_zero, qmin, qmax,
        module.alpha, module.trans_a, module.trans_b)
    if not a.requires_grad:
        return exact
    approx = qgemm_forward_float(
        module, a, a_scale, a_zero, b, b_scale, b_zero, bias)
    approx = ort_quantize(approx, y_scale, y_zero, quant_dtype(y_zero))
    return exact.detach() + approx - approx.detach()


def qgemm_forward_float(module, a, a_scale, a_zero, b, b_scale, b_zero,
                        bias=None):
    a_float = as_float(a) - as_float(a_zero).to(device=a.device)
    b_float = as_float(b) - reshape_quant_param(
        as_float(b_zero).to(device=b.device), b, axis=0)
    if module.trans_a:
        a_float = a_float.transpose(-1, -2)
    if module.trans_b:
        b_float = b_float.transpose(-1, -2)
    y = module.alpha * torch.matmul(a_float, b_float)
    if bias is not None and bias.numel() > 0:
        y = y + as_float(bias).to(device=y.device)
    return y * (
        as_float(a_scale).to(device=y.device)
        * as_float(b_scale).to(device=y.device))


def qlinearconv_attrs(module, w):
    rank = w.ndim - 2
    strides = module.strides or tuple([1] * rank)
    pads = module.pads or tuple([0] * (2 * rank))
    dilations = module.dilations or tuple([1] * rank)
    return strides, pads, dilations


def qlinearconv_selected_grad_mode(module):
    if module.grad_mode == "auto":
        return "exact_ste" if module.force_exact_in_pgd else "surrogate"
    return module.grad_mode


def qlinearconv_record_mismatch(exact, approx):
    with torch.no_grad():
        diff = (exact.float() - approx.float()).abs()
        nonzero = diff != 0
        return {
            "shape": tuple(exact.shape),
            "max": float(diff.max().detach().cpu()),
            "mean": float(diff.mean().detach().cpu()),
            "frac": float(nonzero.float().mean().detach().cpu()),
            "num_mismatch": int(nonzero.sum().detach().cpu()),
            "numel": diff.numel(),
        }


def qlinearconv_forward(
        module, x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero,
        bias=None):
    bias = bias if module.has_bias and bias is not None else empty_bias(x)
    strides, pads, dilations = qlinearconv_attrs(module, w)
    qmin, qmax = quant_bound_tensors(y_zero, x)
    mode = qlinearconv_selected_grad_mode(module)
    needs_grad = x.requires_grad and torch.is_grad_enabled()
    if not needs_grad and mode != "measure":
        return ort_qlinear_conv_forward(
            x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, bias,
            qmin, qmax, strides, pads, dilations,
            module.group, module.channels_last)
    if mode == "ort":
        return ort_qlinear_conv_forward(
            x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, bias,
            qmin, qmax, strides, pads, dilations,
            module.group, module.channels_last)
    approx = qlinearconv_surrogate(
        module, x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, bias,
        qmin, qmax, strides, pads, dilations)
    if mode == "surrogate":
        return approx
    if mode in ("exact_ste", "measure"):
        exact = ort_qlinear_conv_forward(
            x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero, bias,
            qmin, qmax, strides, pads, dilations,
            module.group, module.channels_last)
        if mode == "measure":
            module.last_mismatch = qlinearconv_record_mismatch(exact, approx)
        return exact.detach() + approx - approx.detach()
    raise RuntimeError(f"Unsupported QLinearConv grad mode: {mode}")


def qlinearconv_surrogate(
        module, x, x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero,
        bias, qmin, qmax, strides, pads, dilations):
    rank = w.ndim
    x_float = as_float(x).to(dtype=torch.float32)
    if module.channels_last:
        x_float = x_float.permute(
            0, rank - 1, *range(1, rank - 1)).contiguous()
    conv_bias = (
        as_float(bias).to(device=x.device, dtype=torch.float32).reshape(-1)
        if bias.numel() > 0 else None)
    x_float = x_float - as_float(x_zero).to(
        device=x.device, dtype=torch.float32)
    w_float = as_float(w).to(device=x.device, dtype=torch.float32)
    w_float = w_float - reshape_quant_param(
        as_float(w_zero).to(device=x.device, dtype=torch.float32),
        w_float, axis=0)
    x_float = pad_conv_input(x_float, pads)
    if rank == 3:
        acc = F.conv1d(
            x_float, w_float, bias=conv_bias, stride=strides,
            dilation=dilations, groups=module.group)
    elif rank == 4:
        acc = F.conv2d(
            x_float, w_float, bias=conv_bias, stride=strides,
            dilation=dilations, groups=module.group)
    elif rank == 5:
        acc = F.conv3d(
            x_float, w_float, bias=conv_bias, stride=strides,
            dilation=dilations, groups=module.group)
    else:
        raise RuntimeError(f"Unsupported QLinearConv weight rank: {rank}.")
    scale = (
        as_float(x_scale).to(device=acc.device, dtype=torch.float32)
        * reshape_channel_param(
            as_float(w_scale).to(device=acc.device, dtype=torch.float32),
            acc))
    y = torch_quantize_ste(acc * scale, y_scale, y_zero, qmin, qmax)
    if module.channels_last:
        y = y.permute(0, *range(2, y.ndim), 1).contiguous()
    return y.to(dtype=as_float(x).dtype)


def _ort_quantize_session(zero_dtype):
    zero_dtype = np.dtype(zero_dtype)
    if zero_dtype not in _ORT_QUANTIZE_SESSIONS:
        import onnxruntime as ort

        graph = helper.make_graph(
            [helper.make_node(
                "QuantizeLinear", ["x", "scale", "zero_point"], ["y"],
                name="quantize")],
            "ort_quantize",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info(
                    "zero_point", onnx_tensor_type(zero_dtype), None),
            ],
            [helper.make_tensor_value_info("y", onnx_tensor_type(zero_dtype), None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_QUANTIZE_SESSIONS[zero_dtype] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_QUANTIZE_SESSIONS[zero_dtype]


def _ort_qlinear_matmul_session(a_dtype, b_dtype, y_dtype):
    key = (np.dtype(a_dtype), np.dtype(b_dtype), np.dtype(y_dtype))
    if key not in _ORT_QLINEAR_MATMUL_SESSIONS:
        import onnxruntime as ort

        graph = helper.make_graph(
            [helper.make_node(
                "QLinearMatMul",
                ["a", "a_scale", "a_zero", "b", "b_scale", "b_zero",
                 "y_scale", "y_zero"],
                ["y"],
                name="qlinear_matmul")],
            "ort_qlinear_matmul",
            [
                helper.make_tensor_value_info("a", onnx_tensor_type(a_dtype), None),
                helper.make_tensor_value_info("a_scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("a_zero", onnx_tensor_type(a_dtype), None),
                helper.make_tensor_value_info("b", onnx_tensor_type(b_dtype), None),
                helper.make_tensor_value_info("b_scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("b_zero", onnx_tensor_type(b_dtype), None),
                helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y_zero", onnx_tensor_type(y_dtype), None),
            ],
            [helper.make_tensor_value_info("y", onnx_tensor_type(y_dtype), None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        _ORT_QLINEAR_MATMUL_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_QLINEAR_MATMUL_SESSIONS[key]


def _ort_qgemm_session(a_dtype, b_dtype, y_dtype, bias_dtype,
                       alpha, trans_a, trans_b):
    key = (
        np.dtype(a_dtype), np.dtype(b_dtype), np.dtype(y_dtype),
        None if bias_dtype is None else np.dtype(bias_dtype),
        float(alpha), int(trans_a), int(trans_b))
    if key not in _ORT_QGEMM_SESSIONS:
        import onnxruntime as ort

        inputs = ["a", "a_scale", "a_zero", "b", "b_scale", "b_zero"]
        graph_inputs = [
            helper.make_tensor_value_info("a", onnx_tensor_type(a_dtype), None),
            helper.make_tensor_value_info("a_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("a_zero", onnx_tensor_type(a_dtype), None),
            helper.make_tensor_value_info("b", onnx_tensor_type(b_dtype), None),
            helper.make_tensor_value_info("b_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("b_zero", onnx_tensor_type(b_dtype), None),
        ]
        if bias_dtype is not None:
            inputs.append("bias")
            graph_inputs.append(helper.make_tensor_value_info(
                "bias", onnx_tensor_type(bias_dtype), None))
        else:
            inputs.append("")
        inputs.extend(["y_scale", "y_zero"])
        graph_inputs.extend([
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("y_zero", onnx_tensor_type(y_dtype), None),
        ])
        graph = helper.make_graph(
            [helper.make_node(
                "QGemm", inputs, ["y"], name="qgemm", domain="com.microsoft",
                alpha=float(alpha), transA=int(trans_a), transB=int(trans_b))],
            "ort_qgemm",
            graph_inputs,
            [helper.make_tensor_value_info("y", onnx_tensor_type(y_dtype), None)],
            [])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 14),
                helper.make_opsetid("com.microsoft", 1),
            ])
        _ORT_QGEMM_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_QGEMM_SESSIONS[key]


def _ort_qlinear_conv_session(x_dtype, w_dtype, y_dtype, bias_dtype, rank,
                              strides, pads, dilations, group, channels_last):
    if channels_last:
        raise RuntimeError(
            "channels_last QLinearConv inputs must be converted to "
            "channels-first before constructing the ORT helper session.")
    key = (
        np.dtype(x_dtype), np.dtype(w_dtype), np.dtype(y_dtype),
        None if bias_dtype is None else np.dtype(bias_dtype),
        int(rank), tuple(strides), tuple(pads), tuple(dilations), int(group))
    if key not in _ORT_QLINEAR_CONV_SESSIONS:
        import onnxruntime as ort

        inputs = [
            "x", "x_scale", "x_zero", "w", "w_scale", "w_zero",
            "y_scale", "y_zero"]
        graph_inputs = [
            helper.make_tensor_value_info("x", onnx_tensor_type(x_dtype), None),
            helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("x_zero", onnx_tensor_type(x_dtype), None),
            helper.make_tensor_value_info("w", onnx_tensor_type(w_dtype), None),
            helper.make_tensor_value_info("w_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("w_zero", onnx_tensor_type(w_dtype), None),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("y_zero", onnx_tensor_type(y_dtype), None),
        ]
        if bias_dtype is not None:
            inputs.append("bias")
            graph_inputs.append(helper.make_tensor_value_info(
                "bias", onnx_tensor_type(bias_dtype), None))
        attrs = {
            "strides": list(strides),
            "pads": list(pads),
            "dilations": list(dilations),
            "group": int(group),
        }
        graph = helper.make_graph(
            [helper.make_node(
                "QLinearConv", inputs, ["y"], name="qlinear_conv", **attrs)],
            "ort_qlinear_conv",
            graph_inputs,
            [helper.make_tensor_value_info("y", onnx_tensor_type(y_dtype), None)],
            [])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 14)])
        _ORT_QLINEAR_CONV_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_QLINEAR_CONV_SESSIONS[key]


def _ort_qlinear_softmax_session(x_dtype, y_dtype, axis):
    key = (np.dtype(x_dtype), np.dtype(y_dtype), int(axis))
    if key not in _ORT_QLINEAR_SOFTMAX_SESSIONS:
        import onnxruntime as ort

        graph = helper.make_graph(
            [helper.make_node(
                "QLinearSoftmax",
                ["x", "x_scale", "x_zero", "y_scale", "y_zero"],
                ["y"],
                name="qlinear_softmax",
                domain="com.microsoft",
                opset=13,
                axis=int(axis))],
            "ort_qlinear_softmax",
            [
                helper.make_tensor_value_info("x", onnx_tensor_type(x_dtype), None),
                helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("x_zero", onnx_tensor_type(x_dtype), None),
                helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, None),
                helper.make_tensor_value_info("y_zero", onnx_tensor_type(y_dtype), None),
            ],
            [helper.make_tensor_value_info("y", onnx_tensor_type(y_dtype), None)],
            [])
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("com.microsoft", 1),
            ])
        _ORT_QLINEAR_SOFTMAX_SESSIONS[key] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
    return _ORT_QLINEAR_SOFTMAX_SESSIONS[key]


def _dtype_from_value_or_zero_point(
        value, zero_point, qmin=None, qmax=None,
        prefer_signed_float_weight=False):
    zero_dtype = None if zero_point is None else np.dtype(_to_numpy(zero_point).dtype)
    if zero_dtype in _QUANT_DTYPES:
        return zero_dtype
    value_dtype = np.dtype(_to_numpy(value).dtype)
    if value_dtype in _QUANT_DTYPES:
        return value_dtype
    if qmin is not None and qmax is not None:
        return dtype_from_quant_bounds(qmin, qmax)
    value_array = _to_numpy(value)
    if value_array.size and np.any(value_array < 0):
        return np.dtype(np.int8)
    # Weight constants may be float-coded u8s8 data. Only weight-like call sites
    # pass this hint; typed values/zero-points and visible negatives still win.
    if prefer_signed_float_weight and _is_float_coded_zero_point(zero_point):
        return np.dtype(np.int8)
    return dtype_from_zero_point(zero_point, qmin, qmax)


def _bias_dtype(value):
    if _is_empty_tensor(value):
        return None
    return np.dtype(np.int32)


def _as_bias_numpy(value, dtype):
    dtype = np.dtype(dtype)
    array = _to_numpy(value)
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(np.rint(array).astype(dtype), dtype=dtype)
    return np.asarray(array.astype(dtype, copy=False), dtype=dtype)


def _is_float_coded_zero_point(zero_point):
    if zero_point is None:
        return False
    array = _to_numpy(zero_point)
    return (
        np.issubdtype(array.dtype, np.floating)
        and array.size > 0
        and bool(np.all(array == 0.0)))


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_float_numpy(value):
    return np.asarray(
        _to_numpy(value).astype(np.float32, copy=False), dtype=np.float32)


def _float_tensor_like(array, primary):
    return torch.from_numpy(array.astype(np.float32)).to(device=_device(primary))


def _device(value):
    if isinstance(value, torch.Tensor):
        return value.device
    return torch.device("cpu")


def _first_item(value):
    return _to_numpy(value).reshape(-1)[0].item()


def _as_float_scalar(value):
    return float(_first_item(value))


def _as_int(value):
    return int(_first_item(value))


def _as_int_tuple(value):
    return tuple(int(v) for v in _to_numpy(value).reshape(-1).tolist())


def _rank(value):
    if isinstance(value, torch.Tensor):
        return value.ndim
    return np.asarray(value).ndim


def _to_channels_first(value, weight_rank):
    perm = (0, int(weight_rank) - 1, *range(1, int(weight_rank) - 1))
    if isinstance(value, torch.Tensor):
        return value.permute(*perm)
    return np.transpose(_to_numpy(value), perm)


def _to_channels_last(value):
    if isinstance(value, torch.Tensor):
        return value.permute(0, *range(2, value.ndim), 1)
    array = _to_numpy(value)
    return np.transpose(array, (0, *range(2, array.ndim), 1))


def _is_empty_tensor(value):
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    return np.asarray(value).size == 0
