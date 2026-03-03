from torch import nn
import torch
import onnx

from ..operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
)
from .attribute import extract_attributes, extract_attr_values


class LSTMUnrolledImpl(nn.Module):
    """Unrolled single-layer LSTM (batch_first=False). Returns ONNX-compatible (Y, Y_h, Y_c)."""

    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        if bidirectional:
            self.cell_reverse = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, h_0=None, c_0=None):
        seq_len, batch, input_size = x.shape
        h_fwd, c_fwd = h_0[0], c_0[0]
        fwd_outputs = []
        for i in range(seq_len):
            h_fwd, c_fwd = self.cell(x[i], (h_fwd, c_fwd))
            fwd_outputs.append(h_fwd)
        fwd_out = torch.stack(fwd_outputs, dim=1).transpose(0, 1)  # [seq_len, batch, hidden]

        if self.bidirectional:
            h_bwd, c_bwd = h_0[1], c_0[1]
            bwd_outputs = []
            for i in reversed(range(seq_len)):
                h_bwd, c_bwd = self.cell_reverse(x[i], (h_bwd, c_bwd))
                bwd_outputs.append(h_bwd)
            bwd_out = torch.stack(list(reversed(bwd_outputs)), dim=1).transpose(0, 1)
            Y = torch.stack([fwd_out, bwd_out], dim=1)   # [seq_len, 2, batch, hidden]
            Y_h = torch.stack([h_fwd, h_bwd], dim=0)     # [2, batch, hidden]
            Y_c = torch.stack([c_fwd, c_bwd], dim=0)     # [2, batch, hidden]
        else:
            Y = fwd_out.unsqueeze(1)   # [seq_len, 1, batch, hidden]
            Y_h = h_fwd.unsqueeze(0)   # [1, batch, hidden]
            Y_c = c_fwd.unsqueeze(0)   # [1, batch, hidden]

        return Y, Y_h, Y_c


def extract_params(params):
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}".format(param_length))
    return weight, bias


def load_params(layer, weight, bias):
    """Load weight and bias to a given layer from onnx format."""
    layer.weight.data = torch.from_numpy(onnx.numpy_helper.to_array(weight).copy())
    if bias is not None:
        layer.bias.data = torch.from_numpy(onnx.numpy_helper.to_array(bias).copy())


def convert_layer(node, layer_type, params=None):
    """Use to convert Conv, MaxPool, AvgPool layers."""
    assert layer_type in [
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AvgPool",
    ], "Incorrect layer type: {}".format(layer_type)
    kwargs = extract_attributes(node)
    kernel_size_length = len(kwargs["kernel_size"])
    try:
        layer = getattr(nn, "{}{}d".format(layer_type, kernel_size_length))
    except AttributeError:
        raise ValueError(
            "Unexpected length of kernel_size dimension: {}".format(kernel_size_length)
        )

    pad_layer = None
    if params:
        weight, bias = extract_params(params)
        kwargs["bias"] = bias is not None
        kwargs["in_channels"] = weight.dims[1] * kwargs.get("groups", 1)
        kwargs["out_channels"] = weight.dims[0]

        if layer_type == "ConvTranspose":
            kwargs["in_channels"], kwargs["out_channels"] = (
                kwargs["out_channels"],
                kwargs["in_channels"],
            )

        # if padding is a layer, remove from kwargs and prepend later
        if "padding" in kwargs and isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")

        # initialize layer and load weights
        layer = layer(**kwargs)
        load_params(layer, weight, bias)
    else:
        # initialize operations without parameters (MaxPool, AvgPool, etc.)
        # if padding is a layer, remove from kwargs and prepend later
        if "padding" in kwargs and isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")
        layer = layer(**kwargs)

    if pad_layer is not None:
        layer = nn.Sequential(pad_layer, layer)

    return layer


def convert_batch_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skip input dimension check, not possible before forward pass
    layer = BatchNormWrapper
    torch_params = [torch.from_numpy(onnx.numpy_helper.to_array(param).copy()) for param in params]

    # Initialize layer and load weights
    layer = layer(torch_params, **kwargs)
    return layer


def convert_instance_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skip input dimension check, not possible before forward pass
    layer = InstanceNormWrapper
    torch_params = [torch.from_numpy(onnx.numpy_helper.to_array(param).copy()) for param in params]

    # Initialize layer and load weights
    layer = layer(torch_params, **kwargs)
    return layer


def convert_linear_layer(node, params):
    """Convert linear layer from onnx node and params."""
    # Default Gemm attributes
    dc = dict(
        transpose_weight=True,
        transpose_activation=False,
        weight_multiplier=1,
        bias_multiplier=1,
    )
    dc.update(extract_attributes(node))
    for attr in node.attribute:
        if attr.name in ["transA"] and extract_attr_values(attr) != 0:
            raise NotImplementedError(
                "Not implemented for attr.name={} and value!=0.".format(attr.name)
            )

    kwargs = {}
    weight, bias = extract_params(params)
    kwargs["bias"] = bias is not None
    kwargs["in_features"] = weight.dims[1]
    kwargs["out_features"] = weight.dims[0]

    # initialize layer and load weights
    layer = nn.Linear(**kwargs)
    load_params(layer, weight, bias)

    # apply onnx gemm attributes
    if dc.get("transpose_weight"):
        layer.weight.data = layer.weight.data.t()

    layer.weight.data *= dc.get("weight_multiplier")
    if layer.bias is not None:
        layer.bias.data *= dc.get("bias_multiplier")

    return layer


def extract_and_load_params_lstm(node, weights):
    X = None
    W = None
    R = None
    B = None
    sequence_lens = None
    initial_h = None
    initial_c = None
    P = None

    for par_ix, par_name in enumerate(node.input):
        if par_ix == 0:
            if par_name in weights:
                X = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 1:
            if par_name in weights:
                W = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 2:
            if par_name in weights:
                R = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 3:
            if par_name != "" and par_name in weights:
                B = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 4:
            if par_name != "" and par_name in weights:
                sequence_lens = torch.from_numpy(
                    onnx.numpy_helper.to_array(weights[par_name]).copy()
                )
        elif par_ix == 5:
            if par_name != "" and par_name in weights:
                initial_h = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 6:
            if par_name != "" and par_name in weights:
                initial_c = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        elif par_ix == 7:
            if par_name != "" and par_name in weights:
                P = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
    return (X, W, R, B, sequence_lens, initial_h, initial_c, P)


class GRUUnrolledImpl(nn.Module):
    """Unrolled single-layer GRU (batch_first=False). Returns ONNX-compatible (Y, Y_h)."""

    def __init__(self, input_size, hidden_size, bidirectional=False, linear_before_reset=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.linear_before_reset = linear_before_reset
        self.cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        if bidirectional:
            self.cell_reverse = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, h_0=None):
        seq_len, batch, input_size = x.shape

        h_fwd = h_0[0]
        fwd_outputs = []
        for i in range(seq_len):
            h_fwd = self.cell(x[i], h_fwd)
            fwd_outputs.append(h_fwd)
        fwd_out = torch.stack(fwd_outputs, dim=1).transpose(0, 1)  # [seq_len, batch, hidden]

        if self.bidirectional:
            h_bwd = h_0[1]
            bwd_outputs = []
            for i in reversed(range(seq_len)):
                h_bwd = self.cell_reverse(x[i], h_bwd)
                bwd_outputs.append(h_bwd)
            bwd_out = torch.stack(list(reversed(bwd_outputs)), dim=1).transpose(0, 1)
            Y = torch.stack([fwd_out, bwd_out], dim=1)  # [seq_len, 2, batch, hidden]
            Y_h = torch.stack([h_fwd, h_bwd], dim=0)    # [2, batch, hidden]
        else:
            Y = fwd_out.unsqueeze(1)   # [seq_len, 1, batch, hidden]
            Y_h = h_fwd.unsqueeze(0)   # [1, batch, hidden]

        return Y, Y_h


def extract_and_load_params_gru(node, weights):
    X = W = R = B = sequence_lens = initial_h = None
    for par_ix, par_name in enumerate(node.input):
        if par_name == "":
            continue
        if par_name not in weights:
            continue
        val = torch.from_numpy(onnx.numpy_helper.to_array(weights[par_name]).copy())
        if par_ix == 0:
            X = val
        elif par_ix == 1:
            W = val
        elif par_ix == 2:
            R = val
        elif par_ix == 3:
            B = val
        elif par_ix == 4:
            sequence_lens = val
        elif par_ix == 5:
            initial_h = val
    return X, W, R, B, sequence_lens, initial_h


def convert_gru_layer(node, weights):
    """Convert GRU layer from onnx node and params."""
    X, W, R, B, sequence_lens, initial_h = extract_and_load_params_gru(node, weights)
    if initial_h is not None:
        raise NotImplementedError("GRU initial_h not yet implemented.")

    dc = dict(
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction="forward",
        hidden_size=None,
        linear_before_reset=0,
        layout=0,
    )
    dc.update(extract_attributes(node))
    if dc["activation_alpha"] is not None:
        raise NotImplementedError("GRU activation_alpha {}.".format(dc["activation_alpha"]))
    if dc["activation_beta"] is not None:
        raise NotImplementedError("GRU activation_beta {}.".format(dc["activation_beta"]))
    if dc["activations"] is not None:
        raise NotImplementedError("GRU activations {}.".format(dc["activations"]))
    if dc["clip"] is not None:
        raise NotImplementedError("GRU clip {}.".format(dc["clip"]))
    if dc["direction"] not in ("forward", "bidirectional"):
        raise ValueError("GRU direction {}.".format(dc["direction"]))
    if dc["hidden_size"] is None:
        raise ValueError("GRU hidden_size is None.")
    if dc["layout"] != 0:
        raise NotImplementedError("GRU not implemented for layout={}.".format(dc["layout"]))

    input_size = W.shape[2]
    hidden_size = dc["hidden_size"]
    bidirectional = dc["direction"] == "bidirectional"
    num_directions = 2 if bidirectional else 1

    layer = GRUUnrolledImpl(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
    )

    def _reorder_zrh_to_rzn_2d(mat, h):
        """Reorder ONNX (z, r, h) gate order to PyTorch (r, z, n) for 2D weight matrices."""
        return torch.cat((mat[h:2*h, :], mat[0:h, :], mat[2*h:3*h, :]), dim=0)

    def _reorder_zrh_to_rzn_1d(vec, h):
        """Reorder ONNX (z, r, h) gate order to PyTorch (r, z, n) for 1D bias vectors."""
        return torch.cat((vec[h:2*h], vec[0:h], vec[2*h:3*h]), dim=0)

    if bidirectional:
        W_zrh = W.transpose(0, 1).view(3 * hidden_size, num_directions, input_size)
        R_zrh = R.transpose(0, 1).view(3 * hidden_size, num_directions, hidden_size)
        for dir_dim, cell in [(0, layer.cell), (1, layer.cell_reverse)]:
            cell.weight_ih.data = _reorder_zrh_to_rzn_2d(W_zrh[:, dir_dim, :], hidden_size)
            cell.weight_hh.data = _reorder_zrh_to_rzn_2d(R_zrh[:, dir_dim, :], hidden_size)
            cell.bias_ih.data = _reorder_zrh_to_rzn_1d(B[dir_dim, :3 * hidden_size], hidden_size)
            cell.bias_hh.data = _reorder_zrh_to_rzn_1d(B[dir_dim, 3 * hidden_size:], hidden_size)
    else:
        W_zrh = W.transpose(0, 1).view(3 * hidden_size, input_size)
        R_zrh = R.transpose(0, 1).view(3 * hidden_size, hidden_size)
        layer.cell.weight_ih.data = _reorder_zrh_to_rzn_2d(W_zrh, hidden_size)
        layer.cell.weight_hh.data = _reorder_zrh_to_rzn_2d(R_zrh, hidden_size)
        layer.cell.bias_ih.data = _reorder_zrh_to_rzn_1d(B[0, :3 * hidden_size], hidden_size)
        layer.cell.bias_hh.data = _reorder_zrh_to_rzn_1d(B[0, 3 * hidden_size:], hidden_size)

    return layer


def convert_lstm_layer(node, weights):
    """Convert LSTM layer from onnx node and params."""
    params_tuple = extract_and_load_params_lstm(node, weights)
    (X, W, R, B, sequence_lens, initial_h, initial_c, P) = params_tuple
    if initial_h is not None:
        raise NotImplementedError("LSTM initial_h not yet implemented.")
    if initial_c is not None:
        raise NotImplementedError("LSTM initial_c not yet implemented.")
    if P is not None:
        raise NotImplementedError("LSTM P not yet implemented.")

    dc = dict(
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction="forward",
        hidden_size=None,
        input_forget=0,
        layout=0,
    )
    dc.update(extract_attributes(node))
    if dc["activation_alpha"] is not None:
        raise NotImplementedError(
            "LSTM activation_alpha {}.".format(dc["activation_alpha"])
        )
    if dc["activation_beta"] is not None:
        raise NotImplementedError(
            "LSTM activation_beta {}.".format(dc["activation_beta"])
        )
    if dc["activations"] is not None:
        # TODO allow if torch-compatible activations are set explicitly
        raise NotImplementedError("LSTM activations {}.".format(dc["activations"]))
    if dc["clip"] is not None:
        raise NotImplementedError("LSTM clip {}".format(dc["clip"]))
    if dc["direction"] not in ("forward", "bidirectional"):
        raise ValueError("LSTM direction {}.".format(dc["direction"]))
    if dc["hidden_size"] is None:
        raise ValueError("LSTM hidden_size is None.")
    if dc["input_forget"] != 0:
        raise NotImplementedError("LSTM input_forget {}.".format(dc["input_forget"]))
    if dc["layout"] != 0:
        raise NotImplementedError(
            "LSTM not implemented for layout={}".format(dc["layout"])
        )

    input_size = W.shape[2]
    hidden_size = dc["hidden_size"]
    bidirectional = dc["direction"] == "bidirectional"
    num_directions = 2 if bidirectional else 1

    layer = LSTMUnrolledImpl(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)

    def _reorder_iofc_to_ifco_2d(mat, h):
        """Reorder ONNX iofc gate order to PyTorch ifco order for 2D weight matrices."""
        return torch.cat((mat[0:h, :], mat[2*h:4*h, :], mat[h:2*h, :]), dim=0)

    def _reorder_iofc_to_ifco_1d(vec, h):
        """Reorder ONNX iofc gate order to PyTorch ifco order for 1D bias vectors."""
        return torch.cat((vec[0:h], vec[2*h:4*h], vec[h:2*h]), dim=0)

    if bidirectional:
        W_iofc = W.transpose(0, 1).view(4 * hidden_size, num_directions, input_size)
        R_iofc = R.transpose(0, 1).view(4 * hidden_size, num_directions, hidden_size)
        for dir_dim, cell in [(0, layer.cell), (1, layer.cell_reverse)]:
            cell.weight_ih.data = _reorder_iofc_to_ifco_2d(W_iofc[:, dir_dim, :], hidden_size)
            cell.weight_hh.data = _reorder_iofc_to_ifco_2d(R_iofc[:, dir_dim, :], hidden_size)
            cell.bias_ih.data = _reorder_iofc_to_ifco_1d(B[dir_dim, :4 * hidden_size], hidden_size)
            cell.bias_hh.data = _reorder_iofc_to_ifco_1d(B[dir_dim, 4 * hidden_size:], hidden_size)
    else:
        W_iofc = W.transpose(0, 1).view(4 * hidden_size, input_size)
        R_iofc = R.transpose(0, 1).view(4 * hidden_size, hidden_size)
        layer.cell.weight_ih.data = _reorder_iofc_to_ifco_2d(W_iofc, hidden_size)
        layer.cell.weight_hh.data = _reorder_iofc_to_ifco_2d(R_iofc, hidden_size)
        layer.cell.bias_ih.data = _reorder_iofc_to_ifco_1d(B[0, :4 * hidden_size], hidden_size)
        layer.cell.bias_hh.data = _reorder_iofc_to_ifco_1d(B[0, 4 * hidden_size:], hidden_size)

    return layer
