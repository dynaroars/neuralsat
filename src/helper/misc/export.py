from beartype import beartype
import onnxruntime as ort
import numpy as np
import torch

from helper.network.read_onnx import inference_onnx

@beartype
def get_adv_string(inputs: torch.Tensor, net_path: str, gtrsb_nhwc: tuple[int, int, int] | None = None) -> str:
    x = inputs.detach().cpu().float()
    if gtrsb_nhwc is not None:
        from helper.spec.objective import gtrsb_nchw_to_nhwc
        h, w, c = gtrsb_nhwc
        x = gtrsb_nchw_to_nhwc(x, h, w, c)
    x = x.numpy()
    y = inference_onnx(net_path, x)[0]
    # flatten
    x = x.flatten()
    y = y.flatten()
    # export
    string_x = [f'(X_{i} {x[i]})' for i in range(len(x))]
    string_y = [f'(Y_{i} {y[i]})' for i in range(len(y))]
    string = '\n '.join(string_x + string_y)
    return f"({string})"


def validate_cex(inputs: torch.Tensor, net_path: str, vnnlib_path: str, tol: float = 1e-5,
                 gtrsb_nhwc: tuple[int, int, int] | None = None) -> bool:
    """Check that the counterexample satisfies vnnlib input bounds and output property.

    Returns True if the CEX is valid (correctly witnesses SAT), False otherwise.
    """
    from helper.spec.read_vnnlib import read_vnnlib

    x_t = inputs.detach().cpu().float()
    if gtrsb_nhwc is not None:
        from helper.spec.objective import gtrsb_nchw_to_nhwc
        h, w, c = gtrsb_nhwc
        x_t = gtrsb_nchw_to_nhwc(x_t, h, w, c)
    x = x_t.numpy().flatten()

    # Parse vnnlib for input bounds and output spec
    specs = read_vnnlib(vnnlib_path)

    for box, spec_list in specs:
        # 1. Check input bounds
        for i, (lb, ub) in enumerate(box):
            if x[i] < lb - tol or x[i] > ub + tol:
                return False

        # 2. Run model and check output property (disjunction of conjuncts)
        sess = ort.InferenceSession(net_path)
        inp_info = sess.get_inputs()[0]
        x_np = x_t.numpy()
        # Reshape to match the ONNX model's expected rank (some models expect rank-1, others rank-2)
        expected_ndim = len(inp_info.shape)
        if x_np.ndim != expected_ndim:
            x_np = x_np.reshape([s for s in inp_info.shape if s not in (None, -1, 0)] or x_np.flatten().shape)
        y = sess.run(None, {inp_info.name: x_np})[0].flatten()
        sat = False
        for mat, rhs in spec_list:
            mat = np.array(mat)
            rhs = np.array(rhs)
            if np.all(mat @ y <= rhs + tol):
                sat = True
                break
        if not sat:
            return False

    return True