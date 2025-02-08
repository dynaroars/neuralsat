from beartype import beartype
import torch

from util.network.read_onnx import inference_onnx

@beartype
def get_adv_string(inputs: torch.Tensor, net_path: str, is_nhwc: bool = False) -> str:
    if is_nhwc:
        assert inputs.ndim == 4
        inputs = inputs.permute(0, 2, 3, 1)
    x = inputs.detach().cpu().float().numpy()
    y = inference_onnx(net_path, x)[0]
    # flatten
    x = x.flatten()
    y = y.flatten()
    # export
    string_x = [f'(X_{i} {x[i]})' for i in range(len(x))]
    string_y = [f'(Y_{i} {y[i]})' for i in range(len(y))]
    string = '\n '.join(string_x + string_y)
    return f"({string})"