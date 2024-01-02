from beartype import beartype
import torch

@beartype
def get_adv_string(inputs: torch.Tensor, outputs: torch.Tensor, is_nhwc: bool = False) -> str:
    if is_nhwc:
        assert inputs.ndim == 4
        inputs = inputs.permute(0, 2, 3, 1)
    x = inputs.flatten().detach().cpu().numpy()
    y = outputs.flatten().detach().cpu().numpy()
    string_x = [f'(X_{i} {x[i]})' for i in range(len(x))]
    string_y = [f'(Y_{i} {y[i]})' for i in range(len(y))]
    string = '\n '.join(string_x + string_y)
    return f"({string})"