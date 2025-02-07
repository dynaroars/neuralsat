from beartype import beartype
import torch

from onnx2pytorch.convert.model import ConvertModel

@beartype
@torch.no_grad()
def check_solution(net: ConvertModel | torch.nn.Module, adv: torch.Tensor, 
                   cs: torch.Tensor, rhs: torch.Tensor, 
                   data_min: torch.Tensor, data_max: torch.Tensor) -> torch.Tensor:
    old_dtype = adv.dtype
    new_adv = adv.clone().to(data_min.dtype)
    # new_adv = torch.clamp(torch.clamp(new_adv, max=data_max), min=data_min)
    assert torch.all(data_min <= new_adv) and torch.all(new_adv <= data_max)
    net.to(data_min.dtype)
    output = net(new_adv).detach().flatten(1)
    cond = torch.matmul(cs, output.unsqueeze(-1)).squeeze(-1) - rhs
    net.to(old_dtype)
    valid = (cond.amax(dim=-1, keepdim=True) < 0.0).any(dim=-1).any(dim=-1)
    return valid