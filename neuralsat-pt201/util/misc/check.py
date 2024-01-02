from beartype import beartype
import torch

from onnx2pytorch.convert.model import ConvertModel

@beartype
@torch.no_grad()
def check_solution(net: ConvertModel, adv: torch.Tensor, 
                   cs: torch.Tensor, rhs: torch.Tensor, 
                   data_min: torch.Tensor, data_max: torch.Tensor) -> torch.Tensor:
    old_dtype = adv.dtype
    adv = adv.to(data_min.dtype)
    adv = torch.clamp(torch.clamp(adv, max=data_max), min=data_min)
    assert torch.all(data_min <= adv) and torch.all(adv <= data_max)
    net.to(data_min.dtype)
    output = net(adv).detach()
    cond = torch.matmul(cs, output.unsqueeze(-1)).squeeze(-1) - rhs
    net.to(old_dtype)
    # print(cond.detach().cpu().numpy())
    return (cond.amax(dim=-1, keepdim=True) < 0.0).any(dim=-1).any(dim=-1)