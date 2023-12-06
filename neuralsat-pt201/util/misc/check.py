import torch

@torch.no_grad()
def check_solution(net, adv, cs, rhs, data_min, data_max):
    if torch.all(data_min <= adv) and torch.all(adv <= data_max):
        output = net(adv).detach()
        cond = torch.matmul(cs, output.unsqueeze(-1)).squeeze(-1) - rhs
        return (cond.amax(dim=-1, keepdim=True) < 0.0).any(dim=-1).any(dim=-1)
    return False