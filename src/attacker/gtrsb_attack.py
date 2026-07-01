from beartype import beartype
import torch
import time

from abstractor.auto_LiRPA import BoundedModule
from abstractor.graph_optimizer import merge_sign
from attacker.pgd_attack.util import serialize_specs, check_adv_multi, get_loss
from helper.misc.adam_clipping import AdamClipping


def _signmerge_layers(net: BoundedModule) -> list:
    return [n.name for n in net.nodes() if '/merge' in n.name]


def _set_signmerge_mode(net: BoundedModule, loose: bool) -> None:
    for name in _signmerge_layers(net):
        node = net[name]
        node.signmergefunction = node.loose_function if loose else node.tight_function


def _pick_single_adv(inputs: torch.Tensor, output: torch.Tensor,
                     serialized_conditions: tuple, data_max: torch.Tensor, data_min: torch.Tensor,
                     input_shape: tuple) -> torch.Tensor:
    """Extract one (batch, *spatial) input from a multi-restart PGD batch."""
    for r in range(inputs.shape[1]):
        inp_r = inputs[:, r:r + 1]
        out_r = output[:, r:r + 1]
        if check_adv_multi(inp_r, out_r, serialized_conditions, data_max, data_min):
            return inputs[0, r].reshape(*input_shape)
    return inputs[0, 0].reshape(*input_shape)


def _gtrsb_signmerge_loss(net: BoundedModule, num_restarts: int, num_specs: int) -> torch.Tensor:
    layers = _signmerge_layers(net)[1:] 
    if not layers:
        return torch.zeros(1, num_restarts, num_specs, device=net.device)
    threshold, scaler = 1e-4, 10.0
    losses = []
    for name in layers:
        inp = net.get_forward_value(net[name].inputs[0])
        elem = torch.clamp(threshold - torch.abs(inp), min=0)
        elem = elem.view(num_restarts * num_specs, -1)
        losses.append(-torch.mean(elem, dim=1))
    stacked = torch.mean(torch.stack(losses), dim=0)
    return (stacked / (scaler * threshold)).view(1, num_restarts, num_specs)


@beartype
def gtrsb_attack(
        model: torch.nn.Module,
        x: torch.Tensor,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        cs: torch.Tensor,
        rhs: torch.Tensor,
        input_shape: tuple,
        device: str,
        timeout: float,
        num_restarts: int = 50,
        attack_iters: int = 100000,
) -> tuple[bool, torch.Tensor | None]:
    net = BoundedModule(
        model=model,
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
        device=device,
    )
    merge_sign(net)
    net.train()
    for p in net.parameters():
        p.requires_grad_(False)

    serialized = serialize_specs(x, cs, rhs)
    loose = len(_signmerge_layers(net)) == 2
    deadline = time.time() + timeout

    for _ in range(40):
        if time.time() > deadline:
            break
        _set_signmerge_mode(net, loose)
        hit, adv = _gtrsb_pgd_loop(
            net, x, data_min, data_max, serialized, input_shape,
            num_restarts=num_restarts, attack_iters=attack_iters,
            deadline=deadline,
        )
        if hit:
            return True, adv
        loose = not loose

    return False, None


def _gtrsb_pgd_loop(net, X, data_min, data_max, serialized_conditions, input_shape,
                    num_restarts, attack_iters, deadline):
    lr_decay = 0.99
    num_specs = len(serialized_conditions[-1][0])
    extra_dim = (num_restarts, num_specs)

    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)
    X = X.view(X.shape[0], 1, 1, *X.shape[1:]).expand(-1, *extra_dim, *(-1,) * (len(input_shape) - 1))
    delta_lower = data_min - X
    delta_upper = data_max - X
    lr = torch.max(data_max - data_min).item() / 8
    delta = (torch.empty_like(X).uniform_() * (delta_upper - delta_lower) + delta_lower).requires_grad_()
    opt = AdamClipping(params=[delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    for _ in range(attack_iters):
        if time.time() > deadline:
            return False, None
        inputs = torch.max(torch.min(X + delta, data_max), data_min)
        flat = inputs.reshape(-1, *input_shape[1:])
        output = net(flat)
        output = output.view(X.shape[0], *extra_dim, output.shape[-1])
        if check_adv_multi(inputs, output, serialized_conditions, data_max, data_min):
            return True, _pick_single_adv(
                inputs, output, serialized_conditions, data_max, data_min, input_shape)
        loss = get_loss(None, output, serialized_conditions)
        sm_loss = _gtrsb_signmerge_loss(net, num_restarts, num_specs)
        (loss.sum() + sm_loss.sum()).backward()
        opt.step(clipping=True, lower_limit=delta_lower, upper_limit=delta_upper, sign=1)
        opt.zero_grad(set_to_none=True)
        scheduler.step()
    return False, None
