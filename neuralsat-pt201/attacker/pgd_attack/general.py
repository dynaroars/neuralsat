from beartype import beartype
import torch
import time

from onnx2pytorch.convert.model import ConvertModel

from .util import get_loss, check_adv_multi, serialize_specs
from util.misc.adam_clipping import AdamClipping


@beartype
def attack(model: ConvertModel, x: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, 
           cs: torch.Tensor, rhs: torch.Tensor, timeout: float,
           attack_iters: int = 100, num_restarts: int = 30) -> tuple[bool, torch.Tensor | None]:
    # set all parameters without gradient, this can speedup things significantly.
    grad_status = {}
    for p in model.parameters():
        grad_status[p] = p.requires_grad
        p.requires_grad_(False)

    output = model(x).detach()
    
    # specifications
    serialized_conditions = serialize_specs(x, cs, rhs)
    
    # sanity check
    output = output[:, None, None].repeat(1, 1, len(serialized_conditions[-1][0]), 1)
    if check_adv_multi(x, output, serialized_conditions, data_max.unsqueeze(1), data_min.unsqueeze(1)):
        x = x[:, None, None]
        return True, x.repeat(1, 1, len(serialized_conditions[-1][0]), *[1]*(x.ndim - 3)).detach()

    data_min = data_min.to(x.device)
    data_max = data_max.to(x.device)
    attack_images = general_attack(
        model=model, 
        X=x, 
        data_min=data_min, 
        data_max=data_max, 
        serialized_conditions=serialized_conditions,
        attack_iters=attack_iters, 
        num_restarts=num_restarts, 
        use_gama=False,
        timeout=timeout,
    )
    
    if attack_images is None:
        attack_images = general_attack(
            model=model, 
            X=x, 
            data_min=data_min, 
            data_max=data_max, 
            serialized_conditions=serialized_conditions,
            attack_iters=attack_iters, 
            num_restarts=num_restarts, 
            use_gama=True,
            timeout=timeout,
        )

    if attack_images is not None:
        return True, attack_images.detach()
    
    # set back to original requires_grad status.
    for p in model.parameters():
        p.requires_grad_(grad_status[p])
        
    return False, None



@beartype
def general_attack(model: ConvertModel, X: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor, 
                   serialized_conditions: tuple[torch.Tensor, torch.Tensor, list[list[int]]], timeout: float,
                   use_gama: bool = False, num_restarts: int = 10, attack_iters: int = 100, 
                   only_replicate_restarts: bool = False) -> torch.Tensor | None:
    # hyper params
    lr_decay = 0.99
    gama_lambda = 10.0

    # shapes
    input_shape = (X.shape[0], *X.shape[2:]) if only_replicate_restarts else X.size()
    num_classes = serialized_conditions[0].shape[-1]
    num_specs = len(serialized_conditions[-1][0])
    extra_dim = (num_restarts,) if only_replicate_restarts else (num_restarts, num_specs) 

    # [1, 1, num_spec, *input_shape]
    data_min = data_min.unsqueeze(1)
    data_max = data_max.unsqueeze(1)

    X_ndim_orig = X.ndim
    # shape of x: [num_example, *shape_of_x]
    X = X.view(X.shape[0], *[1] * len(extra_dim), *X.shape[1:])
    X = X.expand(-1, *extra_dim, *(-1,) * (X_ndim_orig - 1))
    extra_dim = (X.shape[1], X.shape[2])

    delta_lower_limit = data_min - X
    delta_upper_limit = data_max - X
    
    lr = torch.max(data_max - data_min).item() / 8
    delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
        
    # optimizer
    opt = AdamClipping(params=[delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)

    start = time.time()
    for _ in range(attack_iters):
        inputs = torch.max(torch.min((X + delta), data_max), data_min)
        output = model(inputs.view(-1, *input_shape[1:])).view(input_shape[0], *extra_dim, num_classes)
        
        # early stop
        if check_adv_multi(inputs, output, serialized_conditions, data_max, data_min):
            return inputs
        
        if time.time() - start > timeout:
            break
        
        origin_out = torch.softmax(model(X.reshape(-1, *input_shape[1:])), 1).view(output.shape) if use_gama else None

        loss = get_loss(origin_out, output, serialized_conditions, gama_lambda if use_gama else 0.0)
        loss.sum().backward()

        # optimize
        opt.step(clipping=True, lower_limit=delta_lower_limit, upper_limit=delta_upper_limit, sign=1)
        opt.zero_grad(set_to_none=True)
        scheduler.step()
        gama_lambda *= 0.9
        

    return None
    