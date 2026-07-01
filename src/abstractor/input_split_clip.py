import torch
from torch import Tensor
from typing import Tuple, Union


def concretize_bounds(xhat: Tensor, eps: Tensor, lA: Tensor, lbias: Tensor, is_lower: bool = True):
    sign = 1 if is_lower else -1
    eop = 'bsn,bn->bs'
    return torch.einsum(eop, lA, xhat) - sign * torch.einsum(eop, lA.abs(), eps) + lbias


def _clip_main_fn(x_L: Tensor, x_U: Tensor, lA: Tensor, lbias: Union[Tensor, None], thresholds: Tensor,
                  dm_lb: Tensor, num_iters: int, is_lower: int) -> Tuple[Tensor, Tensor]:
    sign = 1 if is_lower else -1
    batches, num_spec, _ = lA.shape
    thresholds = thresholds.reshape(batches, num_spec, 1)
    dm_lb = dm_lb.reshape(batches, num_spec, 1)
    for i in range(num_iters):
        xhat = (x_U + x_L) / 2
        eps = (x_U - x_L) / 2
        if i > 0:
            dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower).reshape(batches, num_spec, 1)
        eop = 'bsn,bn->bsn'
        concrete_minus_one = dm_lb - torch.einsum(eop, lA, xhat) + sign * torch.einsum(eop, lA.abs(), eps)
        curr_x = (thresholds - concrete_minus_one) / lA
        x_U_candidates = torch.where(lA > 0, curr_x, torch.inf)
        x_L_candidates = torch.where(lA < 0, curr_x, -torch.inf)
        x_U = torch.min(x_U_candidates.amin(dim=1), x_U)
        x_L = torch.max(x_L_candidates.amax(dim=1), x_L)
    return x_L, x_U


def clip_domains(
        x_L: Tensor,
        x_U: Tensor,
        thresholds: Tensor,
        lA: Tensor,
        lbias: Tensor | None = None,
        dm_lb: Tensor | None = None,
        is_lower: bool = True,
        num_iters: int = 1,
        calculate_volume: bool = False,
) -> Tuple[Tensor, Tensor]:
    x_L_shape = x_L.shape
    x_U_shape = x_U.shape
    lA = lA.flatten(2)
    batches, num_spec, input_dim = lA.shape

    x_L = x_L.clone().view(batches, input_dim)
    x_U = x_U.clone().view(batches, input_dim)
    xhat = (x_U + x_L) / 2
    eps = (x_U - x_L) / 2

    if dm_lb is None:
        if lbias is None:
            raise ValueError('lbias or dm_lb required for clip_domains')
        dm_lb = concretize_bounds(xhat, eps, lA, lbias, is_lower)

    if lbias is None:
        lbias = torch.zeros(lA.shape[0], lA.shape[1], device=lA.device, dtype=lA.dtype)
    x_L, x_U = _clip_main_fn(x_L, x_U, lA, lbias, thresholds, dm_lb, num_iters, is_lower)
    return x_L.view(x_L_shape), x_U.view(x_U_shape)
