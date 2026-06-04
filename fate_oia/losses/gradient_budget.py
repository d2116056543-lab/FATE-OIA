from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _grad_norm(grads: Sequence[torch.Tensor | None], params: Sequence[nn.Parameter]) -> torch.Tensor:
    device = params[0].device if params else torch.device("cpu")
    total = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return total.sqrt()


def compute_gradient_budget_scale(
    main_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    shared_params: Sequence[nn.Parameter],
    rho: float = 0.15,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    params = [p for p in shared_params if p.requires_grad]
    if not params:
        scale = aux_loss.new_tensor(0.0)
        return scale, {
            "norm_main": 0.0,
            "norm_aux": 0.0,
            "budget_scale": 0.0,
            "aux_scale": 0.0,
            "rho": float(rho),
            "gradient_budget_rho": float(rho),
            "used_true_grad_norm": True,
        }
    g_main = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    g_aux = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)
    norm_main = _grad_norm(g_main, params)
    norm_aux = _grad_norm(g_aux, params)
    scale = torch.clamp(float(rho) * norm_main / (norm_aux + float(eps)), max=1.0)
    return scale.detach(), {
        "norm_main": float(norm_main.detach().cpu()),
        "norm_aux": float(norm_aux.detach().cpu()),
        "budget_scale": float(scale.detach().cpu()),
        "aux_scale": float(scale.detach().cpu()),
        "rho": float(rho),
        "gradient_budget_rho": float(rho),
        "used_true_grad_norm": True,
    }


def gradient_budget_scale(
    main_loss: torch.Tensor,
    aux_loss: torch.Tensor,
    rho: float = 0.15,
    shared_params: Sequence[nn.Parameter] | None = None,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    if shared_params is None:
        raise ValueError("gradient_budget_scale requires shared_params for true gradient-norm budgeting")
    scale, stats = compute_gradient_budget_scale(main_loss, aux_loss, shared_params, rho=rho)
    return aux_loss * scale, stats
