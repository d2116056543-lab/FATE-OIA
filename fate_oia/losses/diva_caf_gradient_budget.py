from __future__ import annotations

import torch


def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True, allow_unused=True)
    norms = []
    for grad in grads:
        if grad is not None:
            norms.append(grad.pow(2).sum())
    if not norms:
        return loss.new_zeros(())
    return torch.sqrt(torch.stack(norms).sum() + 1e-12)


def apply_gradient_budget(main_loss: torch.Tensor, aux_loss: torch.Tensor, params, rho: float = 0.15):
    params = [p for p in params if getattr(p, "requires_grad", False)]
    if len(params) == 0:
        scale = aux_loss.new_tensor(float(rho))
        return main_loss + scale * aux_loss, {"norm_main": 0.0, "norm_aux": 0.0, "budget_scale": float(scale.detach().cpu())}
    norm_main = _grad_norm(main_loss, params)
    norm_aux = _grad_norm(aux_loss, params)
    target = float(rho) * norm_main.detach()
    scale = torch.clamp(target / (norm_aux.detach() + 1e-8), max=1.0)
    scaled_loss = main_loss + scale * aux_loss
    stats = {
        "norm_main": float(norm_main.detach().cpu()),
        "norm_aux": float(norm_aux.detach().cpu()),
        "budget_scale": float(scale.detach().cpu()),
    }
    return scaled_loss, stats
