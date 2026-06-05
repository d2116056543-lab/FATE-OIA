from __future__ import annotations

import torch


def _norm(grads: list[torch.Tensor | None], device: torch.device) -> torch.Tensor:
    vals = [g.detach().pow(2).sum() for g in grads if g is not None]
    if not vals:
        return torch.tensor(0.0, device=device)
    return torch.sqrt(torch.stack(vals).sum())


def true_gradient_budget(main_loss: torch.Tensor, aux_loss: torch.Tensor, params, rho: float = 0.10, eps: float = 1e-8, retain_graph: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    ps = [p for p in params if getattr(p, "requires_grad", False)]
    if not ps or float(aux_loss.detach().abs().cpu()) == 0.0:
        return aux_loss, {"norm_main": 0.0, "norm_aux": 0.0, "budget_scale": 1.0, "rho": float(rho), "used_true_grad_norm": bool(ps)}
    g_main = torch.autograd.grad(main_loss, ps, retain_graph=True, allow_unused=True)
    g_aux = torch.autograd.grad(aux_loss, ps, retain_graph=retain_graph, allow_unused=True)
    nm = _norm(list(g_main), aux_loss.device)
    na = _norm(list(g_aux), aux_loss.device)
    scale = torch.clamp(float(rho) * nm / (na + eps), max=1.0)
    return aux_loss * scale.detach(), {"norm_main": float(nm.detach().cpu()), "norm_aux": float(na.detach().cpu()), "budget_scale": float(scale.detach().cpu()), "rho": float(rho), "used_true_grad_norm": True}
