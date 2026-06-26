from __future__ import annotations

import torch


def common_descent_direction(g_action: torch.Tensor, g_exp: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    dot = torch.dot(g_action, g_exp)
    if dot >= 0:
        out = g_action + g_exp
        alpha = 0.5
        conflict = False
    else:
        diff = g_action - g_exp
        denom = torch.dot(diff, diff).clamp_min(1e-12)
        alpha_t = torch.clamp(torch.dot(g_action, diff) / denom, 0.0, 1.0)
        out = alpha_t * g_exp + (1.0 - alpha_t) * g_action
        alpha = float(alpha_t.detach().cpu())
        conflict = True
    return out, {
        "adapter_gradient_conflict": bool(conflict),
        "adapter_gradient_dot": float(dot.detach().cpu()),
        "adapter_gradient_alpha": float(alpha),
        "adapter_gradient_common_norm": float(out.norm().detach().cpu()),
    }


def flatten_grads(params: list[torch.nn.Parameter]) -> torch.Tensor:
    vals = []
    for p in params:
        if p.grad is None:
            vals.append(torch.zeros_like(p).reshape(-1))
        else:
            vals.append(p.grad.detach().reshape(-1))
    if not vals:
        return torch.empty(0)
    return torch.cat(vals)

