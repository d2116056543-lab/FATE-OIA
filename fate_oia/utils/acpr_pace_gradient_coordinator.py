from __future__ import annotations

import torch


def common_descent_gradient(g_action: torch.Tensor, g_exp: torch.Tensor, max_common_scale: float = 2.0, epsilon: float = 1.0e-12) -> tuple[torch.Tensor, dict[str, float]]:
    if g_action.numel() == 0:
        return g_exp, {"gradient_conflict": 0.0, "gradient_alpha": 0.0, "gradient_cosine": 0.0, "gradient_common_norm": float(g_exp.norm().detach().cpu())}
    if g_exp.numel() == 0:
        return g_action, {"gradient_conflict": 0.0, "gradient_alpha": 1.0, "gradient_cosine": 0.0, "gradient_common_norm": float(g_action.norm().detach().cpu())}
    dot = torch.dot(g_action, g_exp)
    denom = g_action.norm().clamp_min(epsilon) * g_exp.norm().clamp_min(epsilon)
    cosine = dot / denom
    if dot >= 0:
        common = g_action + g_exp
        alpha = torch.ones((), device=g_action.device, dtype=g_action.dtype)
        conflict = False
    else:
        d = g_action - g_exp
        alpha = torch.clamp(-torch.dot(g_exp, d) / torch.dot(d, d).clamp_min(epsilon), 0.0, 1.0)
        common = alpha * g_action + (1.0 - alpha) * g_exp
        max_norm = max_common_scale * torch.maximum(g_action.norm(), g_exp.norm()).clamp_min(epsilon)
        if common.norm() > max_norm:
            common = common * (max_norm / common.norm().clamp_min(epsilon))
        conflict = True
    return common, {
        "gradient_conflict": float(1.0 if conflict else 0.0),
        "gradient_alpha": float(alpha.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_common_norm": float(common.norm().detach().cpu()),
        "gradient_action_norm": float(g_action.norm().detach().cpu()),
        "gradient_exp_norm": float(g_exp.norm().detach().cpu()),
    }


class PACEGradientCoordinator:
    def __init__(self, enabled: bool = False, start_epoch: int = 3, max_common_scale: float = 2.0) -> None:
        self.enabled = bool(enabled)
        self.start_epoch = int(start_epoch)
        self.max_common_scale = float(max_common_scale)

    def should_apply(self, epoch: int) -> bool:
        return self.enabled and int(epoch) >= self.start_epoch
