from __future__ import annotations

import torch


def gradient_budget_scale(main_loss: torch.Tensor, aux_loss: torch.Tensor, rho: float = 0.15) -> tuple[torch.Tensor, dict[str, float]]:
    main_mag = main_loss.detach().abs().clamp_min(1e-6)
    aux_mag = aux_loss.detach().abs().clamp_min(1e-6)
    scale = torch.clamp(rho * main_mag / aux_mag, max=1.0)
    return aux_loss * scale, {"gradient_budget_rho": float(rho), "aux_scale": float(scale.detach().cpu())}
