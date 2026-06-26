from __future__ import annotations

import torch


def pair_budget_ratio(epoch: int) -> float:
    if epoch < 3:
        return 0.0
    if epoch < 8:
        return 0.20
    return 0.10


def apply_pair_budget(
    pair_raw: torch.Tensor,
    main_loss: torch.Tensor,
    epoch: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = pair_budget_ratio(int(epoch))
    cap = ratio * main_loss.detach()
    denom = pair_raw.detach().abs().clamp_min(eps)
    scale = torch.clamp(cap / denom, max=1.0) if ratio > 0 else torch.zeros_like(pair_raw.detach())
    used = pair_raw * scale
    stats = {
        "pair_budget_ratio": float(ratio),
        "pair_raw_weighted": float(pair_raw.detach().cpu()),
        "pair_used_weighted": float(used.detach().cpu()),
        "pair_budget_cap": float(cap.detach().cpu()),
        "pair_budget_scale": float(scale.detach().cpu()),
        "pair_budget_active": bool(ratio > 0),
        "pair_to_main_raw": float((pair_raw.detach() / main_loss.detach().clamp_min(eps)).cpu()),
        "pair_to_main_used": float((used.detach() / main_loss.detach().clamp_min(eps)).cpu()),
    }
    return used, stats

