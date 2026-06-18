from __future__ import annotations

import torch


def apply_pair_budget(
    pair_logit_loss: torch.Tensor,
    pair_embed_loss: torch.Tensor,
    *,
    pair_logit_weight: float,
    pair_embed_weight: float,
    main_loss: torch.Tensor,
    pair_budget_ratio: float = 0.25,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = float(pair_logit_weight) * pair_logit_loss + float(pair_embed_weight) * pair_embed_loss
    cap = float(pair_budget_ratio) * main_loss.detach()
    scale = torch.minimum(torch.ones_like(raw), cap / raw.detach().clamp_min(eps))
    used = raw * scale
    stats = {
        "loss_pair_weighted_raw": float(raw.detach().cpu()),
        "loss_pair_weighted_used": float(used.detach().cpu()),
        "pair_budget_cap": float(cap.detach().cpu()),
        "pair_budget_scale": float(scale.detach().cpu()),
        "pair_budget_active": bool(float(scale.detach().cpu()) < 0.999999),
        "pair_to_main_ratio_raw": float((raw.detach() / main_loss.detach().clamp_min(eps)).cpu()),
        "pair_to_main_ratio_used": float((used.detach() / main_loss.detach().clamp_min(eps)).cpu()),
    }
    return used, stats
