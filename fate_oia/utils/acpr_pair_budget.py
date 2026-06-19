from __future__ import annotations

import torch


def apply_pair_budget(
    pair_raw_weighted: torch.Tensor,
    action_group_loss: torch.Tensor,
    exp_group_loss: torch.Tensor,
    ratio: float = 0.25,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    main = action_group_loss.detach() + exp_group_loss.detach()
    cap = float(ratio) * main + float(epsilon)
    if pair_raw_weighted.detach() <= cap:
        used = pair_raw_weighted
        scale = torch.ones((), device=pair_raw_weighted.device, dtype=pair_raw_weighted.dtype)
        active = False
    else:
        scale = cap / pair_raw_weighted.detach().clamp_min(float(epsilon))
        used = pair_raw_weighted * scale
        active = True
    denom = main.clamp_min(float(epsilon))
    return used, {
        "pair_budget_cap": float(cap.detach().cpu()),
        "pair_budget_scale": float(scale.detach().cpu()),
        "pair_budget_active": bool(active),
        "pair_to_main_ratio_raw": float((pair_raw_weighted.detach() / denom).cpu()),
        "pair_to_main_ratio_used": float((used.detach() / denom).cpu()),
    }
