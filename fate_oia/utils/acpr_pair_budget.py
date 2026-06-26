from __future__ import annotations

import torch


def pair_budget_ratio(epoch: int) -> float:
    if epoch < 3:
        return 0.0
    if epoch <= 7:
        return 0.20
    return 0.10


def apply_pair_budget(
    pair_raw: torch.Tensor,
    action_direct_loss: torch.Tensor,
    reason_partial_loss: torch.Tensor,
    epoch: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = pair_budget_ratio(int(epoch))
    main_ref = (action_direct_loss.detach() + reason_partial_loss.detach()).clamp_min(eps)
    cap = ratio * main_ref
    scale = torch.clamp(cap / pair_raw.detach().clamp_min(eps), max=1.0)
    used = pair_raw * scale
    return used, {
        "pair_budget_ratio": float(ratio),
        "pair_budget_cap": float(cap.detach().cpu()),
        "pair_budget_scale": float(scale.detach().cpu()),
        "pair_main_ref": float(main_ref.detach().cpu()),
    }
