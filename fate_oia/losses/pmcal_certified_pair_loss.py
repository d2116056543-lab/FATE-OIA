from __future__ import annotations

import torch
import torch.nn.functional as F


def certified_near_boundary_pair_loss(
    reason_logits: torch.Tensor,
    reason_targets: torch.Tensor,
    pu_state: dict,
    *,
    near_boundary_delta: float = 0.35,
    margin: float = 0.20,
    cap_ratio: float = 0.08,
    reference_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    pos = pu_state.get("positive_mask", reason_targets).float()
    neg = pu_state.get("reliable_negative_mask", torch.zeros_like(reason_targets)).float()
    near = (reason_logits.detach().abs() <= float(near_boundary_delta)).float()
    pair_mask = pos[:, :, None] * neg[:, None, :] * near[:, :, None] * near[:, None, :]
    if pair_mask.sum() <= 0:
        zero = reason_logits.sum() * 0.0
        return zero, {"certified_pair_count": 0, "near_boundary_pair_count": 0, "pair_loss_capped": 0.0}
    z_pos = reason_logits[:, :, None]
    z_neg = reason_logits[:, None, :]
    loss_mat = F.relu(float(margin) - z_pos + z_neg) * pair_mask
    raw = loss_mat.sum() / pair_mask.sum().clamp_min(1.0)
    if reference_loss is not None:
        raw = torch.minimum(raw, reference_loss.detach() * float(cap_ratio))
    return raw, {
        "certified_pair_count": int(pair_mask.sum().detach().cpu()),
        "near_boundary_pair_count": int(pair_mask.sum().detach().cpu()),
        "pair_loss_capped": float(raw.detach().cpu()),
    }
