from __future__ import annotations

import torch
import torch.nn.functional as F


def certified_near_boundary_pair_loss(
    reason_logits: torch.Tensor,
    reason_targets: torch.Tensor,
    pu_state: dict | None = None,
    *,
    reliable_mask: torch.Tensor | None = None,
    boundary: float | None = None,
    near_boundary_delta: float = 0.35,
    margin: float = 0.20,
    cap_ratio: float = 0.08,
    reference_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Reason-specific near-boundary pair loss.

    A valid pair is same reason r, positive sample i, reliable-negative sample j,
    both near the deploy decision boundary. This avoids treating all y=0 as
    certified negatives and avoids cross-reason ranking shortcuts.
    """
    if boundary is not None:
        near_boundary_delta = float(boundary)
    pos = (pu_state or {}).get("positive_mask", reason_targets).float()
    if reliable_mask is not None:
        neg = (1.0 - reason_targets.float()) * reliable_mask.float()
    else:
        neg = (pu_state or {}).get("reliable_negative_mask", torch.zeros_like(reason_targets)).float()
    near = (reason_logits.detach().abs() <= float(near_boundary_delta)).float()
    b, r = reason_logits.shape
    pair_mask = pos[:, None, :] * neg[None, :, :] * near[:, None, :] * near[None, :, :]
    eye = torch.eye(b, device=reason_logits.device, dtype=pair_mask.dtype).view(b, b, 1)
    pair_mask = pair_mask * (1.0 - eye)
    if pair_mask.sum() <= 0:
        zero = reason_logits.sum() * 0.0
        return zero, {
            "certified_pair_count": 0,
            "near_boundary_pair_count": 0,
            "reason_specific_pairs": 0,
            "hard_pair_count": 0,
            "semi_pair_count": 0,
            "easy_pair_count": 0,
            "zero_pair_count": int(b * r),
            "pair_loss_capped": 0.0,
        }
    z_pos = reason_logits[:, None, :]
    z_neg = reason_logits[None, :, :]
    violation = float(margin) - z_pos + z_neg
    loss_mat = F.relu(violation) * pair_mask
    raw = loss_mat.sum() / pair_mask.sum().clamp_min(1.0)
    uncapped = raw
    if reference_loss is not None:
        raw = torch.minimum(raw, reference_loss.detach() * float(cap_ratio))
    active = pair_mask > 0
    hard = ((violation.detach() > 0.0) & active).sum()
    semi = ((violation.detach() <= 0.0) & (violation.detach() > -float(margin)) & active).sum()
    easy = ((violation.detach() <= -float(margin)) & active).sum()
    return raw, {
        "certified_pair_count": int(pair_mask.sum().detach().cpu()),
        "near_boundary_pair_count": int(pair_mask.sum().detach().cpu()),
        "reason_specific_pairs": int((pair_mask.sum(dim=(0, 1)) > 0).sum().detach().cpu()),
        "hard_pair_count": int(hard.detach().cpu()),
        "semi_pair_count": int(semi.detach().cpu()),
        "easy_pair_count": int(easy.detach().cpu()),
        "zero_pair_count": int((pair_mask.sum(dim=(0, 1)) == 0).sum().detach().cpu()),
        "pair_loss_uncapped": float(uncapped.detach().cpu()),
        "pair_loss_capped": float(raw.detach().cpu()),
    }
