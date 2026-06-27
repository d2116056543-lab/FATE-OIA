from __future__ import annotations

import torch

from fate_oia.losses.acpr_interactflow_losses import action_soft_kl_loss, non_degradation_soft_kl_hinge_loss


def test_action_primary_and_safety_are_soft_target_kl() -> None:
    final = torch.tensor([[1.0, 0.2, -0.4]], requires_grad=True)
    global_logits = torch.tensor([[0.8, 0.4, -0.2]])
    soft_target = torch.tensor([[0.55, 0.35, 0.10]])

    primary = action_soft_kl_loss(final, soft_target)
    safe = non_degradation_soft_kl_hinge_loss(final, global_logits, soft_target, margin=0.01)
    (primary + safe).backward()

    assert torch.isfinite(primary)
    assert torch.isfinite(safe)
    assert final.grad is not None
