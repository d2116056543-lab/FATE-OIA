from __future__ import annotations

import torch


def compose_final_action_logits(
    visual_logits: torch.Tensor, shadow_logits: torch.Tensor, admitted_actions: torch.Tensor
) -> torch.Tensor:
    if visual_logits.shape != shadow_logits.shape or visual_logits.shape[-1] != 4:
        raise ValueError("action visual and shadow logits must be [B,4]")
    mask = admitted_actions.to(device=visual_logits.device, dtype=torch.bool).view(1, 4)
    return torch.where(mask, shadow_logits, visual_logits)


def partial_action_admission(
    tet_lcb: torch.Tensor,
    tes_lcb: torch.Tensor,
    cca: torch.Tensor,
    visual_ap: torch.Tensor,
    edge_ap: torch.Tensor,
) -> torch.Tensor:
    """Admit an action only from its independently measured edge effects.

    Continuous visual credibility is an uncalibrated diagnostic score.  It is
    intentionally excluded here: using one absolute cV threshold reintroduces
    the CREDO-MAP cold start that target-effect interventions are meant to
    remove.
    """
    values = (tet_lcb, tes_lcb, cca, visual_ap, edge_ap)
    if any(value.shape != (4,) for value in values):
        raise ValueError("partial action admission inputs must be [4]")
    return (tet_lcb > 0.0) & (tes_lcb > 0.0) & (cca >= 0.60) & (edge_ap >= visual_ap - 0.002)
