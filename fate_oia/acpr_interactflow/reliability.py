from __future__ import annotations

import torch


def heuristic_cluster_reliability(
    support_count: torch.Tensor,
    phrase_concentration: torch.Tensor | None = None,
    text_predicate_agreement: torch.Tensor | None = None,
    action_compatibility: torch.Tensor | None = None,
    contradiction_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    score = 0.35 * support_count.float().clamp_min(1).log()
    if phrase_concentration is not None:
        score = score + 0.20 * phrase_concentration.float()
    if text_predicate_agreement is not None:
        score = score + 0.20 * text_predicate_agreement.float()
    if action_compatibility is not None:
        score = score + 0.15 * action_compatibility.float()
    if contradiction_rate is not None:
        score = score - 0.25 * contradiction_rate.float()
    return torch.sigmoid(score)
