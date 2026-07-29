from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_MIRROR_PAIRS = ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19))


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def source_weighted_anchor_loss(
    predicted: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    target = target.flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-6)
    nll = -(target * predicted.clamp_min(1e-8).log()).sum(-1)
    intersection = (target * predicted).sum(-1)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        target.sum(-1) + predicted.sum(-1) + 1e-6
    )
    weight = valid.to(predicted) * source_weight.to(predicted)
    return _weighted_mean(nll, weight), _weighted_mean(dice, weight)


def conditional_state_ce(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
) -> Tensor:
    safe_target = target.clamp_min(0)
    per = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
    return _weighted_mean(per, valid.to(logits) * source_weight.to(logits))


def observability_objective(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    source_weight: Tensor,
    tau: Tensor,
) -> tuple[Tensor, Tensor]:
    per = F.binary_cross_entropy_with_logits(logits, target.to(logits), reduction="none")
    weight = valid.to(logits) * source_weight.to(logits)
    bce = _weighted_mean(per, weight)
    probability = torch.sigmoid(logits)
    per_factor_weight = weight.sum(0)
    observed_mean = (probability * weight).sum(0) / per_factor_weight.clamp_min(1.0)
    available = per_factor_weight > 0
    coverage = (observed_mean - tau.to(logits)).abs()
    return bce, _weighted_mean(coverage, available.to(logits))


def discrimination_and_mirror_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    mirror_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
) -> tuple[Tensor, Tensor]:
    token = output["factor_typed_token"]
    state = output["factor_state_prob"]
    source = targets["factor_source_weight"].to(token)
    same_type_terms: list[Tensor] = []
    for left, right in mirror_pairs:
        clean = F.cosine_similarity(token[:, left], token[:, left].detach(), dim=-1)
        wrong = F.cosine_similarity(token[:, left], token[:, right].detach(), dim=-1)
        weight = torch.minimum(source[:, left], source[:, right])
        same_type_terms.append(_weighted_mean(torch.relu(0.05 + wrong - clean), weight))
    discrimination = (
        torch.stack(same_type_terms).mean()
        if same_type_terms
        else token.new_zeros(())
    )
    anchor = output["factor_anchor_map"].view(token.shape[0], token.shape[1], 45, 80)
    mirror_terms: list[Tensor] = []
    for left, right in mirror_pairs:
        map_term = (torch.flip(anchor[:, left], dims=[-1]) - anchor[:, right]).abs().mean((1, 2))
        state_term = (state[:, left] - state[:, right]).abs().mean(-1)
        weight = torch.minimum(source[:, left], source[:, right])
        mirror_terms.append(_weighted_mean(map_term + state_term, weight))
    mirror = torch.stack(mirror_terms).mean() if mirror_terms else token.new_zeros(())
    return discrimination, mirror


def meter_grounding_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    observability_tau: Tensor | None = None,
) -> dict[str, Tensor]:
    source = targets["factor_source_weight"].to(output["factor_anchor_map"])
    anchor_nll, anchor_dice = source_weighted_anchor_loss(
        output["factor_anchor_map"],
        targets["factor_anchor_map"],
        targets["factor_anchor_valid"],
        source,
    )
    state = conditional_state_ce(
        output["factor_state_logits"],
        targets["factor_state_target"],
        targets["factor_state_valid"],
        source,
    )
    tau = (
        torch.full((output["factor_observability"].shape[1],), 0.5, device=source.device)
        if observability_tau is None
        else observability_tau.to(source)
    )
    obs_bce, obs_coverage = observability_objective(
        output["factor_observability_logit"],
        targets["factor_observability"],
        targets["factor_observability_valid"],
        source,
        tau,
    )
    discrimination, mirror = discrimination_and_mirror_loss(output, targets)
    anchor = anchor_nll + anchor_dice
    observability = obs_bce + obs_coverage
    total = 0.10 * anchor + 0.10 * state + 0.03 * observability + 0.05 * (
        discrimination + mirror
    )
    return {
        "anchor_nll": anchor_nll,
        "anchor_dice": anchor_dice,
        "anchor": anchor,
        "state": state,
        "observability_bce": obs_bce,
        "observability_coverage": obs_coverage,
        "observability": observability,
        "discrimination": discrimination,
        "mirror": mirror,
        "total": total,
    }
