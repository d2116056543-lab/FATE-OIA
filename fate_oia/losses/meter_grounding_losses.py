from __future__ import annotations

import torch
from torch import Tensor


DEFAULT_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (9, 15),
    (10, 16),
    (11, 17),
    (12, 18),
    (13, 19),
    (14, 20),
)


def _map_nll_per_factor(predicted: Tensor, target: Tensor) -> Tensor:
    normalized = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return -(normalized * predicted.clamp_min(1e-6).log()).sum(dim=-1)


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _mirror_balance_loss(
    support_per: Tensor,
    counter_per: Tensor,
    support_valid: Tensor,
    counter_valid: Tensor,
    confidence: Tensor,
    mirror_pairs: tuple[tuple[int, int], ...],
) -> Tensor:
    terms: list[Tensor] = []
    weights: list[Tensor] = []
    factor_count = support_per.shape[1]
    for left, right in mirror_pairs:
        if left >= factor_count or right >= factor_count:
            continue
        support_pair_valid = support_valid[:, left] & support_valid[:, right]
        counter_pair_valid = counter_valid[:, left] & counter_valid[:, right]
        terms.extend(
            [
                (support_per[:, left] - support_per[:, right]).abs(),
                (counter_per[:, left] - counter_per[:, right]).abs(),
            ]
        )
        weights.extend(
            [
                support_pair_valid.to(confidence.dtype)
                * torch.minimum(confidence[:, left], confidence[:, right]),
                counter_pair_valid.to(confidence.dtype)
                * torch.minimum(confidence[:, left], confidence[:, right]),
            ]
        )
    if not terms:
        return support_per.new_zeros(())
    return _weighted_mean(torch.stack(terms, dim=1), torch.stack(weights, dim=1))


def meter_grounding_loss(
    output: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    mirror_pairs: tuple[tuple[int, int], ...] = DEFAULT_MIRROR_PAIRS,
) -> dict[str, Tensor]:
    support_valid = targets["factor_support_valid"].bool()
    counter_valid = targets["factor_counter_valid"].bool()
    confidence = targets.get(
        "factor_source_conf",
        torch.ones_like(output["factor_support_score"]),
    ).to(device=output["factor_support_score"].device, dtype=output["factor_support_score"].dtype)

    support_per = _map_nll_per_factor(
        output["factor_support_map"],
        targets["factor_support_map"].flatten(2),
    )
    counter_per = _map_nll_per_factor(
        output["factor_counter_map"],
        targets["factor_counter_map"].flatten(2),
    )
    support = _weighted_mean(support_per, support_valid * confidence)
    counter = _weighted_mean(counter_per, counter_valid * confidence)

    support_score = output["factor_support_score"]
    counter_score = output["factor_counter_score"]
    support_presence = torch.relu(0.10 - support_score)
    counter_presence = torch.relu(0.10 - counter_score)
    support_only = support_valid & ~counter_valid
    counter_only = counter_valid & ~support_valid
    signed_direction = (
        torch.relu(0.10 - support_score + counter_score) * support_only
        + torch.relu(0.10 - counter_score + support_score) * counter_only
    )
    evidence = _weighted_mean(
        support_presence * support_valid
        + counter_presence * counter_valid
        + signed_direction,
        (support_valid | counter_valid) * confidence,
    )

    mirror = _mirror_balance_loss(
        support_per,
        counter_per,
        support_valid,
        counter_valid,
        confidence,
        mirror_pairs,
    )
    valid_any = (support_valid | counter_valid).to(output["factor_support_score"].dtype) * confidence
    compact_value = output["factor_support_map"].square().sum(-1) + output["factor_counter_map"].square().sum(-1)
    compact = (compact_value * valid_any).sum() / valid_any.sum().clamp_min(1.0)
    compact = (compact + 1e-6).reciprocal() * (valid_any.sum() > 0).to(compact.dtype)
    total = support + counter + evidence + 0.05 * mirror + 0.02 * compact
    return {
        "support": support,
        "counter": counter,
        "evidence": evidence,
        "mirror": mirror,
        "compact": compact,
        "total": total,
    }
