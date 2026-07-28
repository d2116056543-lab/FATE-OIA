from __future__ import annotations

import torch
from torch import Tensor


def _map_nll(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    normalized = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    per = -(normalized * predicted.clamp_min(1e-6).log()).sum(dim=-1)
    mask = valid.to(dtype=per.dtype)
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def meter_grounding_loss(output: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
    support = _map_nll(output["factor_support_map"], targets["factor_support_map"].flatten(2), targets["factor_support_valid"])
    counter = _map_nll(output["factor_counter_map"], targets["factor_counter_map"].flatten(2), targets["factor_counter_valid"])
    valid = (targets["factor_support_valid"] | targets["factor_counter_valid"]).to(output["factor_support_score"].dtype)
    evidence = ((output["factor_support_score"] - output["factor_counter_score"]).square() * valid).sum() / valid.sum().clamp_min(1.0)
    valid_any = (targets["factor_support_valid"] | targets["factor_counter_valid"]).to(output["factor_support_score"].dtype)
    compact_value = output["factor_support_map"].square().sum(-1) + output["factor_counter_map"].square().sum(-1)
    compact = (compact_value * valid_any).sum() / valid_any.sum().clamp_min(1.0)
    compact = (compact + 1e-6).reciprocal() * (valid_any.sum() > 0).to(compact.dtype)
    total = support + counter + evidence + 0.02 * compact
    return {"support": support, "counter": counter, "evidence": evidence, "compact": compact, "total": total}
