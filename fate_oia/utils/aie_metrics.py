from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fate_oia.metrics import multilabel_metrics_from_logits


def aie_branch_metrics(action_logits: Tensor, reason_logits: Tensor, action_target: Tensor, reason_target: Tensor, threshold: float | Tensor = 0.5) -> dict[str, Any]:
    action = multilabel_metrics_from_logits(action_logits.float(), action_target.float(), threshold=threshold, prefix="Act_")
    reason_threshold = threshold[4:] if isinstance(threshold, Tensor) and threshold.numel() == 25 else threshold
    action_threshold = threshold[:4] if isinstance(threshold, Tensor) and threshold.numel() == 25 else threshold
    if isinstance(threshold, Tensor) and threshold.numel() == 25:
        action = multilabel_metrics_from_logits(action_logits.float(), action_target.float(), threshold=action_threshold, prefix="Act_")
    reason = multilabel_metrics_from_logits(reason_logits.float(), reason_target.float(), threshold=reason_threshold, prefix="Exp_")
    return {**action, **reason, "joint": 0.5 * action["Act_mF1"] + 0.5 * reason["Exp_mF1"]}


def probe_health_metrics(evidence_map: Tensor, contribution: Tensor) -> dict[str, float]:
    maps = evidence_map.float().clamp_min(1e-9)
    entropy = -(maps * maps.log()).sum(-1)
    magnitude = contribution.float().abs()
    share = magnitude / magnitude.sum(-1, keepdim=True).clamp_min(1e-8)
    dominant = share.max(-1).values
    normalized = maps / maps.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarity = torch.einsum("bakn,baln->bakl", normalized, normalized)
    off_diag = (similarity.sum((-1, -2)) - 4) / 12
    return {
        "probe_map_entropy": float(entropy.mean().detach().cpu()),
        "dominant_probe_ratio": float(dominant.mean().detach().cpu()),
        "dominant_probe_over_0p9_rate": float((dominant > 0.9).float().mean().detach().cpu()),
        "probe_pairwise_overlap": float(off_diag.mean().detach().cpu()),
        "probe_effective_count": float(torch.exp(-(share.clamp_min(1e-9) * share.clamp_min(1e-9).log()).sum(-1)).mean().detach().cpu()),
    }


def _rank(values: Tensor) -> Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def spearman_correlation(left: Tensor, right: Tensor) -> float:
    if left.numel() < 2 or right.numel() != left.numel():
        return 0.0
    x, y = _rank(left.float()), _rank(right.float())
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum().div(denominator.clamp_min(1e-8)).detach().cpu())


def counterfactual_metrics(cf: dict[str, Any] | None) -> dict[str, Any]:
    if not cf or int(cf.get("cf_valid_count", 0)) == 0:
        return {
            "available": False,
            "cf_valid_count": 0,
            "cf_invalid_count": sum((cf or {}).get("cf_invalid_reason_counts", {}).values()),
            "selected_drop_mean": None,
            "control_drop_mean": None,
            "selected_minus_control_mean": None,
            "contribution_effect_spearman": None,
            "max_selected_control_overlap": None,
            "positive_action_directions": 0,
            "per_action_selected_minus_control": {},
        }
    valid = cf["valid_mask"].bool()
    selected = cf["selected_drop"][valid].float()
    control = cf["control_drop"][valid].float()
    effect = cf["selected_minus_control"][valid].float()
    support = cf["supportive_contribution"][valid].float()
    per_action: dict[str, float] = {}
    for action_id in range(4):
        values = [row["selected_minus_control"] for row in cf["cases"] if int(row["action_id"]) == action_id]
        if values:
            per_action[str(action_id)] = float(sum(values) / len(values))
    return {
        "available": True,
        "cf_valid_count": int(valid.sum().item()),
        "cf_invalid_count": sum(cf.get("cf_invalid_reason_counts", {}).values()),
        "cf_invalid_reason_counts": cf.get("cf_invalid_reason_counts", {}),
        "selected_drop_mean": float(selected.mean().detach().cpu()),
        "control_drop_mean": float(control.mean().detach().cpu()),
        "selected_minus_control_mean": float(effect.mean().detach().cpu()),
        "contribution_effect_spearman": spearman_correlation(support, effect),
        "max_selected_control_overlap": max(cf.get("selected_control_overlap", [1.0])),
        "positive_action_directions": sum(value > 0 for value in per_action.values()),
        "per_action_selected_minus_control": per_action,
        "wrong_probe_drop_mean": float(cf["wrong_probe_drop"].float().mean().detach().cpu()),
        "wrong_action_drop_mean": float(cf["wrong_action_drop"].float().mean().detach().cpu()),
    }


def counterfactual_case_metrics(cases: list[dict[str, Any]], invalid_count: int = 0) -> dict[str, Any]:
    if not cases:
        return counterfactual_metrics({"cf_valid_count": 0, "cf_invalid_reason_counts": {"all": invalid_count}})
    support = torch.tensor([row["supportive_contribution"] for row in cases])
    effect = torch.tensor([row["selected_minus_control"] for row in cases])
    per_action: dict[str, float] = {}
    for action_id in range(4):
        values = [row["selected_minus_control"] for row in cases if int(row["action_id"]) == action_id]
        if values:
            per_action[str(action_id)] = float(sum(values) / len(values))
    return {
        "available": True,
        "cf_valid_count": len(cases),
        "cf_invalid_count": int(invalid_count),
        "selected_drop_mean": float(sum(row["selected_drop"] for row in cases) / len(cases)),
        "control_drop_mean": float(sum(row["control_drop"] for row in cases) / len(cases)),
        "selected_minus_control_mean": float(effect.mean()),
        "contribution_effect_spearman": spearman_correlation(support, effect),
        "max_selected_control_overlap": max(row["selected_control_overlap"] for row in cases),
        "positive_action_directions": sum(value > 0 for value in per_action.values()),
        "per_action_selected_minus_control": per_action,
        "wrong_probe_drop_mean": float(sum(row["wrong_probe_drop"] for row in cases) / len(cases)),
        "wrong_action_drop_mean": float(sum(row["wrong_action_drop"] for row in cases) / len(cases)),
    }
