from __future__ import annotations

from typing import Any

import torch


def influence_delta(original_logits: torch.Tensor, intervened_logits: torch.Tensor) -> torch.Tensor:
    return original_logits.softmax(-1).max(-1).values - intervened_logits.softmax(-1).max(-1).values


def summarize_intervention_audit(intervention_report: dict[str, Any]) -> dict[str, Any]:
    """Compute model-level counterfactual dependence from real intervention deltas."""
    results = intervention_report.get("results", {})
    def action_delta(name: str) -> float:
        return float(results.get(name, {}).get("action_prob_l1_delta", 0.0))

    def exp_delta(name: str) -> float:
        return float(results.get(name, {}).get("exp29_prob_l1_delta", 0.0))

    factor_names = ["regime_off", "phase_off", "source_off", "factor_off", "predicate_off", "evidence_tube_off"]
    temporal_names = ["temporal_reverse", "temporal_shuffle", "last_frame_only", "prefix_5", "prefix_10"]
    summary = {
        "claim_boundary": "model-level counterfactual dependence, not real-world causality",
        "decision_dependence_action_delta": {name: action_delta(name) for name in factor_names},
        "decision_dependence_exp29_delta": {name: exp_delta(name) for name in factor_names},
        "temporal_necessity_action_delta": {name: action_delta(name) for name in temporal_names},
        "lag_necessity_action_delta": action_delta("lag_disabled"),
        "evidence_specificity_proxy": action_delta("evidence_tube_off") - action_delta("equal_mass_random"),
        "predicate_specificity_proxy": action_delta("predicate_off") - action_delta("equal_mass_random"),
        "nonzero_intervention_count": int(intervention_report.get("nonzero_delta_count", 0)),
        "available": bool(results),
    }
    summary["faithfulness_ready"] = bool(
        summary["available"]
        and summary["nonzero_intervention_count"] > 0
        and summary["evidence_specificity_proxy"] > 0
    )
    return summary
