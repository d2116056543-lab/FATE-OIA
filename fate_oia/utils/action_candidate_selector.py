from __future__ import annotations

from typing import Any

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25


def action_primary_score(metrics: dict[str, Any]) -> float:
    return (
        0.60 * float(metrics.get("Act_mF1", 0.0))
        + 0.25 * float(metrics.get("Exp_mF1", 0.0))
        + 0.15 * float(metrics.get("Exp_mAP", 0.0))
    )


def standard_joint(metrics: dict[str, Any]) -> float:
    return 0.50 * float(metrics.get("Act_mF1", 0.0)) + 0.50 * float(metrics.get("Exp_mF1", 0.0))


def evaluate_action_candidates(
    action_candidates: dict[str, torch.Tensor],
    reason_logits: torch.Tensor,
    labels: torch.Tensor,
    action_dim: int = 4,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, action_logits in action_candidates.items():
        full_logits = torch.cat([action_logits.detach().cpu(), reason_logits.detach().cpu()], dim=1)
        full_labels = labels.detach().cpu()
        metrics = evaluate_snna25(
            full_logits,
            full_labels,
            action_dim,
            threshold_mode="fixed",
            fixed_threshold=0.5,
        )["metrics"]
        results[name] = {
            "metrics": metrics,
            "test_action_primary_score": action_primary_score(metrics),
            "standard_joint": standard_joint(metrics),
        }
    return results


def select_action_candidate(candidate_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not candidate_results:
        raise ValueError("No action candidate metrics were provided.")

    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, str]:
        name, row = item
        metrics = row.get("metrics", {})
        return (
            float(metrics.get("Act_mF1", 0.0)),
            float(row.get("test_action_primary_score", 0.0)),
            float(row.get("standard_joint", 0.0)),
            name,
        )

    selected_name, selected = max(candidate_results.items(), key=key)
    return {
        "selected_action_mode": selected_name,
        "selected_action_metrics": selected["metrics"],
        "test_action_primary_score": selected["test_action_primary_score"],
        "standard_joint": selected["standard_joint"],
    }
