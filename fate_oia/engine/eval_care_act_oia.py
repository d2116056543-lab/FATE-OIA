from __future__ import annotations

from typing import Any

import torch

from fate_oia.metrics import multilabel_metrics_from_logits


def select_guarded_action_branch(candidates: dict[str, torch.Tensor], labels: torch.Tensor, threshold: float = 0.5, margin: float = 0.006) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    for name, logits in candidates.items():
        scores[name] = multilabel_metrics_from_logits(logits, labels, threshold=threshold, prefix=f"Act_{name}_")
    def mf1(name: str) -> float:
        return float(scores[name].get(f"Act_{name}_mF1", 0.0))
    selected = max(candidates, key=mf1)
    base_score = mf1("base") if "base" in candidates else 0.0
    cand_score = mf1("candidate") if "candidate" in candidates else base_score
    shutdown = cand_score < base_score - margin
    return {
        "selected_branch": selected,
        "guarded_logits": candidates[selected],
        "branch_metrics": scores,
        "shutdown_action_residual_next_epoch": bool(shutdown),
        "base_mF1": base_score,
        "candidate_mF1": cand_score,
    }
