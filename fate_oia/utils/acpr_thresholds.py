from __future__ import annotations

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25


def acpr_metric_views(action_logits: torch.Tensor, reason_logits: torch.Tensor, action: torch.Tensor, reason: torch.Tensor) -> dict:
    logits = torch.cat([action_logits, reason_logits], dim=-1).detach().cpu()
    labels = torch.cat([action, reason], dim=-1).detach().cpu()
    out = {}
    for mode in ["fixed", "global", "per_label"]:
        out[mode] = evaluate_snna25(logits, labels, action_dim=4, threshold_mode=mode, fixed_threshold=0.5)["metrics"]
    return {
        "metrics_raw_fixed": out["fixed"],
        "metrics_global_threshold": out["global"],
        "metrics_per_label_threshold": out["per_label"],
    }


def standard_joint(metrics: dict) -> float:
    return 0.5 * float(metrics.get("Act_mF1", 0.0)) + 0.5 * float(metrics.get("Exp_mF1", 0.0))
