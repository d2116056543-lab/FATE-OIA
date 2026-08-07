from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.item()
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_safe(value), ensure_ascii=False) + "\n")


REQUIRED_EPOCH = (
    "metrics_primary_raw.json", "metrics_final_raw.json", "metrics_primary_deploy.json",
    "metrics_final_deploy.json", "per_action_metrics.json", "per_reason_metrics.json",
    "calibration.json", "mechanism_summary.json", "predicate_mixture_stats.json",
    "atom_transport_stats.json", "contribution_stats.json", "counterfactual_certificate.json",
    "dual_constraints.json", "reason_budget_stats.json", "ecpo_stats.json", "naming_stats.json",
    "structured_coverage.json", "branch_audit_128.json", "component_diagnosis.json",
    "test_outputs.pt", "fixed_audit_outputs.pt", "checkpoint_pre_eval.pth",
)


def validate_epoch(path: str | Path) -> list[str]:
    root = Path(path)
    missing = [name for name in REQUIRED_EPOCH if not (root / name).exists()]
    for name in REQUIRED_EPOCH:
        item = root / name
        if item.exists() and item.stat().st_size == 0:
            missing.append(f"{name}:empty")
    return missing


def component_diagnosis(branches: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Report evidence-based component deltas without turning heuristics into PASS claims."""
    final = branches.get("final", {})

    def delta(branch: str, metric: str) -> float | None:
        if branch not in branches or metric not in final or metric not in branches[branch]:
            return None
        return float(final[metric] - branches[branch][metric])

    return {
        "predicate_prior": {"Act_mF1_gain_vs_off": delta("predicate_prior_off", "Act_mF1"),
                            "Exp_mF1_gain_vs_off": delta("predicate_prior_off", "Exp_mF1")},
        "local_reread": {"Act_mF1_gain_vs_off": delta("local_reread_off", "Act_mF1"),
                         "Exp_mF1_gain_vs_off": delta("local_reread_off", "Exp_mF1")},
        "atom_transport": {"Act_mF1_gain_vs_off": delta("atom_transport_off", "Act_mF1"),
                           "Exp_mF1_gain_vs_off": delta("atom_transport_off", "Exp_mF1"),
                           "token_delta_rms": diagnostics.get("transport_token_delta_rms"),
                           "map_delta_rms": diagnostics.get("transport_map_delta_rms")},
        "background_center": {"Act_mF1_gain_vs_off": delta("background_center_off", "Act_mF1")},
        "action_residual": {"Act_mF1_gain_vs_primary": delta("action_residual_off", "Act_mF1")},
        "signed_reason": {"Exp_mF1_gain_vs_unsigned": delta("reason_signed_to_unsigned_legacy", "Exp_mF1")},
        "dynamic_reason_budget": {"Exp_mF1_gain_vs_off": delta("reason_budget_off", "Exp_mF1"),
                                  "budget_mean": diagnostics.get("reason_budget_mean")},
        "reason_delta": {"Exp_mF1_gain_vs_primary": delta("reason_delta_off", "Exp_mF1"),
                         "delta_rms": diagnostics.get("reason_delta_rms")},
        "interpretation": "Positive deltas indicate association on the fixed same-field audit, not causal proof.",
    }
