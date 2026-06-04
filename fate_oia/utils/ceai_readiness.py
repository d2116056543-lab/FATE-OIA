from __future__ import annotations

from typing import Any


def default_readiness_state() -> dict[str, Any]:
    return {
        "a2r_active": True,
        "pair_active": True,
        "r2a_active": False,
        "router_reason_active": True,
        "router_action_scale": 0.0,
        "reason_feedback_scale": 0.0,
        "evidence_gate_ok": False,
        "evidence_not_used_for_action": True,
        "reason_branch_mf1": 0.0,
        "base_action_mf1": 0.0,
        "final_action_mf1": 0.0,
        "pair_attention_concentration": 0.0,
        "q_ar_std": 0.0,
        "action_drop_epochs": 0,
    }


def _metric(row: dict[str, Any], branch: str, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get("branch_metrics", {}).get(branch, {}).get(key, default))
    except Exception:
        return default


def _diag(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get("diag", {}).get(key, default))
    except Exception:
        return default


def compute_trainer_readiness_state(
    previous_epoch_row: dict[str, Any] | None,
    *,
    previous_action_drop_epochs: int = 0,
    evidence_gate_ok: bool = False,
    evidence_not_used_for_action: bool = True,
    min_reason_branch_mf1_for_r2a: float = 0.30,
    min_pair_reliability_std: float = 0.015,
    min_pair_attention_concentration: float = 0.08,
    final_action_drop_tolerance: float = 0.006,
) -> dict[str, Any]:
    if previous_epoch_row is None:
        return default_readiness_state()
    reason_branch_mf1 = _metric(previous_epoch_row, "reason_specialist", "Exp_mF1")
    base_action_mf1 = _metric(previous_epoch_row, "base", "Act_mF1")
    final_action_mf1 = _metric(previous_epoch_row, "final", "Act_mF1")
    concentration = _diag(previous_epoch_row, "pair_attention_stats.pair_attention_concentration")
    q_ar_std = _diag(previous_epoch_row, "pair_reliability_stats.q_ar_std")
    dropped = final_action_mf1 < base_action_mf1 - float(final_action_drop_tolerance)
    action_drop_epochs = int(previous_action_drop_epochs) + 1 if dropped else 0
    reason_ready = reason_branch_mf1 >= float(min_reason_branch_mf1_for_r2a)
    pair_ready = concentration >= float(min_pair_attention_concentration) and q_ar_std >= float(min_pair_reliability_std)
    evidence_ok = bool(evidence_gate_ok or evidence_not_used_for_action)
    r2a_active = bool(reason_ready and pair_ready and (not dropped) and action_drop_epochs == 0 and evidence_ok)
    router_action_scale = 0.0 if not r2a_active else 1.0
    if action_drop_epochs > 0:
        router_action_scale *= 0.5
    return {
        "a2r_active": True,
        "pair_active": bool(pair_ready),
        "r2a_active": r2a_active,
        "router_reason_active": True,
        "router_action_scale": float(router_action_scale),
        "reason_feedback_scale": float(router_action_scale),
        "evidence_gate_ok": bool(evidence_gate_ok),
        "evidence_not_used_for_action": bool(evidence_not_used_for_action),
        "reason_branch_mf1": float(reason_branch_mf1),
        "base_action_mf1": float(base_action_mf1),
        "final_action_mf1": float(final_action_mf1),
        "pair_attention_concentration": float(concentration),
        "q_ar_std": float(q_ar_std),
        "action_drop_epochs": int(action_drop_epochs),
    }


def decide_r2a_readiness(
    pair_attention_concentration: float,
    q_ar_std: float,
    reason_branch_mf1: float | None = None,
    min_pair_concentration: float = 0.08,
    min_pair_reliability_std: float = 0.015,
    min_reason_branch_mf1: float = 0.30,
) -> dict:
    row = {
        "branch_metrics": {
            "base": {"Act_mF1": 0.0},
            "final": {"Act_mF1": 0.0},
            "reason_specialist": {"Exp_mF1": min_reason_branch_mf1 if reason_branch_mf1 is None else reason_branch_mf1},
        },
        "diag": {
            "pair_attention_stats.pair_attention_concentration": pair_attention_concentration,
            "pair_reliability_stats.q_ar_std": q_ar_std,
        },
    }
    return compute_trainer_readiness_state(
        row,
        min_pair_attention_concentration=min_pair_concentration,
        min_pair_reliability_std=min_pair_reliability_std,
        min_reason_branch_mf1_for_r2a=min_reason_branch_mf1,
    )
