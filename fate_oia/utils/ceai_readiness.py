from __future__ import annotations


def decide_r2a_readiness(pair_attention_concentration: float, q_ar_std: float, reason_branch_mf1: float | None = None, min_pair_concentration: float = 0.08, min_pair_reliability_std: float = 0.015, min_reason_branch_mf1: float = 0.30) -> dict:
    reason_ready = True if reason_branch_mf1 is None else reason_branch_mf1 >= min_reason_branch_mf1
    active = pair_attention_concentration >= min_pair_concentration and q_ar_std >= min_pair_reliability_std and reason_ready
    return {
        "r2a_active": bool(active),
        "pair_attention_concentration": float(pair_attention_concentration),
        "q_ar_std": float(q_ar_std),
        "reason_ready": bool(reason_ready),
    }
