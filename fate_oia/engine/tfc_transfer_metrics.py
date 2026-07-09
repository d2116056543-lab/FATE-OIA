from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch


def _to_float(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().float().mean().cpu())


def _entropy_from_ids(ids: torch.Tensor) -> tuple[float, float]:
    valid = ids[ids >= 0].detach().cpu().tolist()
    if not valid:
        return 0.0, 0.0
    counts = Counter(int(v) for v in valid)
    total = float(sum(counts.values()))
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    dominant_rate = max(probs)
    return float(entropy), float(dominant_rate)


def build_target_evidence_transfer_rows(
    *,
    epoch: int,
    target_type: str,
    deletion_stats: dict[str, torch.Tensor | dict[str, Any]],
    credit: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict[str, Any]]:
    """Summarize target-conditioned evidence transport from real deletion effects.

    TET uses selected-vs-matched-random deletion on the target logit.
    TES penalizes broad, non-specific effects on unrelated targets.
    CCA checks whether learned credit sign agrees with counterfactual effect.
    Internal credit magnitudes are logged, but never used as causal proof.
    """

    selected_effect = deletion_stats["selected_effect"].detach().float()
    random_effect = deletion_stats["random_effect"].detach().float()
    gap = deletion_stats["selected_vs_random_gap"].detach().float()
    credit_sign = deletion_stats["credit_sign"].detach().float()
    valid_mask = deletion_stats.get("selected_pair_mask")
    if valid_mask is None:
        valid_mask = credit_sign != 0
    valid_mask = valid_mask.detach().bool()
    selected_factor_id = deletion_stats.get("selected_factor_id")
    if selected_factor_id is None:
        selected_factor_id = torch.full_like(credit_sign, -1, dtype=torch.long)
    selected_factor_id = selected_factor_id.detach().long()
    selected_credit = deletion_stats.get("selected_credit_value")
    if selected_credit is None:
        selected_credit = torch.zeros_like(credit_sign)
    selected_credit = selected_credit.detach().float()
    selected_effect_all = deletion_stats.get("selected_effect_all_targets")
    if selected_effect_all is None:
        selected_effect_all = torch.diag_embed(selected_effect)
    selected_effect_all = selected_effect_all.detach().float()

    rows: list[dict[str, Any]] = []
    targets = labels.shape[1]
    for target_id in range(targets):
        target_valid = valid_mask[:, target_id]
        valid_pairs = int(target_valid.sum().cpu())
        if valid_pairs == 0:
            rows.append(
                {
                    "epoch": epoch,
                    "target_type": target_type,
                    "target_id": target_id,
                    "valid_pairs": 0,
                    "positive_label_count": int((labels[:, target_id] > 0.5).sum().cpu()),
                    "target_evidence_transfer_gap": 0.0,
                    "target_evidence_specificity": 0.0,
                    "credit_causality_agreement": None,
                    "gap_positive_rate": None,
                    "specificity_positive_rate": None,
                    "selected_effect_signed_mean": 0.0,
                    "random_effect_signed_mean": 0.0,
                    "unrelated_abs_effect_mean": 0.0,
                    "selected_credit_abs_mean": 0.0,
                    "selected_factor_entropy": 0.0,
                    "dominant_factor_rate": 0.0,
                    "target_without_positive_factor": True,
                }
            )
            continue

        sign = credit_sign[target_valid, target_id].sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        selected_signed = selected_effect[target_valid, target_id] * sign
        random_signed = random_effect[target_valid, target_id] * sign
        sample_gap = selected_signed - random_signed

        unrelated_mask = torch.ones(targets, dtype=torch.bool, device=selected_effect_all.device)
        unrelated_mask[target_id] = False
        unrelated_abs = selected_effect_all[target_valid, target_id][:, unrelated_mask].abs().mean(dim=1)
        specificity = selected_signed - unrelated_abs
        causal_agree = (selected_effect[target_valid, target_id].sign() == sign).float()
        factor_entropy, dominant_rate = _entropy_from_ids(selected_factor_id[target_valid, target_id])

        rows.append(
            {
                "epoch": epoch,
                "target_type": target_type,
                "target_id": target_id,
                "valid_pairs": valid_pairs,
                "positive_label_count": int((labels[:, target_id] > 0.5).sum().cpu()),
                "target_evidence_transfer_gap": _to_float(sample_gap),
                "target_evidence_specificity": _to_float(specificity),
                "credit_causality_agreement": _to_float(causal_agree),
                "gap_positive_rate": _to_float((sample_gap > 0).float()),
                "specificity_positive_rate": _to_float((specificity > 0).float()),
                "selected_effect_signed_mean": _to_float(selected_signed),
                "random_effect_signed_mean": _to_float(random_signed),
                "unrelated_abs_effect_mean": _to_float(unrelated_abs),
                "selected_credit_abs_mean": _to_float(selected_credit[target_valid, target_id].abs()),
                "selected_factor_entropy": factor_entropy,
                "dominant_factor_rate": dominant_rate,
                "target_without_positive_factor": False,
            }
        )
    return rows


def aggregate_target_evidence_transfer(rows: list[dict[str, Any]], prefix: str = "TFC") -> dict[str, float]:
    out: dict[str, float] = {}
    for target_type in ("action", "reason"):
        typed = [r for r in rows if r.get("target_type") == target_type and int(r.get("valid_pairs", 0)) > 0]
        key = f"{prefix}_{target_type}"
        if not typed:
            out[f"{key}_TET_mean"] = 0.0
            out[f"{key}_TES_mean"] = 0.0
            out[f"{key}_CCA_mean"] = 0.0
            out[f"{key}_gap_positive_rate"] = 0.0
            out[f"{key}_specificity_positive_rate"] = 0.0
            out[f"{key}_valid_target_rate"] = 0.0
            continue
        total = float(sum(int(r["valid_pairs"]) for r in typed))

        def weighted_mean(name: str) -> float:
            return float(sum(float(r[name]) * int(r["valid_pairs"]) for r in typed if r.get(name) is not None) / max(total, 1.0))

        out[f"{key}_TET_mean"] = weighted_mean("target_evidence_transfer_gap")
        out[f"{key}_TES_mean"] = weighted_mean("target_evidence_specificity")
        out[f"{key}_CCA_mean"] = weighted_mean("credit_causality_agreement")
        out[f"{key}_gap_positive_rate"] = weighted_mean("gap_positive_rate")
        out[f"{key}_specificity_positive_rate"] = weighted_mean("specificity_positive_rate")
        target_dim = 4 if target_type == "action" else 21
        valid_target_ids = {int(r["target_id"]) for r in typed}
        out[f"{key}_valid_target_rate"] = float(len(valid_target_ids) / target_dim)
    out[f"{prefix}_transfer_macro"] = 0.5 * (out[f"{prefix}_action_TET_mean"] + out[f"{prefix}_reason_TET_mean"])
    out[f"{prefix}_specificity_macro"] = 0.5 * (out[f"{prefix}_action_TES_mean"] + out[f"{prefix}_reason_TES_mean"])
    out[f"{prefix}_CCA_macro"] = 0.5 * (out[f"{prefix}_action_CCA_mean"] + out[f"{prefix}_reason_CCA_mean"])
    return out
