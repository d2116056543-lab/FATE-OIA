from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.cast_action_set_energy import action_targets_to_subset_ids, build_subset_membership


def _f1_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> tuple[float, float, list[float]]:
    pred = (probs >= threshold).float()
    labels = labels.float()
    tp = (pred * labels).sum(dim=0)
    fp = (pred * (1 - labels)).sum(dim=0)
    fn = ((1 - pred) * labels).sum(dim=0)
    per_label = 2 * tp / (2 * tp + fp + fn + 1e-9)
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn + 1e-9)
    return float(per_label.mean().item()), float(micro.item()), [float(x) for x in per_label.detach().cpu()]


def _reason_threshold_diagnostics(reason_logits: torch.Tensor, labels_reason: torch.Tensor) -> dict[str, Any]:
    probs = torch.sigmoid(reason_logits.float())
    labels = labels_reason.float()
    fixed_mf1, fixed_of1, _ = _f1_from_probs(probs, labels, 0.5)
    best_global = {"mF1": fixed_mf1, "oF1": fixed_of1, "threshold": 0.5}
    for idx in range(1, 100):
        threshold = idx / 100.0
        mf1, of1, _ = _f1_from_probs(probs, labels, threshold)
        if mf1 > best_global["mF1"]:
            best_global = {"mF1": mf1, "oF1": of1, "threshold": threshold}
    per_label_f1 = []
    per_label_threshold = []
    for label_idx in range(labels.shape[1]):
        p = probs[:, label_idx:label_idx + 1]
        y = labels[:, label_idx:label_idx + 1]
        best_f1 = 0.0
        best_t = 0.5
        for idx in range(1, 100):
            threshold = idx / 100.0
            mf1, _, _ = _f1_from_probs(p, y, threshold)
            if mf1 > best_f1:
                best_f1 = mf1
                best_t = threshold
        per_label_f1.append(best_f1)
        per_label_threshold.append(best_t)
    return {
        "Exp_mF1_fixed_0.5": fixed_mf1,
        "Exp_oF1_fixed_0.5": fixed_of1,
        "Exp_mF1_global_threshold_best": best_global["mF1"],
        "Exp_oF1_global_threshold_best": best_global["oF1"],
        "Exp_global_threshold_best": best_global["threshold"],
        "Exp_mF1_per_label_threshold_best": float(sum(per_label_f1) / max(1, len(per_label_f1))),
        "Exp_per_label_thresholds_best": per_label_threshold,
        "reason_gt_positive_rate": float(labels.mean().item()),
        "reason_pred_positive_rate@0.5": float((probs >= 0.5).float().mean().item()),
        "reason_pred_positive_rate@0.3": float((probs >= 0.3).float().mean().item()),
        "reason_pred_positive_rate@0.2": float((probs >= 0.2).float().mean().item()),
    }


def compute_action_set_metrics(action_set_probs: torch.Tensor, action_targets: torch.Tensor) -> dict[str, Any]:
    subset_ids = action_targets_to_subset_ids(action_targets)
    top1 = action_set_probs.argmax(-1)
    top3 = action_set_probs.topk(min(3, action_set_probs.shape[-1]), dim=-1).indices
    subset = build_subset_membership(action_targets.shape[1]).to(action_set_probs.device)
    pred_bits = subset[top1].float()
    gt_card = action_targets.float().sum(-1)
    pred_card = pred_bits.sum(-1)
    combo = gt_card > 1
    single_collapse = ((pred_card == 1) & combo).float().mean() if bool(combo.any()) else torch.tensor(0.0)
    superset = ((pred_bits >= action_targets.float()).all(-1) & (pred_card > gt_card)).float().mean()
    all_high = (pred_card >= 3).float().mean()
    combo_recall = {}
    names = {5: "forward+left", 9: "forward+right", 10: "stop+right"}
    for sid, name in names.items():
        mask = subset_ids == sid
        combo_recall[name] = float((top1[mask] == sid).float().mean().item()) if bool(mask.any()) else 0.0
    return {
        "subset_top1_acc": float((top1 == subset_ids).float().mean().item()),
        "subset_top3_recall": float((top3 == subset_ids.view(-1, 1)).any(-1).float().mean().item()),
        "cardinality_acc": float((pred_card == gt_card).float().mean().item()),
        "combo_gt_single_pred_rate": float(single_collapse.item()),
        "superset_pred_rate": float(superset.item()),
        "all_high_rate": float(all_high.item()),
        "combo_recall": combo_recall,
    }


def evaluate_cast_outputs(outputs: dict[str, torch.Tensor], labels_action: torch.Tensor, labels_reason: torch.Tensor) -> dict[str, Any]:
    action_metrics = multilabel_metrics_from_logits(outputs["action_logits"], labels_action, 0.5)
    reason_metrics = multilabel_metrics_from_logits(outputs["reason_logits"], labels_reason, 0.5)
    reason_diag = _reason_threshold_diagnostics(outputs["reason_logits"], labels_reason)
    aset = compute_action_set_metrics(outputs["action_set_probs"], labels_action)
    evidence_common = 0.0
    act = action_metrics["mF1"]
    exp = reason_metrics["mF1"]
    cast_joint = 0.40 * act + 0.35 * exp + 0.10 * aset["subset_top1_acc"] + 0.10 * aset["cardinality_acc"] + 0.05 * evidence_common
    return {
        "Act_mF1": act,
        "Act_oF1": action_metrics["oF1"],
        "Exp_mF1": exp,
        "Exp_oF1": reason_metrics["oF1"],
        "Exp_mAP": reason_metrics["mAP"],
        **reason_diag,
        "per_action_F1": action_metrics["per_label_f1"],
        "per_reason_F1": reason_metrics["per_label_f1"],
        "per_reason_AP": reason_metrics["per_label_ap"],
        "action_set": aset,
        "evidence": {"evidence_audit_common_positive_rate": evidence_common},
        "cast_joint_score": cast_joint,
        "standard_joint": 0.5 * act + 0.5 * exp,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
