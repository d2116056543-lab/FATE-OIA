from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.cast_action_set_energy import action_targets_to_subset_ids, build_subset_membership


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
