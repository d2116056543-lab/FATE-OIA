from __future__ import annotations

from typing import Any

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.eagle_pu_action_set_aux import action_subset_targets
from fate_oia.engine.eagle_pu_thresholds import fixed_thresholds, global_threshold_diagnostic, per_label_threshold_diagnostic


def _subset_metrics(action_set_logits: torch.Tensor, action_targets: torch.Tensor) -> dict[str, Any]:
    subset = action_subset_targets(action_targets)
    probs = torch.softmax(action_set_logits.float(), dim=-1)
    top1 = probs.argmax(1)
    top3 = probs.topk(k=3, dim=1).indices
    card_pred = torch.tensor([int(bin(int(x)).count("1")) for x in top1.detach().cpu()], device=action_targets.device)
    card_gt = action_targets.sum(1).long()
    pred_actions = torch.zeros_like(action_targets)
    for i, code in enumerate(top1.tolist()):
        pred_actions[i] = torch.tensor([(code >> j) & 1 for j in range(action_targets.shape[1])], device=action_targets.device)
    combo_mask = action_targets.sum(1) > 1
    combo_single = ((combo_mask) & (pred_actions.sum(1) <= 1)).float().mean().item() if combo_mask.any() else 0.0
    superset = ((pred_actions >= action_targets).all(1).float().mean().item()) if action_targets.numel() else 0.0
    all_high = (pred_actions.sum(1) == action_targets.shape[1]).float().mean().item() if pred_actions.numel() else 0.0
    return {
        "subset_top1_acc": float((top1 == subset).float().mean().item()),
        "subset_top3_recall": float((top3 == subset.view(-1,1)).any(1).float().mean().item()),
        "cardinality_acc": float((card_pred == card_gt).float().mean().item()),
        "combo_gt_single_pred_rate": float(combo_single),
        "superset_pred_rate": float(superset),
        "all_high_rate": float(all_high),
    }


def evaluate_eagle_pu_tensors(outputs: dict[str, torch.Tensor], labels_action: torch.Tensor, labels_reason: torch.Tensor) -> dict[str, Any]:
    y_all = torch.cat([labels_action, labels_reason], dim=1).float()
    raw_all = torch.cat([outputs["action_logits_final_raw"], outputs["reason_logits_final_raw"]], dim=1)
    cal_all = torch.cat([outputs["action_logits_final_calibrated"], outputs["reason_logits_final_calibrated"]], dim=1)
    views = {}
    for name, logits, thr in [
        ("metrics_raw_fixed", raw_all, fixed_thresholds(raw_all.shape[1], 0.5).to(raw_all.device)),
        ("metrics_global_threshold", raw_all, global_threshold_diagnostic(raw_all, y_all)),
        ("metrics_per_label_threshold", raw_all, per_label_threshold_diagnostic(raw_all, y_all)),
        ("metrics_calibrated", cal_all, fixed_thresholds(cal_all.shape[1], 0.5).to(cal_all.device)),
    ]:
        act = multilabel_metrics_from_logits(logits[:, : labels_action.shape[1]], labels_action, threshold=thr[: labels_action.shape[1]], prefix="Act_")
        exp = multilabel_metrics_from_logits(logits[:, labels_action.shape[1] :], labels_reason, threshold=thr[labels_action.shape[1] :], prefix="Exp_")
        views[name] = {**act, **exp}
    set_metrics = _subset_metrics(outputs["action_set_logits"], labels_action)
    raw = views["metrics_raw_fixed"]
    evidence_common = float(outputs.get("evidence_audit_common_positive_rate", torch.tensor(0.0)))
    raw_act = raw.get("Act_mF1", 0.0); raw_exp = raw.get("Exp_mF1", 0.0)
    final_raw_joint = 0.40 * raw_act + 0.35 * raw_exp + 0.10 * set_metrics["subset_top1_acc"] + 0.10 * set_metrics["cardinality_acc"] + 0.05 * evidence_common
    standard_joint = 0.5 * raw_act + 0.5 * raw_exp
    return {**views, "action_set_metrics": set_metrics, "final_raw_joint": float(final_raw_joint), "standard_joint": float(standard_joint)}
