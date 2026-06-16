from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.utils.acpr_pair_mining import action_vectors_to_subset_id
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def _action_composition_metrics(action_logits: torch.Tensor, action: torch.Tensor) -> dict:
    pred = (torch.sigmoid(action_logits) >= 0.5).float()
    gt_card = action.sum(-1)
    pred_card = pred.sum(-1)
    combo_mask = gt_card > 1
    combo_gt_single_pred_rate = ((combo_mask) & (pred_card <= 1)).float().mean().item() if combo_mask.any() else 0.0
    superset_pred_rate = ((pred >= action).all(-1).float().mean().item()) if action.numel() else 0.0
    all_high_rate = (pred_card >= 4).float().mean().item() if pred_card.numel() else 0.0
    return {
        "action_composition_combo_gt_single_pred_rate": combo_gt_single_pred_rate,
        "action_composition_superset_pred_rate": superset_pred_rate,
        "action_composition_all_high_rate": all_high_rate,
        "action_composition_subset_targets": action_vectors_to_subset_id(action).tolist(),
    }


def _tail_reason_metrics(reason_logits: torch.Tensor, reason: torch.Tensor, tail_indices: list[int] | None = None) -> dict:
    tail_indices = tail_indices or [12, 9, 5, 14, 6, 11, 10, 13]
    views = acpr_metric_views(torch.zeros(reason_logits.shape[0], 4), reason_logits, torch.zeros(reason.shape[0], 4), reason)
    per_reason = views["metrics_raw_fixed"].get("per_reason_F1", [])
    vals = [float(per_reason[i]) for i in tail_indices if isinstance(per_reason, list) and i < len(per_reason)]
    return {"tail_reason_indices": tail_indices, "tail_reason_mF1": sum(vals) / max(len(vals), 1)}


def evaluate_tensors(
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    action: torch.Tensor,
    reason: torch.Tensor,
    predicate_logits: torch.Tensor | None = None,
    pair_margins: dict | None = None,
    branch: str = "deploy_fixed",
    base_action_logits: torch.Tensor | None = None,
    base_reason_logits: torch.Tensor | None = None,
    calibrated_action_logits: torch.Tensor | None = None,
    calibrated_reason_logits: torch.Tensor | None = None,
    action_candidate_logits: dict[str, torch.Tensor] | None = None,
) -> dict:
    views = acpr_metric_views(action_logits, reason_logits, action, reason)
    base_views = acpr_metric_views(base_action_logits, base_reason_logits, action, reason) if base_action_logits is not None and base_reason_logits is not None else None
    cal_views = acpr_metric_views(calibrated_action_logits, calibrated_reason_logits, action, reason) if calibrated_action_logits is not None and calibrated_reason_logits is not None else None
    raw = views["metrics_raw_fixed"]
    predicate_group_metrics = {"predicate_group_available": predicate_logits is not None}
    if predicate_logits is not None:
        predicate_group_metrics["predicate_group_positive_rate"] = float((torch.sigmoid(predicate_logits) >= 0.5).float().mean())
    pair_margin_by_reason = pair_margins or {"pair_margin_available": False}
    candidate_metrics = {}
    for name, cand_action in (action_candidate_logits or {}).items():
        candidate_metrics[name] = acpr_metric_views(cand_action, reason_logits, action, reason)["metrics_raw_fixed"]
    return {
        **views,
        "primary_branch": branch,
        "metrics_deploy_fixed": views["metrics_raw_fixed"],
        "metrics_base_fixed": base_views["metrics_raw_fixed"] if base_views else views["metrics_raw_fixed"],
        "metrics_calibrated": cal_views["metrics_raw_fixed"] if cal_views else views["metrics_raw_fixed"],
        "metrics_test_oracle_global_threshold": views["metrics_global_threshold"],
        "metrics_test_oracle_per_label_threshold": views["metrics_per_label_threshold"],
        "final_raw_joint": standard_joint(raw),
        "base_fixed_joint": standard_joint(base_views["metrics_raw_fixed"]) if base_views else standard_joint(raw),
        "final_calibrated_joint": standard_joint(cal_views["metrics_raw_fixed"]) if cal_views else standard_joint(raw),
        "action_composition_metrics": _action_composition_metrics(action_logits, action),
        "tail_reason_metrics": _tail_reason_metrics(reason_logits, reason),
        "predicate_group_metrics": predicate_group_metrics,
        "pair_margin_by_reason": pair_margin_by_reason,
        "metrics_action_candidates_fixed": candidate_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_action", required=True)
    ap.add_argument("--logits_reason", required=True)
    ap.add_argument("--labels_action", required=True)
    ap.add_argument("--labels_reason", required=True)
    ap.add_argument("--predicate_logits", default=None)
    ap.add_argument("--branch", default="deploy_fixed", choices=["base_fixed", "deploy_fixed", "calibrated", "final_raw", "final_calibrated"])
    ap.add_argument("--logits_action_base", default=None)
    ap.add_argument("--logits_reason_base", default=None)
    ap.add_argument("--logits_action_calibrated", default=None)
    ap.add_argument("--logits_reason_calibrated", default=None)
    ap.add_argument("--logits_action_candidate", action="append", default=[], help="Candidate action logits as name=path.")
    ap.add_argument("--evaluate_action_candidates", action="store_true")
    ap.add_argument("--candidate_gate_json", default=None)
    ap.add_argument("--output_action_candidate_metrics", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    candidates = {}
    for item in args.logits_action_candidate:
        if "=" not in item:
            raise ValueError("--logits_action_candidate must be name=path")
        name, path = item.split("=", 1)
        candidates[name] = torch.load(path, map_location="cpu")
    result = evaluate_tensors(
        torch.load(args.logits_action, map_location="cpu"),
        torch.load(args.logits_reason, map_location="cpu"),
        torch.load(args.labels_action, map_location="cpu"),
        torch.load(args.labels_reason, map_location="cpu"),
        torch.load(args.predicate_logits, map_location="cpu") if args.predicate_logits else None,
        branch=args.branch,
        base_action_logits=torch.load(args.logits_action_base, map_location="cpu") if args.logits_action_base else None,
        base_reason_logits=torch.load(args.logits_reason_base, map_location="cpu") if args.logits_reason_base else None,
        calibrated_action_logits=torch.load(args.logits_action_calibrated, map_location="cpu") if args.logits_action_calibrated else None,
        calibrated_reason_logits=torch.load(args.logits_reason_calibrated, map_location="cpu") if args.logits_reason_calibrated else None,
        action_candidate_logits=candidates if args.evaluate_action_candidates or candidates else None,
    )
    if args.candidate_gate_json:
        result["candidate_gate_json"] = json.loads(Path(args.candidate_gate_json).read_text(encoding="utf-8"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_action_candidate_metrics:
        Path(args.output_action_candidate_metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_action_candidate_metrics).write_text(json.dumps(result.get("metrics_action_candidates_fixed", {}), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
