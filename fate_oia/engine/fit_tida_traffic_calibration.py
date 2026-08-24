from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.evaluate_tida_oia import collect_tida_outputs
from fate_oia.engine.train_tida_oia import build_runtime
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.tida_traffic_calibration import fit_action_traffic_calibration
from fate_oia.utils.tida_traffic_calibration import (
    apply_action_traffic_calibration,
    apply_action_traffic_utility,
    fit_action_traffic_calibration_oof,
    fit_action_traffic_utility_oof,
)


def _flip_counts(base, deployed, target, thresholds):
    base_pred = torch.sigmoid(base) >= thresholds.view(1, -1)
    deployed_pred = torch.sigmoid(deployed) >= thresholds.view(1, -1)
    positive = target > 0.5
    return {
        "fn_to_tp": ((~base_pred) & deployed_pred & positive).sum(0).tolist(),
        "fp_to_tn": (base_pred & (~deployed_pred) & (~positive)).sum(0).tolist(),
        "tp_to_fn": (base_pred & (~deployed_pred) & positive).sum(0).tolist(),
        "tn_to_fp": ((~base_pred) & deployed_pred & (~positive)).sum(0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", choices=("online", "ema"), default="online")
    parser.add_argument("--epoch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--context-chunk-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--branch", choices=("trajectory", "legacy"), default="trajectory")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    runtime = build_runtime(args, evaluation_only=True)
    calib = collect_tida_outputs(runtime.model, runtime.loaders["train_calib"], runtime.device)
    delta_key = "traffic_trajectory_delta" if args.branch == "trajectory" else "traffic_action_delta"
    if args.branch == "trajectory":
        fitted = fit_action_traffic_utility_oof(
            calib["semantic_action"],
            calib["traffic_trajectory_candidate_delta"],
            calib["traffic_trajectory_utility_gate"],
            calib["action_target"],
            folds=args.folds,
        )
    else:
        fitted = fit_action_traffic_calibration(
            calib["semantic_action"], calib[delta_key], calib["action_target"]
        )
    semantic = torch.load(args.epoch_dir / "semantic_action_test.pt", map_location="cpu", weights_only=True)
    delta = torch.load(args.epoch_dir / f"{delta_key}_test.pt", map_location="cpu", weights_only=True)
    action_target = torch.load(args.epoch_dir / "action_target_test.pt", map_location="cpu", weights_only=True)
    reason_logits = torch.load(args.epoch_dir / "video_reason_test.pt", map_location="cpu", weights_only=True)
    reason_target = torch.load(args.epoch_dir / "reason_target_test.pt", map_location="cpu", weights_only=True)
    scales = fitted["scales"].cpu()
    thresholds = fitted["thresholds"].cpu()
    cutoffs = fitted.get("cutoffs")
    if args.branch == "trajectory" and cutoffs is not None:
        candidate_delta = torch.load(
            args.epoch_dir / "traffic_trajectory_candidate_delta_test.pt",
            map_location="cpu", weights_only=True,
        )
        utility_gate = torch.load(
            args.epoch_dir / "traffic_trajectory_utility_gate_test.pt",
            map_location="cpu", weights_only=True,
        )
        calibrated_action = apply_action_traffic_utility(
            semantic, candidate_delta, utility_gate, scales, cutoffs.cpu()
        )
    else:
        candidate_delta = delta
        utility_gate = None
        calibrated_action = apply_action_traffic_calibration(semantic, delta, scales)
    calibration_path = args.epoch_dir / "calibration.json"
    if calibration_path.exists():
        reason_thresholds = torch.tensor(
            json.loads(calibration_path.read_text(encoding="utf-8"))["video"][4:]
        )
    else:
        reason_thresholds = torch.full((reason_logits.shape[1],), 0.5)
    all_thresholds = torch.cat((thresholds, reason_thresholds))
    metrics = aie_branch_metrics(
        calibrated_action, reason_logits, action_target, reason_target, threshold=all_thresholds
    )
    base_metrics = aie_branch_metrics(
        semantic, reason_logits, action_target, reason_target, threshold=all_thresholds
    )
    locked_curve = []
    for candidate in fitted.get("candidates", [1.0]):
        if isinstance(candidate, dict):
            scale = float(candidate["scale"])
            cutoff = float(candidate["cutoff"])
            curve_action = semantic + scale * candidate_delta * (utility_gate >= cutoff)
        else:
            scale = float(candidate)
            cutoff = None
            curve_action = semantic + scale * delta
        curve_metrics = aie_branch_metrics(
            curve_action, reason_logits, action_target, reason_target, threshold=all_thresholds
        )
        locked_curve.append({
            "scale": scale,
            "utility_cutoff": cutoff,
            "Act_mF1": curve_metrics["Act_mF1"],
            "Act_oF1": curve_metrics["Act_oF1"],
            "Act_mAP": curve_metrics["Act_mAP"],
        })
    result = {
        "source": "train_calib_only",
        "selection": "deterministic_train_calib_oof" if args.branch == "trajectory" else "train_calib_fit",
        "branch": args.branch,
        "checkpoint_view": args.checkpoint_view,
        "scales": scales.tolist(),
        "utility_cutoffs": None if cutoffs is None else cutoffs.tolist(),
        "action_thresholds": thresholds.tolist(),
        "train_calib_f1_by_action": fitted["calib_f1_by_action"],
        "oof_gain_by_action": fitted.get("oof_gain_by_action", torch.zeros_like(scales)).tolist(),
        "oof_scores": fitted.get("oof_scores", torch.empty(0)).tolist(),
        "base_test_metrics_at_locked_thresholds": base_metrics,
        "test_metrics": metrics,
        "test_action_mf1_gain": metrics["Act_mF1"] - base_metrics["Act_mF1"],
        "decision_flips": _flip_counts(semantic, calibrated_action, action_target, thresholds),
        "test_locked_threshold_scale_curve_diagnostic_only": locked_curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
