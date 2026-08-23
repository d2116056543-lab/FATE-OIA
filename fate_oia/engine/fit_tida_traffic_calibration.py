from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.evaluate_tida_oia import collect_tida_outputs
from fate_oia.engine.train_tida_oia import build_runtime
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.tida_traffic_calibration import fit_action_traffic_calibration


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
    args = parser.parse_args()
    runtime = build_runtime(args, evaluation_only=True)
    calib = collect_tida_outputs(runtime.model, runtime.loaders["train_calib"], runtime.device)
    fitted = fit_action_traffic_calibration(
        calib["semantic_action"], calib["traffic_action_delta"], calib["action_target"]
    )
    semantic = torch.load(args.epoch_dir / "semantic_action_test.pt", map_location="cpu", weights_only=True)
    delta = torch.load(args.epoch_dir / "traffic_action_delta_test.pt", map_location="cpu", weights_only=True)
    action_target = torch.load(args.epoch_dir / "action_target_test.pt", map_location="cpu", weights_only=True)
    reason_logits = torch.load(args.epoch_dir / "video_reason_test.pt", map_location="cpu", weights_only=True)
    reason_target = torch.load(args.epoch_dir / "reason_target_test.pt", map_location="cpu", weights_only=True)
    scales = fitted["scales"].cpu()
    thresholds = fitted["thresholds"].cpu()
    calibrated_action = semantic + scales * delta
    all_thresholds = torch.cat((thresholds, torch.full((reason_logits.shape[1],), 0.5)))
    metrics = aie_branch_metrics(
        calibrated_action, reason_logits, action_target, reason_target, threshold=all_thresholds
    )
    result = {
        "source": "train_calib_only",
        "checkpoint_view": args.checkpoint_view,
        "scales": scales.tolist(),
        "action_thresholds": thresholds.tolist(),
        "train_calib_f1_by_action": fitted["calib_f1_by_action"],
        "test_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
