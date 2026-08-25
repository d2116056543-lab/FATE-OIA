from __future__ import annotations

import json
from pathlib import Path

import torch


V71 = Path(r"F:\FATE_Drive_runs\tida_trajectory_v7_1_selective_state_continue")
V83 = Path(r"F:\FATE_Drive_runs\tida_v8_3_v71_firewall_expanded_epoch1_retry1\epoch_000")


def metrics(logits: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> dict:
    prediction = torch.sigmoid(logits) >= thresholds
    truth = labels > 0.5
    tp = (prediction & truth).sum(0).float()
    fp = (prediction & ~truth).sum(0).float()
    fn = (~prediction & truth).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return {"mF1": float(f1.mean()), "per_label_f1": f1.tolist()}


def flip_metrics(image_logits: torch.Tensor, video_logits: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> dict:
    image = torch.sigmoid(image_logits) >= thresholds
    video = torch.sigmoid(video_logits) >= thresholds
    truth = labels > 0.5
    image_correct = image == truth
    video_correct = video == truth
    return {
        "error_recovered": int((~image_correct & video_correct).sum()),
        "correct_damaged": int((image_correct & ~video_correct).sum()),
        "net_corrected": int((~image_correct & video_correct).sum() - (image_correct & ~video_correct).sum()),
        "recovery_rate": float((~image_correct & video_correct).sum() / (~image_correct).sum().clamp_min(1)),
        "damage_rate": float((image_correct & ~video_correct).sum() / image_correct.sum().clamp_min(1)),
    }


def main() -> None:
    v71_metrics = json.loads((V71 / "metrics_summary.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    thresholds = torch.tensor(v71_metrics["online"]["thresholds"]["video"])
    action_target = torch.load(V83 / "action_target_test.pt", map_location="cpu").float()
    reason_target = torch.load(V83 / "reason_target_test.pt", map_location="cpu").float()
    image_action = torch.load(V83 / "image_action_test.pt", map_location="cpu").float()
    video_action = torch.load(V83 / "video_action_test.pt", map_location="cpu").float()
    image_reason = torch.load(V83 / "image_reason_test.pt", map_location="cpu").float()
    video_reason = torch.load(V83 / "video_reason_test.pt", map_location="cpu").float()
    output = {
        "threshold_source": "V7.1 train-calib locked (no test writeback)",
        "action_image": metrics(image_action, action_target, thresholds[:4]),
        "action_video": metrics(video_action, action_target, thresholds[:4]),
        "reason_image": metrics(image_reason, reason_target, thresholds[4:]),
        "reason_video": metrics(video_reason, reason_target, thresholds[4:]),
        "action_temporal_flips": flip_metrics(image_action, video_action, action_target, thresholds[:4]),
        "reason_temporal_flips": flip_metrics(image_reason, video_reason, reason_target, thresholds[4:]),
        "action_branch_ablation": {},
        "reason_branch_ablation": {},
    }
    for branch in (
        "image_action", "prefix_action", "semantic_action", "traffic_action",
        "geometric_action", "video_action_base", "video_action",
    ):
        path = V83 / f"{branch}_test.pt"
        if path.is_file():
            value = torch.load(path, map_location="cpu").float()
            output["action_branch_ablation"][branch] = (
                metrics(value, action_target, thresholds[:4])
                if value.shape == action_target.shape
                else {"available": False, "shape": list(value.shape)}
            )
    for branch in (
        "image_reason", "prefix_reason", "semantic_reason", "geometric_reason", "video_reason",
    ):
        path = V83 / f"{branch}_test.pt"
        if path.is_file():
            value = torch.load(path, map_location="cpu").float()
            output["reason_branch_ablation"][branch] = (
                metrics(value, reason_target, thresholds[4:])
                if value.shape == reason_target.shape
                else {"available": False, "shape": list(value.shape)}
            )
    (V83 / "v71_locked_threshold_diagnostic.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
