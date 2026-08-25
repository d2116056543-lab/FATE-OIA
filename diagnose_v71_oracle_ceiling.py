from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(r"F:\FATE_Drive_runs\tida_trajectory_v7_1_selective_state_continue\epoch_000")


def best_thresholds(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(0.001, 0.999, 999)
    values, thresholds = [], []
    for index in range(logits.shape[1]):
        prediction = torch.sigmoid(logits[:, index])[:, None] >= grid[None]
        truth = labels[:, index, None] > 0.5
        tp = (prediction & truth).sum(0).float()
        fp = (prediction & ~truth).sum(0).float()
        fn = (~prediction & truth).sum(0).float()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        best = f1.argmax()
        values.append(f1[best])
        thresholds.append(grid[best])
    return torch.stack(values), torch.stack(thresholds)


def main() -> None:
    action_target = torch.load(ROOT / "action_target_test.pt", map_location="cpu").float()
    reason_target = torch.load(ROOT / "reason_target_test.pt", map_location="cpu").float()
    result = {}
    for branch, target in (("image_action", action_target), ("video_action", action_target), ("image_reason", reason_target), ("video_reason", reason_target)):
        logits = torch.load(ROOT / f"{branch}_test.pt", map_location="cpu").float()
        f1, threshold = best_thresholds(logits, target)
        result[branch] = {
            "oracle_mf1": float(f1.mean()),
            "per_label_f1": f1.tolist(),
            "thresholds": threshold.tolist(),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
