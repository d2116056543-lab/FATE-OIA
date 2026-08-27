from __future__ import annotations

import argparse
import json

import torch


def macro_f1(logits: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> float:
    prediction = logits.sigmoid() >= threshold
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    rows = torch.load(args.artifact, map_location="cpu")
    base = rows["base_action_logits"]
    delta = rows["boundary_delta"]
    target = rows["action_target"]
    threshold = rows["action_thresholds"]
    scales = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
    print(json.dumps({str(scale): macro_f1(
        base - scale * delta, target, threshold
    ) for scale in scales}))


if __name__ == "__main__":
    main()
