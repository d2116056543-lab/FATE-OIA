from __future__ import annotations

import argparse
from pathlib import Path

import torch


def f1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.bool()
    target = target.bool()
    tp = (prediction & target).sum(0).float()
    fp = (prediction & ~target).sum(0).float()
    fn = (~prediction & target).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.artifacts:
        value = torch.load(path, map_location="cpu")
        target = value["action_target"]
        print(path)
        for key in (
            "base_prediction", "full_prediction", "margin_prediction",
            "traffic_shuffle_prediction",
        ):
            if key in value:
                scores = f1(value[key], target)
                print(key, scores.tolist(), float(scores.mean()))


if __name__ == "__main__":
    main()
