from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


def evaluate_tensors(action_logits: torch.Tensor, reason_logits: torch.Tensor, action: torch.Tensor, reason: torch.Tensor) -> dict:
    views = acpr_metric_views(action_logits, reason_logits, action, reason)
    raw = views["metrics_raw_fixed"]
    return {**views, "final_raw_joint": standard_joint(raw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_action", required=True)
    ap.add_argument("--logits_reason", required=True)
    ap.add_argument("--labels_action", required=True)
    ap.add_argument("--labels_reason", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = evaluate_tensors(
        torch.load(args.logits_action, map_location="cpu"),
        torch.load(args.logits_reason, map_location="cpu"),
        torch.load(args.labels_action, map_location="cpu"),
        torch.load(args.labels_reason, map_location="cpu"),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
