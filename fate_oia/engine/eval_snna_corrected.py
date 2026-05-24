from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.threshold_tuning import tune_global_threshold, tune_per_label_thresholds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True, help=".pt tensor [N,C] or JSON list")
    ap.add_argument("--labels", required=True, help=".pt tensor [N,C] or JSON list")
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="global")
    ap.add_argument("--fixed_threshold", type=float, default=0.5)
    args = ap.parse_args()
    logits = torch.load(args.logits) if args.logits.endswith((".pt", ".pth")) else torch.tensor(json.loads(Path(args.logits).read_text()))
    labels = torch.load(args.labels) if args.labels.endswith((".pt", ".pth")) else torch.tensor(json.loads(Path(args.labels).read_text()))
    if args.threshold_mode == "fixed":
        threshold = args.fixed_threshold
        metrics = multilabel_metrics_from_logits(logits, labels, threshold)
    elif args.threshold_mode == "global":
        threshold, metrics = tune_global_threshold(logits, labels)
    else:
        threshold, metrics = tune_per_label_thresholds(logits, labels)
        threshold = threshold.tolist()
    result = {"threshold_mode": args.threshold_mode, "threshold": threshold, "metrics": metrics}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
