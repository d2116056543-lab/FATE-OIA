from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.threshold_tuning import tune_global_threshold, tune_per_label_thresholds


def run_threshold_sweep(logits: torch.Tensor, labels: torch.Tensor, output_dir: str | Path, prefix: str = "Exp") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid = torch.arange(0.05, 0.951, 0.01)
    fixed = multilabel_metrics_from_logits(logits, labels, 0.5, prefix=f"{prefix}_")
    global_threshold, global_metrics = tune_global_threshold(logits, labels, grid)
    per_label_thresholds, per_label_metrics = tune_per_label_thresholds(logits, labels, grid)
    result = {
        "fixed": {"threshold": 0.5, "metrics": fixed},
        "global": {"threshold": float(global_threshold), "metrics": {f"{prefix}_{k}": v for k, v in global_metrics.items()}},
        "per_label": {
            "thresholds": [float(x) for x in per_label_thresholds],
            "metrics": {f"{prefix}_{k}": v for k, v in per_label_metrics.items()},
        },
    }
    (output / "threshold_sweep_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "threshold_sweep_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "per_label_thresholds_test.json").write_text(
        json.dumps({"thresholds": result["per_label"]["thresholds"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline threshold sweep for saved logits.")
    ap.add_argument("--logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", default="Exp")
    args = ap.parse_args()
    result = run_threshold_sweep(
        torch.load(args.logits, map_location="cpu"),
        torch.load(args.labels, map_location="cpu"),
        args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

