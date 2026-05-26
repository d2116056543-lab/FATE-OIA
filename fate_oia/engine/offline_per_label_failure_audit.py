from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from fate_oia.metrics import binary_average_precision, multilabel_metrics_from_logits
from fate_oia.threshold_tuning import tune_per_label_thresholds


def run_failure_audit(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str | Path,
    train_labels: torch.Tensor | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thresholds, best_metrics = tune_per_label_thresholds(logits, labels, torch.arange(0.05, 0.951, 0.01))
    fixed_metrics = multilabel_metrics_from_logits(logits, labels, 0.5)
    probs = torch.sigmoid(logits.float())
    train_counts = train_labels.float().sum(0) if train_labels is not None and train_labels.numel() else labels.float().sum(0)
    rows: list[dict] = []
    for idx in range(labels.shape[1]):
        y = labels[:, idx].float()
        pred_fixed = (probs[:, idx] >= 0.5).float()
        pred_best = (probs[:, idx] >= thresholds[idx]).float()
        fn = int(((1 - pred_fixed) * y).sum().item())
        fp = int((pred_fixed * (1 - y)).sum().item())
        positives = int(y.sum().item())
        train_pos = int(train_counts[idx].item())
        group = "head" if train_pos >= 1000 else ("medium" if train_pos >= 200 else "tail")
        rows.append(
            {
                "reason_index": idx,
                "positives_train": train_pos,
                "positives_test": positives,
                "AP": binary_average_precision(probs[:, idx], y),
                "F1@0.5": fixed_metrics["per_label_f1"][idx],
                "best_F1": best_metrics["per_label_f1"][idx],
                "best_threshold": float(thresholds[idx]),
                "false_negative_count": fn,
                "false_positive_count": fp,
                "group": group,
            }
        )
    summary = {
        "rows": rows,
        "top_failed_reason_indices": [r["reason_index"] for r in sorted(rows, key=lambda r: (r["best_F1"], -r["positives_test"]))[:8]],
    }
    (output / "per_label_failure_audit_test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "top_failed_reasons.json").write_text(
        json.dumps(summary["top_failed_reason_indices"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "per_label_failure_audit_test.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["reason_index"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-label failure audit for BDD-OIA reason logits.")
    ap.add_argument("--logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_labels", default="")
    args = ap.parse_args()
    train_labels = torch.load(args.train_labels, map_location="cpu") if args.train_labels else None
    result = run_failure_audit(
        logits=torch.load(args.logits, map_location="cpu"),
        labels=torch.load(args.labels, map_location="cpu"),
        output_dir=args.output_dir,
        train_labels=train_labels,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

