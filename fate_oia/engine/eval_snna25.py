from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.threshold_tuning import tune_global_threshold, tune_per_label_thresholds


def _load_tensor(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix in {".pt", ".pth"}:
        return torch.load(p, map_location="cpu")
    return torch.tensor(json.loads(p.read_text(encoding="utf-8")))


def evaluate_snna25(logits: torch.Tensor, labels: torch.Tensor, action_dim: int = 4, threshold_mode: str = "global", fixed_threshold: float = 0.5) -> dict:
    if logits.shape != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    action_logits = logits[:, :action_dim]
    action_labels = labels[:, :action_dim]
    reason_logits = logits[:, action_dim:]
    reason_labels = labels[:, action_dim:]
    if threshold_mode == "fixed":
        action_threshold = fixed_threshold
        reason_threshold = fixed_threshold
        action_metrics = multilabel_metrics_from_logits(action_logits, action_labels, action_threshold, prefix="Act_")
        reason_metrics = multilabel_metrics_from_logits(reason_logits, reason_labels, reason_threshold, prefix="Exp_")
    elif threshold_mode == "per_label":
        action_threshold, action_metrics = tune_per_label_thresholds(action_logits, action_labels)
        reason_threshold, reason_metrics = tune_per_label_thresholds(reason_logits, reason_labels)
        action_metrics = {f"Act_{k}": v for k, v in action_metrics.items()}
        reason_metrics = {f"Exp_{k}": v for k, v in reason_metrics.items()}
        action_threshold = action_threshold.tolist()
        reason_threshold = reason_threshold.tolist()
    else:
        action_threshold, action_metrics = tune_global_threshold(action_logits, action_labels)
        reason_threshold, reason_metrics = tune_global_threshold(reason_logits, reason_labels)
        action_metrics = {f"Act_{k}": v for k, v in action_metrics.items()}
        reason_metrics = {f"Exp_{k}": v for k, v in reason_metrics.items()}
    return {
        "action_dim": action_dim,
        "reason_dim": int(reason_logits.shape[1]),
        "threshold_mode": threshold_mode,
        "action_threshold": action_threshold,
        "reason_threshold": reason_threshold,
        "metrics": {**action_metrics, **reason_metrics},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate BDD-OIA SNNA-25 action/reason logits with sigmoid metrics.")
    ap.add_argument("--logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--action_dim", type=int, default=4, choices=[4, 5])
    ap.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="global")
    ap.add_argument("--fixed_threshold", type=float, default=0.5)
    args = ap.parse_args()
    result = evaluate_snna25(_load_tensor(args.logits), _load_tensor(args.labels), args.action_dim, args.threshold_mode, args.fixed_threshold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()