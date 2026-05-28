from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.offline_per_label_failure_audit import run_failure_audit
from fate_oia.engine.offline_threshold_sweep import run_threshold_sweep


def joint_score(metrics: dict, *, action_key: str = "Act_mF1", reason_key: str = "Exp_mF1") -> float:
    values = metrics.get("metrics", metrics)
    return 0.5 * float(values[action_key]) + 0.5 * float(values[reason_key])


def evaluate_action_reason(
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    *,
    threshold_mode: str = "fixed",
    fixed_threshold: float = 0.5,
) -> dict:
    logits = torch.cat([action_logits.float(), reason_logits.float()], dim=1)
    labels = torch.cat([action_labels.float(), reason_labels.float()], dim=1)
    result = evaluate_snna25(logits, labels, action_dim=action_logits.shape[1], threshold_mode=threshold_mode, fixed_threshold=fixed_threshold)
    result["joint"] = joint_score(result)
    return result


def write_eval_bundle(
    *,
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    output_dir: str | Path,
    prefix: str,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixed = evaluate_action_reason(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="fixed")
    global_eval = evaluate_action_reason(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="global")
    per_label = evaluate_action_reason(action_logits, reason_logits, action_labels, reason_labels, threshold_mode="per_label")
    threshold_sweep = run_threshold_sweep(reason_logits, reason_labels, output / f"{prefix}_threshold_sweep", prefix="Exp")
    failure = run_failure_audit(logits=reason_logits, labels=reason_labels, output_dir=output / f"{prefix}_failure_audit")
    bundle = {
        "prefix": prefix,
        "fixed": fixed,
        "global": global_eval,
        "per_label": per_label,
        "reason_threshold_sweep": threshold_sweep,
        "failure_audit": {
            "top_failed_reason_indices": failure.get("top_failed_reason_indices", []),
            "num_rows": len(failure.get("rows", [])),
        },
    }
    (output / f"{prefix}_metrics_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate cached/calibrated FATE-OIA logits.")
    ap.add_argument("--action_logits", required=True)
    ap.add_argument("--reason_logits", required=True)
    ap.add_argument("--action_labels", required=True)
    ap.add_argument("--reason_labels", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", default="calibrated")
    args = ap.parse_args()
    bundle = write_eval_bundle(
        action_logits=torch.load(args.action_logits, map_location="cpu"),
        reason_logits=torch.load(args.reason_logits, map_location="cpu"),
        action_labels=torch.load(args.action_labels, map_location="cpu"),
        reason_labels=torch.load(args.reason_labels, map_location="cpu"),
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
