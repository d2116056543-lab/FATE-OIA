from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25


def evaluate_score_logits(
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
    metrics = result["metrics"]
    result["joint"] = 0.5 * float(metrics["Act_mF1"]) + 0.5 * float(metrics["Exp_mF1"])
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ScoreV2 action/reason logits with fixed/global/per-label thresholds.")
    ap.add_argument("--action_logits", required=True)
    ap.add_argument("--reason_logits", required=True)
    ap.add_argument("--action_labels", required=True)
    ap.add_argument("--reason_labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    payload = {
        mode: evaluate_score_logits(
            torch.load(args.action_logits, map_location="cpu"),
            torch.load(args.reason_logits, map_location="cpu"),
            torch.load(args.action_labels, map_location="cpu"),
            torch.load(args.reason_labels, map_location="cpu"),
            threshold_mode=mode,
        )
        for mode in ("fixed", "global", "per_label")
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
