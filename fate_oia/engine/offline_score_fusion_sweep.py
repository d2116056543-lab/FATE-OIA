from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.eval_score_calibrated import evaluate_score_logits


def run_score_fusion_sweep(
    *,
    action_logits_a: torch.Tensor,
    reason_logits_a: torch.Tensor,
    action_logits_b: torch.Tensor,
    reason_logits_b: torch.Tensor,
    action_labels: torch.Tensor,
    reason_labels: torch.Tensor,
    output_dir: str | Path,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in [x / 10.0 for x in range(11)]:
        action = alpha * action_logits_a + (1.0 - alpha) * action_logits_b
        reason = alpha * reason_logits_a + (1.0 - alpha) * reason_logits_b
        result = evaluate_score_logits(action, reason, action_labels, reason_labels, threshold_mode="fixed")
        rows.append({"alpha_a": alpha, "joint": result["joint"], **result["metrics"]})
    best = max(rows, key=lambda row: (row["joint"], row["Exp_mF1"]))
    payload = {"rows": rows, "best": best}
    (output / "score_fusion_sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline fixed-alpha sweep between two score branches.")
    ap.add_argument("--action_logits_a", required=True)
    ap.add_argument("--reason_logits_a", required=True)
    ap.add_argument("--action_logits_b", required=True)
    ap.add_argument("--reason_logits_b", required=True)
    ap.add_argument("--action_labels", required=True)
    ap.add_argument("--reason_labels", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    result = run_score_fusion_sweep(
        action_logits_a=torch.load(args.action_logits_a, map_location="cpu"),
        reason_logits_a=torch.load(args.reason_logits_a, map_location="cpu"),
        action_logits_b=torch.load(args.action_logits_b, map_location="cpu"),
        reason_logits_b=torch.load(args.reason_logits_b, map_location="cpu"),
        action_labels=torch.load(args.action_labels, map_location="cpu"),
        reason_labels=torch.load(args.reason_labels, map_location="cpu"),
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
