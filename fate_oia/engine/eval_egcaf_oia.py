from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.metrics import multilabel_metrics_from_logits


def evaluate_logits(action_core: torch.Tensor, action_final: torch.Tensor, guarded: torch.Tensor, reason: torch.Tensor, y_action: torch.Tensor, y_reason: torch.Tensor, threshold: float = 0.5) -> dict:
    core = multilabel_metrics_from_logits(action_core, y_action, threshold, prefix="Act_")
    final = multilabel_metrics_from_logits(action_final, y_action, threshold, prefix="Act_")
    guarded_m = multilabel_metrics_from_logits(guarded, y_action, threshold, prefix="Act_")
    exp = multilabel_metrics_from_logits(reason, y_reason, threshold, prefix="Exp_")
    return {
        "action_core": core,
        "action_final": final,
        "guarded_action": guarded_m,
        "reason": exp,
        "joint_core": 0.5 * core["Act_mF1"] + 0.5 * exp["Exp_mF1"],
        "joint_final": 0.5 * final["Act_mF1"] + 0.5 * exp["Exp_mF1"],
        "joint_guarded": 0.5 * guarded_m["Act_mF1"] + 0.5 * exp["Exp_mF1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    d = Path(args.logits_dir)
    result = evaluate_logits(
        torch.load(d / "action_core_test.pt", map_location="cpu"),
        torch.load(d / "action_final_test.pt", map_location="cpu"),
        torch.load(d / "guarded_action_test.pt", map_location="cpu"),
        torch.load(d / "reason_test.pt", map_location="cpu"),
        torch.load(d / "labels_action_test.pt", map_location="cpu"),
        torch.load(d / "labels_reason_test.pt", map_location="cpu"),
    )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
