from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from fate_oia.metrics import multilabel_metrics_from_logits


def task_rows(base, candidate, target, thresholds, prefix, scales):
    sign = 2.0 * target.float() - 1.0
    base_prediction = base.sigmoid() >= thresholds[None]
    base_correct = base_prediction == target.bool()
    rows = []
    for scale in scales:
        final = base + float(scale) * candidate
        metrics = multilabel_metrics_from_logits(
            final, target.float(), threshold=thresholds, prefix=prefix,
        )
        final_correct = (final.sigmoid() >= thresholds[None]) == target.bool()
        recovered = (~base_correct & final_correct).sum()
        damaged = (base_correct & ~final_correct).sum()
        rows.append({
            "scale": float(scale),
            "mF1": float(metrics[f"{prefix}mF1"]),
            "oF1": float(metrics[f"{prefix}oF1"]),
            "mAP": float(metrics[f"{prefix}mAP"]),
            "nll_improvement": float(
                F.binary_cross_entropy_with_logits(base, target.float())
                - F.binary_cross_entropy_with_logits(final, target.float())
            ),
            "brier_improvement": float(
                (base.sigmoid() - target).square().mean()
                - (final.sigmoid() - target).square().mean()
            ),
            "signed_margin_mean": float((sign * (final - base)).mean()),
            "signed_margin_benefit_rate": float((sign * candidate * float(scale) > 0).float().mean()),
            "errors_recovered": int(recovered),
            "correct_predictions_damaged": int(damaged),
            "net_corrected_labels": int(recovered - damaged),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    epoch_dir = Path(args.epoch_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thresholds = torch.tensor(config["deployment"]["locked_image_thresholds"])
    action_target = torch.load(epoch_dir / "action_target_test.pt", weights_only=True).float()
    reason_target = torch.load(epoch_dir / "reason_target_test.pt", weights_only=True).float()
    scales = (-512, -256, -128, -64, -16, 0, 1, 16, 64, 128, 256, 512)
    result = {
        "status": "test_oracle_diagnostic_only_never_write_back",
        "action": task_rows(
            torch.load(epoch_dir / "pre_object_intent_action_test.pt", weights_only=True).float(),
            torch.load(epoch_dir / "object_intent_action_candidate_test.pt", weights_only=True).float(),
            action_target, thresholds[:4], "Act_", scales,
        ),
        "reason": task_rows(
            torch.load(epoch_dir / "pre_object_intent_reason_test.pt", weights_only=True).float(),
            torch.load(epoch_dir / "object_intent_reason_candidate_test.pt", weights_only=True).float(),
            reason_target, thresholds[4:], "Exp_", scales,
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
