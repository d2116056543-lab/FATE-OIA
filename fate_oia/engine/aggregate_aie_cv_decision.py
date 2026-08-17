from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import (
    build_model,
    compatible_checkpoint_state_dict,
    load_config,
    make_dataset,
    make_loader,
)
from fate_oia.engine.train_aie_trainable_decision import (
    decision_state,
    evaluate,
    load_decision_state,
)
from fate_oia.models.aie_trainable_decision_model import AIETrainableDecisionModel
from fate_oia.utils.aie_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--fold-checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    base = build_model(cfg, device)
    base.load_state_dict(compatible_checkpoint_state_dict(base, source["model"]), strict=True)
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.fold_checkpoints]
    model = AIETrainableDecisionModel(
        base,
        reason_scale=float(payloads[0]["reason_scale"]),
        reason_action_scale=float(payloads[0]["reason_action_scale"]),
    ).to(device)
    action_scales = []
    threshold_probabilities = []
    for payload in payloads:
        load_decision_state(model, payload["decision_state"])
        action_scales.append(model.action_scales.detach().cpu())
        threshold_probabilities.append(model.threshold_prob.detach().cpu())
    median_action_scale = torch.stack(action_scales).median(0).values.to(device)
    median_threshold = torch.stack(threshold_probabilities).median(0).values.to(device)
    with torch.no_grad():
        model.action_scale_raw.copy_(torch.logit(median_action_scale.clamp(1e-6, 1.0 - 1e-6)))
        normalized = (
            (median_threshold - model.threshold_lower)
            / (model.threshold_upper - model.threshold_lower)
        ).clamp(1e-6, 1.0 - 1e-6)
        model.threshold_raw.copy_(torch.logit(normalized))

    dataset = make_dataset(cfg, "test")
    loader = make_loader(
        Subset(dataset, list(range(len(dataset)))),
        args.batch_size,
        False,
        args.num_workers,
        cfg,
        persistent_workers=False,
    )
    metrics = evaluate(model, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "aggregation": "coordinatewise_median_of_five_trained_fold_heads",
        "test_evaluation_count": 1,
        "fold_checkpoints": [str(Path(path).resolve()) for path in args.fold_checkpoints],
        "metrics": metrics,
    }
    checkpoint = {
        "decision_state": decision_state(model),
        "reason_scale": model.reason_scale,
        "reason_action_scale": model.reason_action_scale,
        "aggregation": result["aggregation"],
        "fold_checkpoints": result["fold_checkpoints"],
        "final_test_metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "checkpoint_cv_median_trained_decision.pth")
    write_json(output_dir / "cv_median_full_test_metrics.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
