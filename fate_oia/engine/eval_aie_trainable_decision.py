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
from fate_oia.engine.train_aie_trainable_decision import evaluate, load_decision_state
from fate_oia.models.aie_trainable_decision_model import AIETrainableDecisionModel
from fate_oia.utils.aie_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--decision-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    decision = torch.load(args.decision_checkpoint, map_location="cpu", weights_only=False)
    base = build_model(config, device)
    base.load_state_dict(compatible_checkpoint_state_dict(base, source["model"]), strict=True)
    model = AIETrainableDecisionModel(
        base,
        reason_scale=float(decision["reason_scale"]),
        reason_action_scale=float(decision["reason_action_scale"]),
    ).to(device)
    load_decision_state(model, decision["decision_state"])
    dataset = make_dataset(config, "test")
    count = min(args.max_test_samples or len(dataset), len(dataset))
    loader = make_loader(
        Subset(dataset, list(range(count))),
        args.batch_size,
        False,
        args.num_workers,
        config,
        persistent_workers=False,
    )
    metrics = evaluate(model, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "decision_checkpoint": str(Path(args.decision_checkpoint).resolve()),
        "test_evaluation_count": 1,
        "metrics": metrics,
    }
    write_json(output_dir / "full_test_metrics.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
