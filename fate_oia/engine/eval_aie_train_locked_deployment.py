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
from fate_oia.models.aie_train_locked_deployment import AIETrainLockedDeployment
from fate_oia.utils.aie_artifacts import write_json
from fate_oia.utils.aie_metrics import aie_branch_metrics


@torch.no_grad()
def evaluate(model, loader, device):
    storage = {key: [] for key in ("action_raw", "reason_raw", "action_deploy", "reason_deploy", "action_target", "reason_target")}
    named_sum = 0.0
    sample_count = 0
    delta_square = torch.zeros(4, dtype=torch.float64)
    primary_square = torch.zeros(4, dtype=torch.float64)
    value_count = 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)
        for key, source in (
            ("action_raw", "action_logits_final"),
            ("reason_raw", "reason_logits_final"),
            ("action_deploy", "action_logits_deploy"),
            ("reason_deploy", "reason_logits_deploy"),
        ):
            storage[key].append(output[source].detach().cpu())
        storage["action_target"].append(batch["action"].cpu())
        storage["reason_target"].append(batch["reason"].cpu())
        batch_size = images.shape[0]
        named_sum += float(output["named_coverage"].detach().cpu()) * batch_size
        sample_count += batch_size
        delta_square += output["action_delta"].double().square().sum(0).cpu()
        primary_square += output["action_logits_primary"].double().square().sum(0).cpu()
        value_count += batch_size
    joined = {key: torch.cat(value) for key, value in storage.items()}
    raw = aie_branch_metrics(joined["action_raw"], joined["reason_raw"], joined["action_target"], joined["reason_target"])
    deploy = aie_branch_metrics(joined["action_deploy"], joined["reason_deploy"], joined["action_target"], joined["reason_target"])
    delta_ratio = torch.sqrt(delta_square / max(value_count, 1)) / torch.sqrt(primary_square / max(value_count, 1)).clamp_min(1e-12)
    return {
        "raw_fixed": raw,
        "train_locked_deploy": deploy,
        "named_coverage": named_sum / max(sample_count, 1),
        "action_delta_to_primary_rms_by_action": delta_ratio.tolist(),
        "action_delta_to_primary_rms_mean": float(delta_ratio.mean()),
        "sample_count": sample_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    base = build_model(config, device)
    base.load_state_dict(compatible_checkpoint_state_dict(base, payload["model"]), strict=True)
    deployment = AIETrainLockedDeployment(
        base,
        payload["deployment"]["action_scales"],
        payload["deployment"]["threshold_prob"],
        reason_action_scale=float(payload["deployment"].get("reason_action_scale", 0.0)),
        reason_scale=float(payload["deployment"].get("reason_scale", 1.0)),
    ).to(device)
    dataset = make_dataset(config, "test")
    sample_count = min(args.max_test_samples or len(dataset), len(dataset))
    loader = make_loader(
        Subset(dataset, list(range(sample_count))),
        args.batch_size or int(config["training"]["batch_size"]),
        False,
        args.num_workers,
        config,
        persistent_workers=False,
    )
    result = evaluate(deployment, loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_locked_deployment_metrics.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
