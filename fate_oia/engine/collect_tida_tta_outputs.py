from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.train_tida_oia import build_runtime
from fate_oia.utils.tida_artifacts import atomic_write_json


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def collect_split(model, loader, device: torch.device) -> dict[str, Any]:
    tensor_keys = (
        "action_original", "action_flip", "image_action_original", "image_action_flip",
        "reason_original", "image_reason_original", "action_target", "reason_target",
    )
    store: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_keys}
    file_names: list[str] = []
    concept_rows: list[dict[str, Any]] = []
    model.eval()
    for batch in loader:
        batch = _device_batch(batch, device)
        original = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
        )
        flipped = model(
            batch["target_image"].flip(-1), batch["context_images"].flip(-1),
            batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
            canonicalize_horizontal_flip=True,
        )
        values = {
            "action_original": original["video_action_logits"],
            "action_flip": flipped["video_action_logits"],
            "image_action_original": original["image_action_logits"],
            "image_action_flip": flipped["image_action_logits"],
            "reason_original": original["video_reason_logits"],
            "image_reason_original": original["image_reason_logits"],
            "action_target": batch["action"], "reason_target": batch["reason"],
        }
        for key, value in values.items():
            store[key].append(value.detach().float().cpu())
        for index, (name, concepts) in enumerate(zip(batch["file_name"], original["dynamic_concepts"])):
            concept_rows.append({
                "file_name": name, "dynamic_concepts": concepts,
                "innovation_reliability": original["innovation_reliability"][index].detach().float().cpu().tolist(),
                "action_temporal_route": original["action_route"][index].detach().float().cpu().tolist(),
                "action_factor_contribution": original["action_factor_contribution"][index].detach().float().cpu().tolist(),
                "predicate_velocity_norm": original["predicate_velocity_norm"][index].detach().float().cpu().tolist(),
                "predicate_region_mass_velocity": original["predicate_region_mass_velocity"][index].detach().float().cpu().tolist(),
                "image_action_logits": original["image_action_logits"][index].detach().float().cpu().tolist(),
                "video_action_logits": original["video_action_logits"][index].detach().float().cpu().tolist(),
                "image_reason_logits": original["image_reason_logits"][index].detach().float().cpu().tolist(),
                "video_reason_logits": original["video_reason_logits"][index].detach().float().cpu().tolist(),
            })
        file_names.extend(batch["file_name"])
    return {key: torch.cat(values) for key, values in store.items()} | {
        "file_names": file_names, "dynamic_concepts": concept_rows,
    }


def save_split(output_dir: Path, split: str, rows: dict[str, Any]) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, value in rows.items():
        if torch.is_tensor(value):
            torch.save(value, split_dir / f"{key}.pt")
    atomic_write_json(split_dir / "file_names.json", rows["file_names"])
    with (split_dir / "dynamic_explanation_examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows["dynamic_concepts"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-chunk-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    runtime = build_runtime(args, evaluation_only=True)
    output_dir = Path(args.output_dir)
    for split in ("train_calib", "train_audit", "test"):
        save_split(output_dir, split, collect_split(runtime.model, runtime.loaders[split], runtime.device))
    atomic_write_json(output_dir / "tta_collection_manifest.json", {
        "pass": True, "parameter_fit_splits": ["train_calib", "train_audit"],
        "evaluation_split": "test", "test_labels_used_for_parameter_fit": False,
        "reason_tta": "original_only", "horizontal_flip_canonicalized": True,
    })


if __name__ == "__main__":
    main()
