from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from fate_oia.engine.train_aie_oia import (
    build_model,
    canonical_model_state_dict,
    collect_logits,
    load_config,
    make_dataset,
    make_loader,
)
from fate_oia.engine.train_vetra_strong_refine import build_refiner, run_refiner
from fate_oia.utils.vetra_stage_contracts import sha256_file, validate_stage_checkpoint


class HorizontalFlipDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = dict(self.dataset[index])
        row["image"] = torch.flip(row["image"], dims=(-1,))
        return row


def remap_action_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for key in ("action_primary", "action_final", "action_delta"):
        if key not in outputs:
            continue
        value = outputs[key].clone()
        value[:, 2], value[:, 3] = outputs[key][:, 3], outputs[key][:, 2]
        outputs[key] = value
    return outputs


def checkpoint_inference_scales(checkpoint, cfg) -> tuple[float, float]:
    scales = checkpoint.get("inference_scales") or {}
    return (
        float(scales.get("action", 1.0)),
        float(scales.get("reason", cfg["reason_private"]["reason_scale_max"])),
    )


def apply_optional_refiner(source, refiner, *, action_scale: float):
    if refiner is None:
        return {
            "action_logits_final": source["action_logits_final"],
            "reason_logits_final": source["reason_logits_final"],
            "action_delta": torch.zeros_like(source["action_logits_final"]),
        }
    refined = run_refiner(refiner, source, action_scale=action_scale)
    if not torch.equal(
        refined["reason_logits_final"].detach().cpu(),
        source["reason_logits_final"].detach().cpu(),
    ):
        raise RuntimeError("Stage B changed reason logits during collection")
    return refined


@torch.no_grad()
def collect_with_refiner(model, refiner, loader, device, action_scale, reason_scale):
    store = {
        key: []
        for key in (
            "action_primary", "action_final", "reason_primary", "reason_final",
            "action_target", "reason_target", "action_delta",
        )
    }
    names = []
    model.eval(); refiner.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            field = model.encode_images(images)
            source = model.decode_from_field(
                field, action_scale=action_scale, reason_scale=reason_scale
            )
        refined = apply_optional_refiner(source, refiner, action_scale=action_scale)
        rows = {
            "action_primary": source["action_logits_primary"],
            "action_final": refined["action_logits_final"],
            "reason_primary": source["reason_logits_primary"],
            "reason_final": refined["reason_logits_final"],
            "action_target": batch["action"],
            "reason_target": batch["reason"],
            "action_delta": refined["action_delta"],
        }
        for key, value in rows.items():
            store[key].append(value.detach().float().cpu())
        names.extend(batch["file_name"])
    return {key: torch.cat(values) for key, values in store.items()} | {
        "file_name": names
    }


def collect(model, dataset, cfg, args, device, *, flipped: bool, action_scale: float, reason_scale: float, refiner=None):
    source = HorizontalFlipDataset(dataset) if flipped else dataset
    loader = make_loader(source, args.batch_size, False, args.num_workers, cfg)
    if refiner is None:
        outputs, names, _ = collect_logits(
            model, loader, device, action_scale, reason_scale,
        )
        outputs["action_delta"] = torch.zeros_like(outputs["action_final"])
        outputs["file_name"] = names
    else:
        outputs = collect_with_refiner(
            model, refiner, loader, device, action_scale, reason_scale
        )
    return remap_action_outputs(outputs) if flipped else outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage-b-checkpoint")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
    action_scale, reason_scale = checkpoint_inference_scales(checkpoint, cfg)
    refiner = None
    stage_b_metadata = {"provided": False, "refiner_selected": False}
    if args.stage_b_checkpoint:
        identity = checkpoint.get("run_identity")
        if not identity:
            raise RuntimeError("Stage A checkpoint is missing run identity")
        stage_b = validate_stage_checkpoint(
            args.stage_b_checkpoint,
            identity,
            expected_stage="action_refined",
            expected_parent_sha256=sha256_file(args.checkpoint),
        )
        stage_b_metadata = {
            "provided": True,
            "checkpoint": str(Path(args.stage_b_checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.stage_b_checkpoint),
            "refiner_selected": bool(stage_b["refiner_selected"]),
            "deployment_gain": stage_b["deployment_gain"].tolist(),
        }
        if stage_b["refiner_selected"]:
            refiner = build_refiner(model, cfg).to(device)
            refiner.load_state_dict(stage_b["refiner"], strict=True)
            refiner.set_deployment_gain(stage_b["deployment_gain"].to(device))
            refiner.eval()

    train = make_dataset(cfg, "train")
    by_name = {sample.file_name: index for index, sample in enumerate(train.samples)}
    datasets = {}
    for split in ("train_calib", "train_audit"):
        names = json.loads(Path(args.run_root, f"{split}_ids.json").read_text(encoding="utf-8"))
        datasets[split] = Subset(train, [by_name[name] for name in names])
    datasets["test"] = make_dataset(cfg, "test")
    if args.max_samples_per_split is not None:
        datasets = {
            split: Subset(dataset, range(min(len(dataset), args.max_samples_per_split)))
            for split, dataset in datasets.items()
        }

    payload = {"original": {}, "flip": {}}
    for split, dataset in datasets.items():
        payload["original"][split] = collect(
            model, dataset, cfg, args, device, flipped=False,
            action_scale=action_scale, reason_scale=reason_scale, refiner=refiner,
        )
        payload["flip"][split] = collect(
            model, dataset, cfg, args, device, flipped=True,
            action_scale=action_scale, reason_scale=reason_scale, refiner=refiner,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["metadata"] = {
        "stage_a_checkpoint": str(Path(args.checkpoint).resolve()),
        "stage_a_checkpoint_sha256": sha256_file(args.checkpoint),
        "stage_b": stage_b_metadata,
        "reason_identity": True,
    }
    torch.save(payload, output)
    print(json.dumps({
        "inference_scales": {"action": action_scale, "reason": reason_scale},
        "counts": {
            view: {
                split: len(rows["action_target"])
                for split, rows in payload[view].items()
            }
            for view in ("original", "flip")
        },
        "stage_b": stage_b_metadata,
    }))


if __name__ == "__main__":
    main()
