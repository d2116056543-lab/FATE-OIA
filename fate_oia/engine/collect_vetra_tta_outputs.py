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
    for key in ("action_primary", "action_final"):
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


def collect(model, dataset, cfg, args, device, *, flipped: bool, action_scale: float, reason_scale: float):
    source = HorizontalFlipDataset(dataset) if flipped else dataset
    loader = make_loader(source, args.batch_size, False, args.num_workers, cfg)
    outputs, _, _ = collect_logits(
        model,
        loader,
        device,
        action_scale,
        reason_scale,
    )
    return remap_action_outputs(outputs) if flipped else outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
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
            action_scale=action_scale, reason_scale=reason_scale,
        )
        payload["flip"][split] = collect(
            model, dataset, cfg, args, device, flipped=True,
            action_scale=action_scale, reason_scale=reason_scale,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "inference_scales": {"action": action_scale, "reason": reason_scale},
        "counts": {view: {split: len(rows["action_target"]) for split, rows in splits.items()} for view, splits in payload.items()},
    }))


if __name__ == "__main__":
    main()
