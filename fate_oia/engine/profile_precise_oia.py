from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.train_precise_oia import build_optimizers, build_train_grounding_targets
from fate_oia.losses.precise_losses import total_precise_losses
from fate_oia.models.precise_oia_model import PRECISEOIAModel
from fate_oia.transforms_precise import PRECISEImageTransform
from fate_oia.utils.precise_runtime import gpu_memory_gb


def choose_runtime_profile(profiles: list[dict[str, Any]], hard_limit_gb: float) -> dict[str, Any]:
    safe = [item for item in profiles if item.get("valid") and float(item.get("peak_reserved_gb", float("inf"))) <= hard_limit_gb]
    if not safe:
        raise RuntimeError("No PRECISE runtime profile completed the complete path safely")
    fastest = max(float(item["samples_per_sec"]) for item in safe)
    near_fastest = [item for item in safe if float(item["samples_per_sec"]) >= fastest * 0.97]
    return min(near_fastest, key=lambda item: float(item["peak_reserved_gb"]))


def _make_loader(config: dict[str, Any], batch_size: int, max_samples: int) -> DataLoader:
    dataset = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "train", 4, 21, True, PRECISEImageTransform(return_mirror=False))
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


def _profile_one(config: dict[str, Any], args: argparse.Namespace, batch_size: int, accum: int, device: torch.device, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader = _make_loader(config, batch_size, args.max_train_samples)
    model = PRECISEOIAModel(Path(args.config).parent, config["pretrained_weights"]).to(device)
    optimizers = build_optimizers(model, config)
    adapter, targets = build_train_grounding_targets(loader.dataset, config, root / f"targets_b{batch_size}")
    rows: list[dict[str, Any]] = []
    measured = 0
    total_images = 0
    start = None
    valid = True
    reason = ""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step, batch in enumerate(loader):
        try:
            if step == args.warmup_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                total_images = 0
            output = model(batch["image"].to(device, non_blocking=True))
            target = adapter.stack_batch([targets[name] for name in batch["file_name"]], device)
            losses = total_precise_losses(output, batch["action"].to(device), batch["reason"].to(device), target)
            (losses["loss_total"] / accum).backward()
            if (step + 1) % accum == 0:
                for optimizer in optimizers:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step >= args.warmup_steps:
                measured += 1; total_images += int(batch["image"].shape[0])
            row = {"batch_size": batch_size, "grad_accum": accum, "step": step, "loss_total": float(losses["loss_total"].detach()), "dino_call_count": int(output["diagnostics"]["dino_call_count"]), **gpu_memory_gb(device)}
            rows.append(row)
            if measured >= args.measure_steps:
                break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            valid = False; reason = "oom"; break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6) if start is not None else float("inf")
    peak = max((float(item.get("gpu_reserved_gb", 0.0)) for item in rows), default=0.0)
    core = bool(rows) and all(item["dino_call_count"] == 1 for item in rows)
    profile = {"batch_size": batch_size, "grad_accum": accum, "workers": 0, "warmup_steps": args.warmup_steps, "measure_steps": measured, "samples_per_sec": total_images / elapsed if measured else 0.0, "peak_reserved_gb": peak, "dino_call_count": 1 if core else 0, "core_mechanisms_enabled": core, "valid": bool(valid and measured == args.measure_steps and core), "failure_reason": reason}
    return profile, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    profiles, steps = [], []
    for batch, accum in ((10, 3), (8, 4), (6, 5)):
        profile, rows = _profile_one(config, args, batch, accum, device, root)
        profiles.append(profile); steps.extend(rows)
    selected = choose_runtime_profile(profiles, float(config["runtime"]["hard_max_reserved_gb"]))
    (root / "runtime_profile.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    (root / "runtime_steps.jsonl").write_text("".join(json.dumps(item) + "\n" for item in steps), encoding="utf-8")
    (root / "selected_runtime_profile.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
