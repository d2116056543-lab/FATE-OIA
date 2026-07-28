from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.datasets.meter_dataset import METERDataset
from fate_oia.utils.meter_artifacts import write_json
from fate_oia.utils.meter_config import load_meter_config


def profile_one(model: METEROIAModel, device: torch.device, batch_size: int) -> dict[str, Any]:
    images = torch.randn(batch_size, 3, 360, 640, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        field = model.encode_images(images)
        output = model.decode_from_field(field, progress=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        reserved = 0.0
    elapsed = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "samples_per_sec": batch_size / max(elapsed, 1e-6),
        "reserved_gb": reserved,
        "allocated_gb": allocated if device.type == "cuda" else 0.0,
        "ordinary_dino_calls": model.foundation.ordinary_dino_calls,
        "finite": all(torch.isfinite(value).all().item() for value in output.values() if isinstance(value, torch.Tensor)),
        "action_shape": list(output["action_logits_final"].shape),
        "reason_shape": list(output["reason_logits_final"].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_mock_dino", action="store_true")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", True))
    profiles = []
    for batch_size in (16, 12, 8, 6):
        model = METEROIAModel(
            dim=config["model"]["dim"], action_dim=4, reason_dim=21,
            selected_layers=tuple(config["backbone"]["selected_layers"]),
            pretrained_weights=config["backbone"]["pretrained_weights"], use_mock_dino=args.use_mock_dino,
        ).to(device).eval()
        try:
            profiles.append(profile_one(model, device, batch_size))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            profiles.append({"batch_size": batch_size, "oom": True, "error": str(exc)})
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    report = {
        "real_dino": not args.use_mock_dino,
        "device": str(device),
        "profiles": profiles,
        "selected": max((p for p in profiles if not p.get("oom") and p.get("reserved_gb", 1e9) < 45.0), key=lambda p: p["samples_per_sec"], default=None),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "runtime_profile.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["selected"] is None:
        raise SystemExit("No profile under 45GB reserved memory")


if __name__ == "__main__":
    main()
