from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.utils.meter_artifacts import write_json
from fate_oia.utils.meter_config import load_meter_config


DEFAULT_CANDIDATES = ((6, 5), (5, 6), (4, 8), (3, 10), (2, 15))


def build_two_stage_profile_plan(
    candidates: tuple[tuple[int, int], ...] = DEFAULT_CANDIDATES,
) -> dict[str, Any]:
    return {
        "stage_1": "mock_or_single_batch_contract",
        "stage_2": "real_dino_forward_backward_throughput",
        "candidates": [list(item) for item in candidates],
        "selection": "fastest_stable_below_reserved_limit",
    }


def _profile_candidate(
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    *,
    use_mock_dino: bool,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = METEROIAModel(
        dim=int(config["model"]["dim"]),
        action_dim=4,
        reason_dim=21,
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=use_mock_dino,
        factor_rank=int(config["model"].get("factor_rank", 16)),
    ).to(device)
    model.train()
    images = torch.randn(batch_size, 3, 360, 640, device=device)
    action = torch.randint(0, 2, (batch_size, 4), device=device).float()
    reason = torch.randint(0, 2, (batch_size, 21), device=device).float()
    start = time.perf_counter()
    output = model(images, progress=1.0)
    forward_seconds = time.perf_counter() - start
    loss = meter_action_loss(output, action)["total"] + meter_reason_loss(
        output,
        reason,
        output["factor_reliability"].detach(),
        observability=output["factor_observability"].detach(),
    )["total"]
    start = time.perf_counter()
    (loss / grad_accum).backward()
    backward_seconds = time.perf_counter() - start
    reserved = (
        torch.cuda.max_memory_reserved(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )
    allocated = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else 0.0
    )
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "samples_per_second": batch_size
        / max(forward_seconds + backward_seconds, 1e-6),
        "reserved_gb": reserved,
        "allocated_gb": allocated,
        "dino_call_count": model.foundation.ordinary_dino_calls,
        "finite": bool(torch.isfinite(loss)),
        "stable": bool(torch.isfinite(loss))
        and model.foundation.ordinary_dino_calls == 1
        and reserved < float(config["runtime"]["hard_reserved_gb"]),
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
    rows: list[dict[str, Any]] = []
    for batch_size, grad_accum in DEFAULT_CANDIDATES:
        try:
            rows.append(
                _profile_candidate(
                    config,
                    device,
                    batch_size,
                    grad_accum,
                    use_mock_dino=args.use_mock_dino,
                )
            )
        except torch.cuda.OutOfMemoryError:
            rows.append(
                {
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "stable": False,
                    "failure": "cuda_oom",
                }
            )
            torch.cuda.empty_cache()
    stable = [row for row in rows if row.get("stable")]
    chosen = (
        max(stable, key=lambda row: float(row["samples_per_second"]))
        if stable
        else None
    )
    result = {
        "plan": build_two_stage_profile_plan(),
        "candidates": rows,
        "chosen": chosen,
        "pass": chosen is not None,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "METER_V2_TESA_RUNTIME_PROFILE.json", result)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
