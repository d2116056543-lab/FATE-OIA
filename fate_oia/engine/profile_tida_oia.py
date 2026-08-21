from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import torch

from fate_oia.engine.train_tida_oia import build_optimizer, build_runtime, load_config
from fate_oia.losses.tida_losses import build_tida_loss_registry
from fate_oia.utils.tida_artifacts import atomic_write_json
from fate_oia.utils.tida_contracts import choose_memory_candidate


def normalized_growth_per_100_samples(reserved: torch.Tensor) -> float:
    """Fit reserved-memory slope and express it per 100 measured microsteps."""
    values = reserved.detach().float().flatten()
    if values.numel() < 2:
        return 0.0
    x = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
    x = x - x.mean()
    slope = ((x * (values - values.mean())).sum() / x.square().sum().clamp_min(1e-12)).item()
    return float(slope * 100.0)


def candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    required = ((4, 8, 2), (3, 10, 3), (2, 15, 5), (1, 30, 7))
    rows = [dict(row) for row in config["memory_probe"]["candidates"]]
    actual = tuple((int(r["batch_size"]), int(r["gradient_accumulation_steps"]), int(r["context_chunk_size"])) for r in rows)
    if actual != required:
        raise ValueError(f"memory candidates drifted from the TIDA contract: {actual}")
    return rows


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_candidate(args: Any, spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    local = SimpleNamespace(**vars(args))
    local.batch_size = int(spec["batch_size"])
    local.gradient_accumulation_steps = int(spec["gradient_accumulation_steps"])
    local.context_chunk_size = int(spec["context_chunk_size"])
    local.max_samples = None
    local.checkpoint = None
    runtime = build_runtime(local)
    model, device = runtime.model, runtime.device
    optimizer = build_optimizer(model, config)
    warmup = int(config["memory_probe"]["warmup_updates"])
    measured = int(config["memory_probe"]["measured_updates"])
    if warmup != 10 or measured != 50:
        raise ValueError("profile must use 10 warm-up and 50 measured micro steps")
    timings = {key: 0.0 for key in ("decode", "target_dino", "context_dino", "query_temporal", "backward")}
    hook_start: dict[str, float] = {}
    step_dino = {"target_dino": 0.0, "context_dino": 0.0}

    def dino_pre(_module, inputs):
        _sync(device)
        key = "target_dino" if tuple(inputs[0].shape[-2:]) == (360, 640) else "context_dino"
        hook_start[key] = time.perf_counter()

    def dino_post(_module, _inputs, _output):
        _sync(device)
        for key in ("target_dino", "context_dino"):
            if key in hook_start:
                elapsed = time.perf_counter() - hook_start.pop(key)
                timings[key] += elapsed
                step_dino[key] += elapsed
                break

    dino = model.image_model.foundation.dino
    handles = [dino.register_forward_pre_hook(dino_pre), dino.register_forward_hook(dino_post)]
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(runtime.loaders["train_core"])
    optimizer.zero_grad(set_to_none=True)
    measured_samples = 0
    reserved_samples: list[float] = []
    intervention_events = 0
    previous_end = time.perf_counter()
    oom = False
    nan = False
    try:
        for step in range(warmup + measured):
            if step == warmup:
                for key in timings:
                    timings[key] = 0.0
                reserved_samples.clear()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                previous_end = time.perf_counter()
            step_dino["target_dino"] = 0.0
            step_dino["context_dino"] = 0.0
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(runtime.loaders["train_core"]); batch = next(iterator)
            now = time.perf_counter()
            if step >= warmup:
                timings["decode"] += now - previous_end
            batch = _to_device(batch, device)
            _sync(device); forward_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"], temporal_action_scale=1.0, temporal_reason_scale=1.0)
                loss = build_tida_loss_registry(
                    output, batch["action"], batch["reason"], weights=config["loss"]
                ).total() / int(spec["gradient_accumulation_steps"])
            _sync(device)
            if step >= warmup:
                timings["query_temporal"] += max(
                    0.0,
                    time.perf_counter() - forward_start - step_dino["target_dino"] - step_dino["context_dino"],
                )
            backward_start = time.perf_counter(); loss.backward(); _sync(device)
            if step >= warmup:
                timings["backward"] += time.perf_counter() - backward_start
                measured_samples += int(batch["target_image"].shape[0])
                if device.type == "cuda":
                    reserved_samples.append(torch.cuda.memory_reserved(device) / 2**30)
            if (step + 1) % int(spec["gradient_accumulation_steps"]) == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step in (warmup + measured // 3, warmup + 2 * measured // 3):
                name = "time_shuffle" if intervention_events == 0 else "repeated_last"
                model.rerun_temporal_from_output(output, name, temporal_action_scale=1.0, temporal_reason_scale=1.0)
                intervention_events += 1
            nan = nan or not bool(torch.isfinite(loss).all())
            previous_end = time.perf_counter()
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        for handle in handles:
            handle.remove()
    total_measured = sum(timings.values())
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
    growth = normalized_growth_per_100_samples(torch.tensor(reserved_samples)) if reserved_samples else 0.0
    row = {
        **spec, "warmup_micro_steps": warmup, "measured_micro_steps": measured,
        "profile_real_video": True, "bf16": device.type == "cuda", "intervention_events": intervention_events,
        "oom": oom, "nan": nan, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        "peak_reserved_gib": peak_reserved,
        "growth_gib": growth,
        "growth_gib_per_100_measured_microsteps": growth,
        "samples_per_second": measured_samples / max(total_measured, 1e-9),
        "timing_seconds": timings,
        "timing_fraction": {key: value / max(total_measured, 1e-9) for key, value in timings.items()},
    }
    del optimizer, model, runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = [profile_candidate(args, spec, config) for spec in candidate_specs(config)]
    selected = choose_memory_candidate(rows, float(config["memory_probe"]["max_reserved_gib"]), float(config["memory_probe"]["max_growth_gib"]))
    payload = {"pass": not any(row["nan"] for row in rows), "candidates": rows, "selected": selected}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
