from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


class ICDORProfileError(RuntimeError):
    """Raised when a runtime candidate lacks a complete real measurement."""


def select_runtime_candidate(records: Iterable[Mapping[str, Any]], *, max_reserved_gb: float = 43.5) -> dict[str, Any]:
    """Select only a complete, stable real measurement under the reserved-memory cap."""
    accepted: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for raw in records:
        record = dict(raw)
        if int(record.get("warmup_steps", 0)) < 20:
            incomplete.append("20 warmup steps")
            continue
        if int(record.get("measured_steps", 0)) < 100:
            incomplete.append("100 measured steps")
            continue
        if record.get("status") != "PASS":
            continue
        try:
            throughput = float(record["samples_per_sec"])
            reserved = float(record["max_reserved_gb"])
            batch_size = int(record["batch_size"])
            accum = int(record["grad_accum"])
        except (KeyError, TypeError, ValueError) as error:
            raise ICDORProfileError(f"invalid runtime measurement: {record}") from error
        if not math.isfinite(throughput) or throughput <= 0 or not math.isfinite(reserved):
            continue
        if reserved > max_reserved_gb:
            continue
        record["effective_batch"] = batch_size * accum
        accepted.append(record)
    if not accepted:
        detail = ", ".join(sorted(set(incomplete))) or "no passing measurement under reserved-memory limit"
        raise ICDORProfileError(f"no eligible IC-DOR runtime candidate: {detail}")
    return max(accepted, key=lambda item: (float(item["samples_per_sec"]), int(item["batch_size"])))


def profile_runtime_candidates(
    candidates: Iterable[Mapping[str, Any]],
    worker_candidates: Iterable[int],
    measure: Callable[[dict[str, Any], int], Mapping[str, Any]],
    *,
    max_reserved_gb: float,
) -> dict[str, Any]:
    """Execute every configured batch/worker combination; never accept synthetic records."""
    records: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        if "batch_size" not in candidate or "grad_accum" not in candidate:
            raise ICDORProfileError(f"runtime candidate is incomplete: {candidate}")
        for workers in worker_candidates:
            if int(workers) < 0:
                raise ICDORProfileError("num_workers candidate cannot be negative")
            try:
                record = dict(measure(candidate, int(workers)))
            except torch.cuda.OutOfMemoryError as error:
                record = {
                    **candidate,
                    "num_workers": int(workers),
                    "status": "OOM",
                    "error": str(error),
                    "warmup_steps": 0,
                    "measured_steps": 0,
                }
            record["measurement_origin"] = "real_phase_d_execution"
            records.append(record)
    selected = select_runtime_candidate(records, max_reserved_gb=max_reserved_gb)
    return {
        "pass": True,
        "status": "PASS",
        "selection_rule": "highest_samples_per_sec_under_reserved_limit_then_larger_batch",
        "max_reserved_gb_limit": max_reserved_gb,
        "selected": selected,
        # Trainer compatibility fields are copied from the measured winner.
        "batch_size": int(selected["batch_size"]),
        "grad_accum": int(selected["grad_accum"]),
        "num_workers": int(selected["num_workers"]),
        "candidates": records,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_batch(iterator: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    while True:
        yielded = False
        for batch in iterator:
            yielded = True
            yield batch
        if not yielded:
            raise ICDORProfileError("profile loader produced no batches")


def measure_real_phase_d(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    grad_accum: int,
    num_workers: int,
    warmup_steps: int = 20,
    measured_steps: int = 100,
) -> dict[str, Any]:
    """Measure actual DINO->factor->reason->route forward, backward, and optimizer steps."""
    if warmup_steps < 20 or measured_steps < 100:
        raise ICDORProfileError("IC-DOR profiler requires 20 warmup and 100 measured steps")
    model.train()
    batches = iter(_as_batch(loader))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    step_times: list[float] = []
    data_wait: list[float] = []
    last_end = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for index in range(warmup_steps + measured_steps):
        batch = next(batches)
        loaded = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        reason = batch["reason"].to(device, non_blocking=True)
        _synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                images,
                route_mode="admitted",
                latent_enabled=True,
                reason_route_mode="full",
                return_masks=False,
            )
            action_logits = output.get("action_final_logits")
            reason_logits = output.get("reason_observed_logits")
            if not isinstance(action_logits, torch.Tensor) or not isinstance(reason_logits, torch.Tensor):
                raise ICDORProfileError("Phase-D forward did not return real action/reason logits")
            loss = F.binary_cross_entropy_with_logits(action_logits, action) + F.binary_cross_entropy_with_logits(reason_logits, reason)
        if not torch.isfinite(loss):
            raise ICDORProfileError("Phase-D profile produced non-finite loss")
        (loss / grad_accum).backward()
        if (index + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        _synchronize(device)
        end = time.perf_counter()
        if index >= warmup_steps:
            step_times.append(end - start)
            data_wait.append(loaded - last_end)
        last_end = end
    sorted_steps = sorted(step_times)
    p95_index = min(len(sorted_steps) - 1, math.ceil(0.95 * len(sorted_steps)) - 1)
    total = sum(step_times)
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "num_workers": num_workers,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "status": "PASS",
        "samples_per_sec": batch_size * measured_steps / max(total, 1e-9),
        "step_p50_ms": 1000.0 * statistics.median(step_times),
        "step_p95_ms": 1000.0 * sorted_steps[p95_index],
        "cpu_data_wait_p50_ms": 1000.0 * statistics.median(data_wait),
        "cpu_data_wait_p95_ms": 1000.0 * sorted(data_wait)[p95_index],
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
        "phase_d_route_forward_backward": True,
        "bf16": device.type == "cuda",
    }


def write_runtime_selection(output: str | Path, records: Iterable[Mapping[str, Any]], *, max_reserved_gb: float = 43.5) -> dict[str, Any]:
    records_list = [dict(record) for record in records]
    selected = select_runtime_candidate(records_list, max_reserved_gb=max_reserved_gb)
    payload = {
        "pass": True,
        "selection_rule": "highest_samples_per_sec_under_reserved_limit_then_larger_batch",
        "max_reserved_gb_limit": max_reserved_gb,
        "selected": selected,
        "candidates": records_list,
    }
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="IC-DOR real Phase-D runtime profiler")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_reserved_gb", type=float, default=43.5)
    args = parser.parse_args()
    from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
        build_icdor_loaders,
        build_icdor_model,
        build_icdor_optimizer,
        load_config,
    )
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ICDORProfileError("formal IC-DOR profiler requires a real CUDA DINO execution")
    runtime_candidates = list(config["training"]["runtime_candidates"])
    worker_candidates = list(config["data"]["num_workers_candidates"])
    warmup = int(config["runtime"]["warmup_profile_steps"])
    measured = int(config["runtime"]["timed_profile_steps"])
    profile_root = Path(args.output).resolve().parent / "runtime_profile_data"

    def measure(candidate: dict[str, Any], workers: int) -> Mapping[str, Any]:
        batch_size = int(candidate["batch_size"])
        grad_accum = int(candidate["grad_accum"])
        loader, _, _, _, _ = build_icdor_loaders(
            config, profile_root / f"b{batch_size}_a{grad_accum}_w{workers}",
            batch_size=batch_size, num_workers=workers,
            max_audit_samples=1, max_calib_samples=1, max_test_samples=1,
        )
        model = build_icdor_model(config).to(device)
        optimizer, _ = build_icdor_optimizer(model, config)
        try:
            return measure_real_phase_d(
                model, optimizer, loader, device=device, batch_size=batch_size,
                grad_accum=grad_accum, num_workers=workers,
                warmup_steps=warmup, measured_steps=measured,
            )
        finally:
            del optimizer, model, loader
            torch.cuda.empty_cache()

    payload = profile_runtime_candidates(
        runtime_candidates, worker_candidates, measure,
        max_reserved_gb=float(config["training"].get("max_reserved_vram_gb", args.max_reserved_gb)),
    )
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

