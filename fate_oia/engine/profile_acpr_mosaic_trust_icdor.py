from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


class ICDORProfileError(RuntimeError):
    """Raised when a runtime candidate lacks a complete real measurement."""


def wall_clock_samples_per_second(
    batch_size: int,
    compute_seconds: Iterable[float],
    data_wait_seconds: Iterable[float],
) -> float:
    """Return end-to-end throughput, including synchronous data loading."""
    compute = list(compute_seconds)
    wait = list(data_wait_seconds)
    if batch_size < 1 or not compute or len(compute) != len(wait):
        raise ICDORProfileError("wall-clock throughput requires aligned measured steps")
    elapsed = sum(compute) + sum(wait)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ICDORProfileError("wall-clock throughput observed invalid elapsed time")
    return batch_size * len(compute) / elapsed


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
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    completed_records: Iterable[Mapping[str, Any]] = (),
    checkpoint_callback: Callable[[Iterable[Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Execute every configured batch/worker combination; never accept synthetic records."""
    records = [dict(record) for record in completed_records]
    completed = {
        (int(record["batch_size"]), int(record["grad_accum"]), int(record["num_workers"]))
        for record in records
        if record.get("measurement_origin") == "real_phase_d_execution"
        and record.get("status") in {"PASS", "OOM"}
        and all(key in record for key in ("batch_size", "grad_accum", "num_workers"))
    }
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        if "batch_size" not in candidate or "grad_accum" not in candidate:
            raise ICDORProfileError(f"runtime candidate is incomplete: {candidate}")
        for workers in worker_candidates:
            if int(workers) < 0:
                raise ICDORProfileError("num_workers candidate cannot be negative")
            key = (int(candidate["batch_size"]), int(candidate["grad_accum"]), int(workers))
            if key in completed:
                if progress_callback is not None:
                    progress_callback({
                        "event": "candidate_resume_skip",
                        "batch_size": key[0],
                        "grad_accum": key[1],
                        "num_workers": key[2],
                    })
                continue
            try:
                if progress_callback is not None:
                    progress_callback({
                        "event": "candidate_start",
                        "batch_size": int(candidate["batch_size"]),
                        "grad_accum": int(candidate["grad_accum"]),
                        "num_workers": int(workers),
                    })
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
            completed.add(key)
            if checkpoint_callback is not None:
                checkpoint_callback(records)
            if progress_callback is not None:
                progress_callback({
                    "event": "candidate_end",
                    "batch_size": int(candidate["batch_size"]),
                    "grad_accum": int(candidate["grad_accum"]),
                    "num_workers": int(workers),
                    "status": str(record.get("status", "UNKNOWN")),
                })
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
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Measure actual DINO->factor->reason->route forward, backward, and optimizer steps."""
    if warmup_steps < 20 or measured_steps < 100:
        raise ICDORProfileError("IC-DOR profiler requires 20 warmup and 100 measured steps")
    if progress_every <= 0:
        raise ICDORProfileError("IC-DOR profiler progress_every must be positive")
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
        if progress_callback is not None and (
            (index + 1) % progress_every == 0 or index + 1 == warmup_steps + measured_steps
        ):
            progress_callback({
                "event": "step",
                "batch_size": int(batch_size),
                "grad_accum": int(grad_accum),
                "num_workers": int(num_workers),
                "step": int(index + 1),
                "total_steps": int(warmup_steps + measured_steps),
                "measured_steps_complete": int(max(0, index + 1 - warmup_steps)),
                "gpu_reserved_gb": (
                    float(torch.cuda.max_memory_reserved(device) / 2**30)
                    if device.type == "cuda" else 0.0
                ),
            })
        last_end = end
    sorted_steps = sorted(step_times)
    p95_index = min(len(sorted_steps) - 1, math.ceil(0.95 * len(sorted_steps)) - 1)
    compute_only_throughput = batch_size * measured_steps / max(sum(step_times), 1e-9)
    end_to_end_throughput = wall_clock_samples_per_second(batch_size, step_times, data_wait)
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "num_workers": num_workers,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "status": "PASS",
        "samples_per_sec": end_to_end_throughput,
        "compute_only_samples_per_sec": compute_only_throughput,
        "throughput_includes_data_wait": True,
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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ICDORProfileError("formal IC-DOR profiler requires a real CUDA DINO execution")
    runtime_candidates = list(config["training"]["runtime_candidates"])
    worker_candidates = list(config["data"]["num_workers_candidates"])
    warmup = int(config["runtime"]["warmup_profile_steps"])
    measured = int(config["runtime"]["timed_profile_steps"])
    profile_root = Path(args.output).resolve().parent / "runtime_profile_data"
    output_path = Path(args.output)
    partial_path = output_path.with_name(output_path.stem + "_partial.json")
    config_sha256 = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    completed_records: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if (
            partial.get("schema_version") != "icdor_runtime_profile_partial.v1"
            or partial.get("config_sha256") != config_sha256
            or partial.get("runtime_candidates") != runtime_candidates
            or partial.get("worker_candidates") != worker_candidates
        ):
            raise ICDORProfileError("runtime profile partial checkpoint is incompatible with this request")
        existing = partial.get("candidates")
        if not isinstance(existing, list):
            raise ICDORProfileError("runtime profile partial checkpoint has invalid candidates")
        completed_records = [dict(record) for record in existing]

    def checkpoint(records: Iterable[Mapping[str, Any]]) -> None:
        _write_json_atomic(partial_path, {
            "schema_version": "icdor_runtime_profile_partial.v1",
            "config_sha256": config_sha256,
            "runtime_candidates": runtime_candidates,
            "worker_candidates": worker_candidates,
            "candidates": [dict(record) for record in records],
        })
    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])

    def measure(candidate: dict[str, Any], workers: int) -> Mapping[str, Any]:
        batch_size = int(candidate["batch_size"])
        grad_accum = int(candidate["grad_accum"])
        loader, _, _, _, _, _ = build_icdor_loaders(
            config, profile_root / f"b{batch_size}_a{grad_accum}_w{workers}",
            batch_size=batch_size, num_workers=workers,
            max_audit_samples=1, max_calib_samples=1, max_test_samples=1,
            visual_grounding_index=grounding_index,
        )
        model = build_icdor_model(config).to(device)
        optimizer, _ = build_icdor_optimizer(model, config)
        try:
            return measure_real_phase_d(
                model, optimizer, loader, device=device, batch_size=batch_size,
                grad_accum=grad_accum, num_workers=workers,
                warmup_steps=warmup, measured_steps=measured,
                progress_callback=lambda row: print(
                    "icdor_profile " + json.dumps(dict(row), sort_keys=True), flush=True
                ),
            )
        finally:
            del optimizer, model, loader
            torch.cuda.empty_cache()

    payload = profile_runtime_candidates(
        runtime_candidates, worker_candidates, measure,
        max_reserved_gb=float(config["training"].get("max_reserved_vram_gb", args.max_reserved_gb)),
        progress_callback=lambda row: print(
            "icdor_profile " + json.dumps(dict(row), sort_keys=True), flush=True
        ),
        completed_records=completed_records,
        checkpoint_callback=checkpoint,
    )
    _write_json_atomic(output_path, payload)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

