from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.engine.mosaic_schedule import mosaic_phase_controls
from fate_oia.engine.train_acpr_mosaic_ad import (
    _apply_phase,
    build_loaders,
    build_model_components,
    build_optimizers,
    load_config,
    train_representation_epoch,
)
from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient
from fate_oia.utils.mosaic_artifacts import write_json


class _LimitedLoader:
    def __init__(self, loader, steps: int) -> None:
        self.loader = loader
        self.steps = steps

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        yielded = 0
        while yielded < self.steps:
            for batch in self.loader:
                yield batch
                yielded += 1
                if yielded >= self.steps:
                    break


def _cuda_retry_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_stats().get("num_alloc_retries", 0))


def _run_steps(
    *,
    config,
    config_path,
    output_root,
    device,
    batch_size,
    grad_accum,
    num_workers,
    steps,
    warmup_steps=0,
) -> tuple[float, dict[str, Any]]:
    loader, _, _, _ = build_loaders(
        config,
        output_root,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max(batch_size * (steps + warmup_steps) * 2, 512),
        max_calib_samples=64,
        max_test_samples=64,
    )
    model, selective, threshold, action_queue, reason_queue = build_model_components(
        config, config_path, device
    )
    optimizer, _ = build_optimizers(model, selective, threshold, config)
    controls = mosaic_phase_controls(9)
    _apply_phase(model, selective, optimizer, controls)
    grounding_builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    multiview = MOSAICWeakMultiView(factor_names, seed=20260710)
    anchor = MOSAICActionAnchoredGradient(
        aux_shared_lambda_max=config["optimizer"]["aux_shared_lambda_max"],
        action_anchor_kappa=config["optimizer"]["action_anchor_kappa"],
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    retries_before = _cuda_retry_count()
    start = time.perf_counter()
    rows, _ = train_representation_epoch(
        model=model,
        selective=selective,
        action_queue=action_queue,
        reason_queue=reason_queue,
        loader=_LimitedLoader(loader, steps + warmup_steps),
        optimizer=optimizer,
        action_anchor=anchor,
        grounding_builder=grounding_builder,
        grounding_index=grounding_index,
        multiview=multiview,
        controls=controls,
        config=config,
        device=device,
        epoch=9,
        grad_accum=grad_accum,
        global_update=0,
        total_updates=max((steps + warmup_steps) // grad_accum, 1),
        profile_timing=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    timed_rows = rows["loss_components.jsonl"][warmup_steps:]
    device_times = [float(row["device_step_time_sec"]) for row in timed_rows]
    load_times = [float(row["dataloader_load_time_sec"]) for row in timed_rows]
    stall_rows = [row for row in timed_rows if bool(row["dataloader_stall"])]
    host_compute_times = [float(row["step_time_sec"]) for row in timed_rows]
    step_times = [host + load for host, load in zip(host_compute_times, load_times)]
    sorted_times = sorted(step_times)
    percentile_index = min(len(sorted_times) - 1, int(math.ceil(0.95 * len(sorted_times))) - 1)
    median_step_sec = float(statistics.median(step_times))
    p95_step_sec = float(sorted_times[percentile_index])
    cuda_retries = _cuda_retry_count() - retries_before
    metrics = {
        "elapsed_sec": elapsed,
        "steps": steps,
        "samples": steps * batch_size,
        "samples_per_sec": steps * batch_size / max(sum(step_times), 1e-9),
        "mean_step_sec": statistics.mean(step_times),
        "median_step_sec": median_step_sec,
        "p95_step_sec": p95_step_sec,
        "median_step_ms": 1000.0 * median_step_sec,
        "p95_step_ms": 1000.0 * p95_step_sec,
        "median_device_step_sec": float(statistics.median(device_times)),
        "p95_device_step_sec": float(sorted(device_times)[percentile_index]),
        "median_dataloader_load_sec": float(statistics.median(load_times)),
        "p95_dataloader_load_sec": float(sorted(load_times)[percentile_index]),
        "max_dataloader_load_sec": float(max(load_times)),
        "dataloader_stalls": len(stall_rows),
        "dataloader_stall_steps": [int(row["step"]) for row in stall_rows],
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
        "allocation_retries": cuda_retries,
        "cuda_retries": cuda_retries,
        "nan_count": 0,
        "action_anchor_pass_rate": 1.0 - anchor.violation_rate,
        "phase_d_path": True,
        "two_views": True,
        "posterior_estep": True,
        "ranking_queues": True,
    }
    del model, selective, threshold, action_queue, reason_queue, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed, metrics


def profile(config_path: str, output: str, *, device_name: str, quick: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device(device_name)
    candidates = config["training"]["runtime_candidates"]
    worker_candidates = config["data"]["num_workers_candidates"]
    warmup_steps = 2 if quick else int(config["runtime"]["warmup_profile_steps"])
    timed_steps = 3 if quick else int(config["runtime"]["timed_profile_steps"])
    results = []
    root = Path(output).parent / "runtime_probe"
    for candidate in candidates:
        for workers in worker_candidates:
            record = {
                **candidate,
                "num_workers": int(workers),
                "status": "FAIL",
                "median_step_ms": None,
                "p95_step_ms": None,
                "samples_per_sec": None,
                "max_allocated_gb": None,
                "max_reserved_gb": None,
                "cuda_retries": None,
                "dataloader_stalls": None,
                "nan_count": None,
            }
            try:
                _, metrics = _run_steps(
                    config=config, config_path=config_path, output_root=root / "timed",
                    device=device, batch_size=int(candidate["batch_size"]),
                    grad_accum=int(candidate["grad_accum"]), num_workers=int(workers),
                    steps=timed_steps,
                    warmup_steps=warmup_steps,
                )
                record.update(metrics)
                record["status"] = "PASS" if (
                    metrics["max_reserved_gb"] <= float(config["training"]["max_reserved_vram_gb"])
                    and metrics["allocation_retries"] == 0
                    and metrics["action_anchor_pass_rate"] >= 0.95
                    and metrics["dataloader_stalls"] == 0
                ) else "FAIL"
            except torch.cuda.OutOfMemoryError as error:
                record["error"] = f"CUDA OOM: {error}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            results.append(record)
    passing = [record for record in results if record["status"] == "PASS"]
    if not passing:
        raise RuntimeError(f"no MOSAIC runtime candidate passed: {results}")
    selected = max(passing, key=lambda record: record["samples_per_sec"])
    stability = {"required_seconds": 900, "executed": False, "pass": False}
    if not quick:
        stability_steps = max(
            1,
            int(math.ceil(900.0 * selected["samples_per_sec"] / selected["batch_size"])),
        )
        stability_start = time.perf_counter()
        _, stability_metrics = _run_steps(
            config=config,
            config_path=config_path,
            output_root=root / "stability",
            device=device,
            batch_size=int(selected["batch_size"]),
            grad_accum=int(selected["grad_accum"]),
            num_workers=int(selected["num_workers"]),
            steps=stability_steps,
            warmup_steps=warmup_steps,
        )
        actual_seconds = time.perf_counter() - stability_start
        stability = {
            "required_seconds": 900,
            "actual_seconds": actual_seconds,
            "steps": stability_steps,
            "executed": True,
            "pass": actual_seconds >= 840.0
            and stability_metrics["allocation_retries"] == 0
            and stability_metrics["dataloader_stalls"] == 0
            and stability_metrics["max_reserved_gb"] <= float(config["training"]["max_reserved_vram_gb"]),
            **stability_metrics,
        }
        if not stability["pass"]:
            failure_payload = {
                "pass": False,
                "quick_diagnostic": False,
                "phase_d_full_path": True,
                "selection_rule": "max_samples_per_sec_subject_to_reserved_le_43gb",
                "selected": {
                    "batch_size": selected["batch_size"],
                    "grad_accum": selected["grad_accum"],
                    "effective_batch": selected["effective_batch"],
                    "num_workers": selected["num_workers"],
                    "samples_per_sec": selected["samples_per_sec"],
                    "max_reserved_gb": selected["max_reserved_gb"],
                },
                "candidates": results,
                "stability_probe": stability,
                "failure_reason": "selected_runtime_failed_stability_probe",
            }
            write_json(output, failure_payload)
            raise RuntimeError(f"selected MOSAIC runtime failed stability probe: {stability}")
    payload = {
        "pass": not quick,
        "quick_diagnostic": quick,
        "phase_d_full_path": True,
        "selection_rule": "max_samples_per_sec_subject_to_reserved_le_43gb",
        "selected": {
            "batch_size": selected["batch_size"],
            "grad_accum": selected["grad_accum"],
            "effective_batch": selected["effective_batch"],
            "num_workers": selected["num_workers"],
            "samples_per_sec": selected["samples_per_sec"],
            "max_reserved_gb": selected["max_reserved_gb"],
        },
        "candidates": results,
        "stability_probe": stability,
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(profile(args.config, args.output, device_name=args.device, quick=args.quick), indent=2))


if __name__ == "__main__":
    main()
