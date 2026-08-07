from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from fate_oia.datasets.aie_cert_structured_evidence import AIECertStructuredEvidenceBuilder
from fate_oia.losses.aie_cert_constraints import AIECertDualState
from fate_oia.utils.aie_cert_artifacts import write_json
from fate_oia.utils.aie_cert_preference_queue import AIECertPreferenceQueue
from fate_oia.utils.aie_cert_schedule import schedule_values
from .train_aie_cert_oia import (build_ecpo, build_model, build_optimizer, compute_loss, load_config,
                                 make_dataset, make_loader, run_counterfactual)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_gpu_memory_mb() -> tuple[int, list[dict[str, int]]]:
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    rows: list[dict[str, int]] = []
    if result.returncode != 0:
        return 0, rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            continue
        pid, memory = int(parts[0]), int(parts[1])
        if pid != os.getpid():
            rows.append({"pid": pid, "memory_mb": memory})
    return sum(row["memory_mb"] for row in rows), rows


def _profile_candidate(cfg, device, dataset, candidate, builder):
    batch, accum, chunk, workers = candidate
    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    dual = AIECertDualState(cfg["dual"]["lr"], cfg["dual"]["ema_decay"], cfg["dual"]["lambda_max"]).to(device)
    queue = AIECertPreferenceQueue(cfg["ecpo"]["queue_capacity"], cfg["ecpo"]["max_age_updates"], cfg["ecpo"]["age_tau"])
    loader = make_loader(dataset, batch, True, workers, cfg)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    warmup, measured_steps = 10, 30
    step_times, cf_times, cf_valid = [], [], 0
    memory_trace = []
    start = time.perf_counter()
    for step in range(warmup + measured_steps):
        try:
            row = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            row = next(iterator)
        images = row["image"].to(device, non_blocking=True)
        action, reason = row["action"].to(device), row["reason"].to(device)
        structured = builder.build(row["file_name"], device=device)
        schedule = schedule_values(step, warmup + measured_steps, cfg)
        tick = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            field = model.encode_images(images)
            output = model.decode_from_field(field, action_scale=schedule["action_scale"],
                reason_budget_max=schedule["reason_budget_max"], predicate_prior_scale=schedule["predicate_prior_scale"],
                transport_gamma_cap=schedule["transport_gamma_cap"])
            cf = None
            if step in (10, 30):
                cf_tick = time.perf_counter()
                cf = run_counterfactual(model, field, output, action, schedule)
                if device.type == "cuda": torch.cuda.synchronize(device)
                cf_times.append(time.perf_counter() - cf_tick)
                cf_valid += int(cf["valid_mask"].sum())
            ecpo_pack = build_ecpo(output, {"reason": reason}, structured, step, queue,
                                   cfg["ecpo"]["verified_counter_threshold"], cfg["ecpo"]["pairs_per_label"])
            loss, _, constraints, availability, _ = compute_loss(
                output, {"action": action, "reason": reason}, structured, cfg, schedule, cf, dual, ecpo_pack)
        loss.backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["global_grad_clip"])
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            dual.update({name: value for name, value in constraints.items() if availability.get(name)}, schedule["dual_scale"])
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        if step >= warmup:
            step_times.append(elapsed)
            memory_trace.append(torch.cuda.memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0)
    total = time.perf_counter() - start
    peak = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    growth = max(memory_trace[-5:] or [0.0]) - min(memory_trace[:5] or [0.0])
    return {"batch": batch, "accum": accum, "chunk": chunk, "workers": workers,
            "warmup_steps": warmup, "measured_steps": measured_steps,
            "samples_per_sec": batch * measured_steps / max(sum(step_times), 1e-9),
            "wall_seconds": total, "reserved_gb": peak, "steady_memory_growth_gb": growth,
            "cf_events": len(cf_times), "cf_valid_samples": cf_valid,
            "cf_seconds_mean": sum(cf_times) / max(len(cf_times), 1),
            "valid": peak < cfg["runtime"]["max_reserved_memory_gb"] and growth <= cfg["runtime"]["memory_growth_tolerance_gb"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); root = Path.cwd(); config_path = Path(args.config).resolve()
    cfg = load_config(config_path); device = torch.device(args.device)
    external_mb, external_processes = _external_gpu_memory_mb() if device.type == "cuda" else (0, [])
    binding = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "config_hash": _sha256(config_path), "source_head": cfg["experiment"]["source_head"]}
    if external_mb > 2048:
        write_json(args.output, {"pass": False, "reason": "other_gpu_process_over_2gb",
                                 "external_gpu_memory_mb": external_mb, "external_processes": external_processes,
                                 **binding})
        raise SystemExit(1)
    candidates = [(8, 4, 16, 6), (7, 4, 16, 6), (6, 5, 16, 6), (5, 6, 16, 6), (6, 5, 8, 4)]
    dataset = torch.utils.data.Subset(make_dataset(cfg, "train"), range(320))
    builder = AIECertStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"],
        cfg["primary"]["reason_counter_evidence"], cfg["data"]["bdd100k_root"])
    builder.preload([dataset.dataset.samples[index].file_name for index in dataset.indices])
    results = []
    for candidate in candidates:
        try:
            result = _profile_candidate(cfg, device, dataset, candidate, builder)
        except torch.cuda.OutOfMemoryError:
            result = {"batch": candidate[0], "accum": candidate[1], "chunk": candidate[2],
                      "workers": candidate[3], "valid": False, "reason": "oom"}
        except Exception as exc:
            result = {"batch": candidate[0], "accum": candidate[1], "chunk": candidate[2],
                      "workers": candidate[3], "valid": False, "reason": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    valid = [row for row in results if row.get("valid") and row.get("cf_events", 0) >= 2]
    selected = max(valid, key=lambda row: row["samples_per_sec"]) if valid else None
    write_json(args.output, {"pass": selected is not None, "formal_loss_profile": True,
                             "counterfactual_profile": True, "candidates": results, "selected": selected,
                             "external_gpu_memory_mb": external_mb, "external_processes": external_processes,
                             **binding})
    if selected is None: raise SystemExit(1)


if __name__ == "__main__": main()
