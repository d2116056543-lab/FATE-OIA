from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from .train_aie_oia import build_model, load_config, make_dataset, make_loader
from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder
from fate_oia.utils.aie_artifacts import write_json
from fate_oia.utils.aie_counterfactual import AIECounterfactualConfig, AIECounterfactualEngine
from fate_oia.utils.aie_hashes import file_sha256


CANDIDATES = ((6, 5, 16), (6, 5, 8), (5, 6, 16), (4, 8, 16))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--measure", type=int, default=30); parser.add_argument("--candidate", help="Profile one batch,accum,chunk tuple")
    args = parser.parse_args(); cfg = load_config(args.config); device = torch.device(args.device); rows = []
    candidates = CANDIDATES
    if args.candidate:
        selected_candidate = tuple(int(value) for value in args.candidate.split(","))
        if len(selected_candidate) != 3 or selected_candidate not in CANDIDATES:
            raise ValueError(f"Unsupported AIE profile candidate: {args.candidate}")
        candidates = (selected_candidate,)
    for batch, accum, chunk in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg, device).train(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        model.action_evidence.probe_chunk_size = chunk
        dataset = make_dataset(cfg, "train")
        loader = make_loader(dataset, batch, True, 8, cfg)
        iterator = iter(loader)
        structured_builder = AIEStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"], cfg["data"]["bdd100k_root"])
        cf_cfg = AIECounterfactualConfig(**{key: cfg["counterfactual"][key] for key in AIECounterfactualConfig.__dataclass_fields__})
        cf_engine = AIECounterfactualEngine(cf_cfg)
        failed = None; images = action_target = out = loss = None
        measured = []; data_samples = []; allocated_samples = []; cf_events = 0; cf_seconds = 0.0; cf_valid = 0
        try:
            for step in range(args.warmup + args.measure):
                cf = None
                torch.cuda.synchronize(); start = time.perf_counter()
                batch_data = next(iterator)
                images = batch_data["image"].to(device, non_blocking=True)
                action_target = batch_data["action"].to(device, non_blocking=True)
                structured_builder.build(batch_data["file_name"], device=device)
                torch.cuda.synchronize(); data_seconds = time.perf_counter() - start
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(images, action_scale=1.0, reason_scale=1.0)
                    loss = (
                        out["action_logits_primary"].square().mean()
                        + out["reason_logits_primary"].square().mean()
                        + out["predicate_logits"].square().mean()
                        + out["action_logits_final_train"].square().mean()
                        + out["reason_logits_final_train"].square().mean()
                    )
                    boundary = (step + 1) % accum == 0
                    update = (step + 1) // accum
                    final_available_boundary = boundary and (step + 1 + accum > args.warmup + args.measure)
                    if boundary and update > 0 and (update % 4 == 0 or (final_available_boundary and cf_events < 2)):
                        cf_start = time.perf_counter()
                        cf = cf_engine.run(
                            model, out, action_target, batch_data["file_name"],
                            global_update=update, action_scale=1.0,
                        )
                        loss = loss + 0.01 * cf["selected_minus_control"].mean()
                        cf_events += 1; cf_valid += int(cf["cf_valid_count"])
                        torch.cuda.synchronize(); cf_seconds += time.perf_counter() - cf_start
                (loss / accum).backward()
                if (step + 1) % accum == 0: optimizer.step(); optimizer.zero_grad(set_to_none=True)
                # CF produces several same-field intervention graphs. They are
                # transient by contract and must not contaminate the steady-state
                # allocation measurement after backward has consumed them.
                cf = None
                torch.cuda.synchronize()
                if step >= args.warmup:
                    measured.append(time.perf_counter() - start)
                    data_samples.append(data_seconds)
                    allocated_samples.append(torch.cuda.memory_allocated() / 2**30)
        except torch.cuda.OutOfMemoryError as exc:
            failed = str(exc); optimizer.zero_grad(set_to_none=True)
        reserved = torch.cuda.max_memory_reserved() / 2**30
        window = min(5, len(allocated_samples) // 2)
        steady_growth = (
            statistics.median(allocated_samples[-window:]) - statistics.median(allocated_samples[:window])
            if window else None
        )
        rows.append({"batch_size": batch, "gradient_accumulation_steps": accum, "probe_chunk_size": chunk, "oom": failed is not None,
            "reserved_gb": reserved, "samples_per_second": (batch * len(measured) / sum(measured)) if measured else 0.0,
            "num_workers": 8, "official_dino": True, "dino_calls_per_ordinary_batch": 1, "dino_calls_per_cf_event": 0,
            "cf_amortized": True, "cf_events": cf_events, "cf_valid_count": cf_valid, "cf_seconds": cf_seconds,
            "allocated_gb_first": allocated_samples[0] if allocated_samples else None,
            "allocated_gb_last": allocated_samples[-1] if allocated_samples else None,
            "allocated_growth_gb": (allocated_samples[-1] - allocated_samples[0]) if allocated_samples else None,
            "steady_state_growth_gb": steady_growth,
            "memory_growth_tolerance_gb": 0.25,
            "mean_data_seconds": (sum(data_samples) / len(data_samples)) if data_samples else None,
            "mean_step_seconds": (sum(measured) / len(measured)) if measured else None})
        del model, optimizer, loader, iterator, dataset, images, action_target, out, loss
        if "cf" in locals():
            del cf
    valid = [row for row in rows if not row["oom"]
        and row["reserved_gb"] < float(cfg["runtime"]["max_reserved_memory_gb"])
        and row["cf_events"] >= 2
        and row["steady_state_growth_gb"] is not None
        and row["steady_state_growth_gb"] <= row["memory_growth_tolerance_gb"]]
    if not valid: raise RuntimeError("No AIE runtime profile candidate satisfies memory contract")
    best_speed = max(row["samples_per_second"] for row in valid)
    near = [row for row in valid if row["samples_per_second"] >= best_speed * 0.97]
    selected = min(near, key=lambda row: row["reserved_gb"])
    payload = {
        "pass": True,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hash": file_sha256(args.config),
        "candidates": rows,
        "selected": selected,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); write_json(output / "AIE_RUNTIME_PROFILE.json", payload); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
