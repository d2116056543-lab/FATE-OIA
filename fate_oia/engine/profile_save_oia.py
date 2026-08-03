from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from fate_oia.losses.save_action_losses import save_action_loss
from fate_oia.losses.save_grounding_losses import save_grounding_loss
from fate_oia.losses.save_loss_registry import build_save_loss_registry
from fate_oia.losses.save_reason_losses import save_reason_loss


SAVE_PROFILE_CANDIDATES = ((6, 5), (4, 8), (3, 11))
_CORE_PATHS = (
    "predicate",
    "private_reason",
    "utility_cadence",
    "paired_view_cadence",
)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {name: _move(item, device) for name, item in value.items()}
    return value


def _profile_registry(
    output: Mapping[str, Any], batch: Mapping[str, Any], *, view_output: Mapping[str, Any] | None,
) -> Any:
    action = save_action_loss(output, batch["action"])
    reason = save_reason_loss(output, batch["reason"], view_output=view_output)
    grounding = batch.get("meter_grounding")
    if isinstance(grounding, Mapping):
        measurement = save_grounding_loss(output, grounding, split="train", supervision_source="BDD100K")
    else:
        zero = output["action_logits_final"].new_zeros(())
        measurement = {name: zero for name in ("anchor", "state", "null", "matched_background", "mirror", "identity")}
    return build_save_loss_registry(action=action, reason=reason, measurement=measurement)


def validate_profile_row(row: Mapping[str, Any]) -> bool:
    pair = (int(row.get("batch_size", -1)), int(row.get("gradient_accumulation_steps", -1)))
    core = row.get("core_paths", {})
    return (
        pair in SAVE_PROFILE_CANDIDATES
        and row.get("real_dino") is True
        and row.get("bf16") is True
        and int(row.get("warmup_microbatches", -1)) == 20
        and int(row.get("measured_microbatches", -1)) == 50
        and int(row.get("ordinary_dino_calls_per_microbatch", -1)) == 1
        and isinstance(core, Mapping)
        and all(core.get(name) is True for name in _CORE_PATHS)
        and row.get("finite") is True
        and row.get("oom") is False
        and float(row.get("reserved_gb", float("inf"))) < 45.0
        and float(row.get("samples_per_second", 0.0)) > 0.0
    )


def choose_fastest_stable(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    stable = [dict(row) for row in rows if validate_profile_row(row)]
    if not stable:
        raise RuntimeError("SAVE profiler found no stable real-DINO candidate below 45 GB")
    return max(stable, key=lambda row: float(row["samples_per_second"]))


def profile_candidate(
    model: nn.Module,
    batch_factory: Callable[[int, int], Mapping[str, Any]],
    *,
    batch_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    warmup_microbatches: int = 20,
    measured_microbatches: int = 50,
) -> dict[str, Any]:
    """Measure the actual SAVE forward/backward path, including cadence branches."""
    if (batch_size, gradient_accumulation_steps) not in SAVE_PROFILE_CANDIDATES:
        raise ValueError("unplanned SAVE profile candidate")
    if device.type != "cuda":
        raise ValueError("SAVE runtime profile requires the real CUDA DINO path")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-5
    )
    total = warmup_microbatches + measured_microbatches
    measured_samples = 0
    measured_seconds = 0.0
    finite = True
    ordinary_calls = 0
    core_seen = {name: False for name in _CORE_PATHS}
    for microbatch in range(total):
        batch = _move(batch_factory(batch_size, microbatch), device)
        images = batch["image"]
        action = batch["action"]
        optimizer_update = microbatch // gradient_accumulation_steps
        closes_update = (microbatch + 1) % gradient_accumulation_steps == 0
        cadence = closes_update and ((optimizer_update + 1) % 4 == 0)
        utility_update = optimizer_update + int(closes_update)
        if microbatch == warmup_microbatches:
            torch.cuda.synchronize(device)
            measured_start = time.perf_counter()
        before_ordinary = _dino_calls(model)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                images,
                progress=0.5,
                optimizer_update=utility_update,
                action_targets=action,
                run_teacher=True if cadence else None,
            )
            ordinary_calls += _dino_calls(model) - before_ordinary
            core_seen["predicate"] = core_seen["predicate"] or (
                "predicate_map_action" in output and "predicate_token_action" in output
            )
            core_seen["private_reason"] = core_seen["private_reason"] or (
                "reason_logits_private_direct" in output
            )
            if cadence:
                core_seen["utility_cadence"] = core_seen["utility_cadence"] or (
                    output.get("utility_teacher_plan") is not None
                )
            paired_images = batch.get("image_view2")
            paired_output = None
            if cadence and isinstance(paired_images, Tensor):
                paired_output = model(
                    paired_images,
                    progress=0.5,
                    optimizer_update=None,
                    action_targets=action,
                    run_teacher=None,
                )
                core_seen["paired_view_cadence"] = True
            loss = _profile_registry(output, batch, view_output=paired_output).total()
            loss = loss / gradient_accumulation_steps
        finite = finite and bool(torch.isfinite(loss).item())
        loss.backward()
        if (microbatch + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if microbatch >= warmup_microbatches:
            measured_samples += batch_size
    torch.cuda.synchronize(device)
    measured_seconds = time.perf_counter() - measured_start
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "samples_per_second": measured_samples / max(measured_seconds, 1e-9),
        "reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
        "allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        "real_dino": True,
        "bf16": True,
        "warmup_microbatches": warmup_microbatches,
        "measured_microbatches": measured_microbatches,
        "ordinary_dino_calls_per_microbatch": ordinary_calls / total,
        "core_paths": core_seen,
        "finite": finite,
        "oom": False,
    }


def _dino_calls(model: nn.Module) -> int:
    for owner in (model, getattr(model, "foundation", None), getattr(model, "dino_field", None)):
        if owner is not None:
            for name in ("ordinary_dino_calls", "dino_call_count", "forward_calls"):
                value = getattr(owner, name, None)
                if isinstance(value, (int, float)):
                    return int(value)
    raise RuntimeError("real SAVE model does not expose a DINO call counter")


def write_profile(path: str | Path, rows: list[dict[str, Any]], bindings: Mapping[str, str]) -> dict[str, Any]:
    try:
        chosen = choose_fastest_stable(rows)
        passed = True
        failure = None
    except RuntimeError as exc:
        chosen, passed, failure = None, False, str(exc)
    payload = {
        "pass": passed,
        "bindings": dict(bindings),
        "candidates": rows,
        "chosen": chosen,
        "failure": failure,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from fate_oia.engine.train_save_oia import build_save_runtime_for_profile

    runtime = build_save_runtime_for_profile(args.config, torch.device(args.device))
    rows = []
    for batch, accum in SAVE_PROFILE_CANDIDATES:
        try:
            rows.append(
                profile_candidate(
                    runtime["model"], runtime["batch_factory"], batch_size=batch,
                    gradient_accumulation_steps=accum, device=torch.device(args.device),
                )
            )
        except torch.cuda.OutOfMemoryError:
            rows.append({
                "batch_size": batch, "gradient_accumulation_steps": accum,
                "real_dino": True, "bf16": True, "warmup_microbatches": 20,
                "measured_microbatches": 50, "ordinary_dino_calls_per_microbatch": 0,
                "core_paths": {name: True for name in _CORE_PATHS}, "finite": False,
                "oom": True, "reserved_gb": float("inf"), "samples_per_second": 0.0,
            })
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    payload = write_profile(
        Path(args.output_dir) / "SAVE_RUNTIME_PROFILE.json", rows, runtime["bindings"]
    )
    print(json.dumps(payload, indent=2), flush=True)
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
