from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.meter_dataset import METERDataset, fixed_meter_split_indices
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex
from fate_oia.engine.train_acpr_meter_oia import (
    _counterfactual_event,
    _loader,
    _losses,
    _make_optimizer,
)
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.optim.meter_meta_utility import METERMetaUtility
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import state_hash, write_json
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import fit_train_calib_deploy_theta


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_two_stage_profile_plan(
    candidates: list[tuple[int, int]],
    worker_options: list[int],
    prefetch_options: list[int],
    *,
    stable_num_workers: int,
    stable_prefetch_factor: int,
    selected_candidate: tuple[int, int] | None = None,
) -> list[dict[str, int | str]]:
    """Build a staged search without repeating the full Cartesian product."""
    plan = [
        {
            "stage": "batch_search",
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(accumulation),
            "num_workers": int(stable_num_workers),
            "prefetch_factor": int(stable_prefetch_factor),
        }
        for batch_size, accumulation in candidates
    ]
    if selected_candidate is None:
        return plan
    selected_batch, selected_accumulation = selected_candidate
    for num_workers in worker_options:
        for prefetch_factor in prefetch_options:
            if (
                num_workers == stable_num_workers
                and prefetch_factor == stable_prefetch_factor
            ):
                continue
            plan.append({
                "stage": "loader_search",
                "batch_size": int(selected_batch),
                "gradient_accumulation_steps": int(selected_accumulation),
                "num_workers": int(num_workers),
                "prefetch_factor": int(prefetch_factor),
            })
    return plan


def _select_profile(
    profiles: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    valid = [
        profile
        for profile in profiles
        if not profile.get("oom")
        and profile.get("finite")
        and profile.get("reserved_gb", 1e9)
        < float(config["runtime"]["hard_max_reserved_gb"])
    ]
    preferred = [
        profile
        for profile in valid
        if profile["reserved_gb"] <= float(config["runtime"]["target_reserved_gb"])
    ] or valid
    if not preferred:
        return None
    best_speed = max(profile["event_adjusted_samples_per_sec"] for profile in preferred)
    near_best = [
        profile
        for profile in preferred
        if profile["event_adjusted_samples_per_sec"] >= 0.97 * best_speed
    ]
    return min(near_best, key=lambda profile: profile["reserved_gb"])


def profile_one(
    config: dict[str, Any],
    dataset: METERDataset,
    indices: list[int],
    device: torch.device,
    *,
    batch_size: int,
    accumulation: int,
    num_workers: int,
    prefetch_factor: int,
    use_mock_dino: bool,
    warmup_updates: int = 5,
    measured_updates: int = 20,
) -> dict[str, Any]:
    trial = copy.deepcopy(config)
    trial["training"]["batch_size"] = int(batch_size)
    trial["training"]["gradient_accumulation_steps"] = int(accumulation)
    trial["data"]["num_workers"] = int(num_workers)
    trial["data"]["prefetch_factor"] = int(prefetch_factor)
    loader = _loader(dataset, indices, trial, shuffle=True)
    iterator = iter(loader)
    model = METEROIAModel(
        dim=trial["model"]["dim"],
        action_dim=trial["model"]["action_dim"],
        reason_dim=trial["model"]["reason_dim"],
        selected_layers=tuple(trial["backbone"]["selected_layers"]),
        pretrained_weights=trial["backbone"]["pretrained_weights"],
        use_mock_dino=use_mock_dino,
        factor_rank=trial["model"].get("factor_rank", 16),
    ).to(device).train()
    model.foundation.dino.eval()
    optimizer = _make_optimizer(model, trial)
    pu_lambda = torch.zeros(trial["model"]["reason_dim"], device=device)
    amp_enabled = bool(trial["training"].get("bf16", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    update_durations: list[float] = []
    data_durations: list[float] = []
    finite = True
    last_field: dict[str, Any] | None = None
    last_output: dict[str, Any] | None = None
    last_batch: dict[str, Any] | None = None
    total_updates = warmup_updates + measured_updates
    for update in range(total_updates):
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            data_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            data_durations.append(time.perf_counter() - data_started)
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                field = model.encode_images(images)
                output = model.decode_from_field(field, progress=1.0)
                total, _, _ = _losses(
                    model,
                    output,
                    batch,
                    trial,
                    1.0,
                    device,
                    pu_lambda=pu_lambda,
                )
            finite = finite and bool(torch.isfinite(total))
            (total / accumulation).backward()
            last_field, last_output, last_batch = field, output, batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(trial["training"].get("grad_clip", 1.0)))
        optimizer.step()
        _synchronize(device)
        if update >= warmup_updates:
            update_durations.append(time.perf_counter() - update_started)
    measured_seconds = sum(update_durations)
    measured_samples = measured_updates * batch_size * accumulation

    counterfactual_seconds = 0.0
    counterfactual_valid = 0
    if last_field is not None and last_output is not None and last_batch is not None:
        _synchronize(device)
        event_started = time.perf_counter()
        event = _counterfactual_event(
            model,
            last_field,
            last_output,
            1.0,
            action_target=last_batch["action"].to(device),
            selected_mass=float(trial["counterfactual"].get("selected_mass", 0.60)),
            max_patches=int(trial["counterfactual"].get("max_patches", 128)),
            minimum_patches=int(trial["counterfactual"].get("minimum_patches", 4)),
        )
        _synchronize(device)
        counterfactual_seconds = time.perf_counter() - event_started
        counterfactual_valid = int(event.get("valid_count", 0))

    meta_event_seconds = 0.0
    meta_event_finite = False
    if last_field is not None and last_batch is not None:
        meta = METERMetaUtility(
            factors=trial["model"]["reason_dim"],
            virtual_lr=trial["meta"]["virtual_lr"],
            ema_old_weight=trial["meta"]["ema_old_weight"],
            ema_new_weight=trial["meta"]["ema_new_weight"],
            lower=trial["meta"]["utility_lower"],
            upper=trial["meta"]["utility_upper"],
        )
        parameter_map = {
            "down": model.signed_factors.meta_adapters.down,
            "up": model.signed_factors.meta_adapters.up,
        }
        action_target = last_batch["action"].to(device)
        reason_target = last_batch["reason"].to(device)

        def action_loss_fn(candidate: dict[str, torch.Tensor]) -> torch.Tensor:
            output = model.decode_from_field(
                last_field,
                progress=1.0,
                factor_parameter_override=candidate,
            )
            return meter_action_loss(output, action_target, trial["loss_weights"])["total"]

        def reason_loss_fn(candidate: dict[str, torch.Tensor]) -> torch.Tensor:
            output = model.decode_from_field(
                last_field,
                progress=1.0,
                factor_parameter_override=candidate,
                meta_share_weight_override=torch.ones(trial["model"]["reason_dim"], device=device),
            )
            return meter_reason_loss(
                output,
                reason_target,
                output["factor_reliability"],
                trial["loss_weights"],
            )["total"]

        _synchronize(device)
        meta_started = time.perf_counter()
        meta_event = meta.event(
            parameter_map,
            factor_ids=(0, 1, 2, 3),
            dino_calls=1,
            action_loss_fn=action_loss_fn,
            reason_loss_fn=reason_loss_fn,
            audit_action_loss_fn=action_loss_fn,
        )
        _synchronize(device)
        meta_event_seconds = time.perf_counter() - meta_started
        meta_event_finite = bool(torch.isfinite(meta_event.relative_utility).all())

    calibration_started = time.perf_counter()
    if last_output is not None and last_batch is not None:
        fit_train_calib_deploy_theta(
            last_output["action_logits_final"].detach(),
            last_batch["action"].to(device),
            model_state_hash=state_hash(model),
            label_groups=(0, 0, 1, 1),
        )
    calibration_seconds = time.perf_counter() - calibration_started
    updates_per_epoch = max(1.0, len(indices) / max(batch_size * accumulation, 1))
    amortized_event_seconds = (
        counterfactual_seconds
        * measured_updates
        / max(float(trial["counterfactual"].get("every_optimizer_updates", 8)), 1.0)
        + meta_event_seconds
        * measured_updates
        / max(float(trial["meta"].get("full_every_updates", 100)), 1.0)
        + calibration_seconds * measured_updates / updates_per_epoch
    )
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch": batch_size * accumulation,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "warmup_optimizer_updates": warmup_updates,
        "measured_optimizer_updates": measured_updates,
        "samples_per_sec": measured_samples / max(measured_seconds, 1e-9),
        "event_adjusted_samples_per_sec": measured_samples / max(measured_seconds + amortized_event_seconds, 1e-9),
        "mean_optimizer_update_sec": measured_seconds / max(measured_updates, 1),
        "mean_data_time_sec": sum(data_durations) / max(len(data_durations), 1),
        "counterfactual_event_sec": counterfactual_seconds,
        "counterfactual_valid_count": counterfactual_valid,
        "meta_utility_event_sec": meta_event_seconds,
        "meta_utility_finite": meta_event_finite,
        "posthoc_calibration_sec": calibration_seconds,
        "reserved_gb": reserved,
        "allocated_gb": allocated,
        "ordinary_dino_calls": model.foundation.ordinary_dino_calls,
        "finite": finite,
        "real_data": True,
        "real_dino": not use_mock_dino,
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
        torch.backends.cudnn.allow_tf32 = bool(config["training"].get("tf32", True))
    grounding_index = METERGroundingIndex(
        config["data"]["bdd100k_root"],
        schema_path="configs/meter_factor_schema.yaml",
    )
    dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="train",
        transform=meter_image_transform(),
        grounding_index=grounding_index,
        include_grounding=True,
    )
    split = fixed_meter_split_indices(
        [sample.file_name for sample in dataset.base.samples],
        audit_fraction=config["splits"]["audit_fraction"],
        calib_fraction=config["splits"]["calib_fraction"],
        seed=config["splits"]["seed"],
    )
    profiles = []
    candidates = [tuple(item) for item in config["runtime"]["profile_candidates"]]
    worker_options = [int(item) for item in config["runtime"]["profile_num_workers"]]
    prefetch_options = [int(item) for item in config["runtime"]["profile_prefetch_factor"]]
    stable_num_workers = int(config["data"].get("num_workers", worker_options[0]))
    stable_prefetch_factor = int(
        config["data"].get("prefetch_factor", prefetch_options[0])
    )

    def run_plan(items: list[dict[str, int | str]]) -> None:
        for item in items:
            try:
                profile = profile_one(
                    config,
                    dataset,
                    split["main"],
                    device,
                    batch_size=int(item["batch_size"]),
                    accumulation=int(item["gradient_accumulation_steps"]),
                    num_workers=int(item["num_workers"]),
                    prefetch_factor=int(item["prefetch_factor"]),
                    use_mock_dino=args.use_mock_dino,
                )
                profile["search_stage"] = item["stage"]
                profiles.append(profile)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                profiles.append({
                    **item,
                    "search_stage": item["stage"],
                    "oom": True,
                    "error": str(exc),
                })
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    batch_plan = build_two_stage_profile_plan(
        candidates,
        worker_options,
        prefetch_options,
        stable_num_workers=stable_num_workers,
        stable_prefetch_factor=stable_prefetch_factor,
    )
    run_plan(batch_plan)
    selected_batch_profile = _select_profile(profiles, config)
    if selected_batch_profile is not None:
        selected_candidate = (
            int(selected_batch_profile["batch_size"]),
            int(selected_batch_profile["gradient_accumulation_steps"]),
        )
        complete_plan = build_two_stage_profile_plan(
            candidates,
            worker_options,
            prefetch_options,
            stable_num_workers=stable_num_workers,
            stable_prefetch_factor=stable_prefetch_factor,
            selected_candidate=selected_candidate,
        )
        run_plan(complete_plan[len(batch_plan):])
    selected = _select_profile(profiles, config)
    report = {
        "real_dino": not args.use_mock_dino,
        "real_data": True,
        "device": str(device),
        "profiles": profiles,
        "selected": selected,
        "search_strategy": "two_stage_batch_then_loader",
        "selection_rule": "finite; reserved<45GB; prefer<=42GB; highest event-adjusted throughput; lower memory within 3%",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "runtime_profile.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit("No finite real-data profile under hard memory limit")


if __name__ == "__main__":
    main()
