from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset, tida_video_collate
from fate_oia.engine.evaluate_tida_oia import (
    branch_metrics,
    collect_tida_outputs,
    dynamic_slice_metrics,
    fit_train_calib_thresholds,
    save_epoch_outputs,
)
from fate_oia.engine.train_aie_oia import build_model as build_aie_model, canonical_model_state_dict
from fate_oia.engine.train_vetra_strong_refine import build_refiner
from fate_oia.losses.tida_loss_registry import assert_owner_exact_cover
from fate_oia.losses.tida_losses import build_tida_loss_registry, reason_pu_weight
from fate_oia.models.tida_oia_model import TIDAFrozenVETRAImageBase, TIDAOIAModel
from fate_oia.utils.tida_artifacts import (
    TIDATrainableEMA,
    append_jsonl,
    atomic_write_json,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    save_checkpoint_atomic,
)
from fate_oia.utils.tida_contracts import schedule_values, validate_training_protocol
from fate_oia.utils.vetra_stage_contracts import sha256_file


@dataclass
class TIDARuntime:
    config: dict[str, Any]
    model: TIDAOIAModel
    loaders: dict[str, DataLoader]
    device: torch.device
    image_checkpoint: Path
    clip_manifest: Path


def append_rank_window(
    window: dict[str, list[torch.Tensor]],
    action_logits: torch.Tensor,
    action_target: torch.Tensor,
    reason_logits: torch.Tensor,
    reason_target: torch.Tensor,
    reason_negative_weight: torch.Tensor,
) -> None:
    values = {
        "action_logits": action_logits,
        "action_target": action_target,
        "reason_logits": reason_logits,
        "reason_target": reason_target,
        "reason_negative_weight": reason_negative_weight,
    }
    for key, value in values.items():
        window.setdefault(key, []).append(value.detach())


def rank_window_reference(window: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    if not window.get("action_logits"):
        return None
    return {key: torch.cat(values, dim=0).detach() for key, values in window.items()}


def clear_rank_window(window: dict[str, list[torch.Tensor]]) -> None:
    window.clear()


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    validate_training_protocol(config)
    return config


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, config: dict[str, Any], generator=None) -> DataLoader:
    data = config["data"]
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(workers),
        "pin_memory": bool(data["pin_memory"]),
        "persistent_workers": bool(data["persistent_workers"]) and workers > 0,
        "collate_fn": tida_video_collate,
        "worker_init_fn": _worker_init,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data["prefetch_factor"])
    return DataLoader(**kwargs)


def _checkpoint_scales(checkpoint: dict[str, Any], image_config: dict[str, Any]) -> tuple[float, float]:
    scales = checkpoint.get("inference_scales") or {}
    return float(scales.get("action", 1.0)), float(scales.get("reason", image_config["reason_private"]["reason_scale_max"]))


def load_frozen_vetra_base(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> TIDAFrozenVETRAImageBase:
    image_config = yaml.safe_load(Path(config["image_base"]["config"]).read_text(encoding="utf-8"))
    stage = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stage_b = stage if stage.get("stage") == "action_refined" else None
    parent_path = Path(stage_b["parent_checkpoint"]) if stage_b is not None else checkpoint_path
    if not parent_path.is_file():
        raise FileNotFoundError(f"VETRA parent checkpoint does not exist: {parent_path}")
    if stage_b is not None and sha256_file(parent_path) != stage_b["parent_checkpoint_sha256"]:
        raise RuntimeError("VETRA Stage-B parent hash mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if "model" not in parent:
        raise RuntimeError("VETRA image checkpoint parent has no model state")
    base = build_aie_model(image_config, device)
    base.load_state_dict(canonical_model_state_dict(parent["model"]), strict=True)
    action_scale, reason_scale = _checkpoint_scales(parent, image_config)
    refiner = None
    if stage_b is not None and bool(stage_b.get("refiner_selected")):
        refiner = build_refiner(base, image_config).to(device)
        refiner.load_state_dict(stage_b["refiner"], strict=True)
        refiner.set_deployment_gain(stage_b["deployment_gain"].to(device))
    return TIDAFrozenVETRAImageBase(
        base, refiner, action_scale=action_scale, reason_scale=reason_scale
    ).to(device)


def _arg(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name.replace("-", "_"), default)


def build_runtime(args: Any, evaluation_only: bool = False) -> TIDARuntime:
    config = load_config(_arg(args, "config"))
    device = torch.device(_arg(args, "device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    clip_manifest = Path(_arg(args, "clip_manifest") or config["data"]["manifest_path"])
    image_checkpoint = Path(_arg(args, "image_checkpoint") or config["image_base"]["checkpoint"])
    if not clip_manifest.is_file() or not image_checkpoint.is_file():
        raise FileNotFoundError("--clip-manifest and --image-checkpoint must point to existing files")
    image_base = load_frozen_vetra_base(config, image_checkpoint, device)
    model = TIDAOIAModel(
        image_base,
        dim=int(config["model"]["dim"]),
        context_chunk_size=int(_arg(args, "context_chunk_size", config["model"]["context_chunk_size"])),
    ).to(device)
    batch_size = int(_arg(args, "batch_size", 2))
    workers = int(_arg(args, "num_workers", config["data"]["num_workers"]))
    train_generator = torch.Generator().manual_seed(20260821)
    datasets = {
        partition: BDDOIAVideoDataset(
            clip_manifest,
            partition,
            training=partition == "train_core" and not evaluation_only,
            seed=20260821,
        )
        for partition in ("train_core", "train_calib", "train_audit", "test")
    }
    max_samples = _arg(args, "max_samples", None)
    if max_samples is not None:
        from torch.utils.data import Subset

        datasets = {key: Subset(value, range(min(len(value), int(max_samples)))) for key, value in datasets.items()}
    loaders = {
        partition: make_loader(
            dataset,
            batch_size,
            shuffle=partition == "train_core" and not evaluation_only,
            workers=workers,
            config=config,
            generator=train_generator if partition == "train_core" else None,
        )
        for partition, dataset in datasets.items()
    }
    checkpoint = _arg(args, "checkpoint", None)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainable = dict(model.named_parameters())
        for name, value in payload["tida_trainable_state"].items():
            trainable[name].data.copy_(value.to(trainable[name]))
    return TIDARuntime(config, model, loaders, device, image_checkpoint, clip_manifest)


def build_optimizer(model: TIDAOIAModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    owners = model.owner_parameters()
    assert_owner_exact_cover(model, owners)
    groups = []
    for owner, parameters in owners.items():
        lr = float(config["training"]["lr"][owner])
        groups.append({"params": parameters, "lr": lr, "base_lr": lr, "name": owner})
    return torch.optim.AdamW(groups, weight_decay=float(config["training"]["weight_decay"]))


def _trainable_state(model: TIDAOIAModel) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}


def _counterfactual_errors(model: TIDAOIAModel, output: dict[str, Any], update: int, scale: float) -> dict[str, torch.Tensor]:
    if update % 4:
        return {}
    order_name = "time_shuffle" if (update // 4) % 2 == 0 else "time_reverse"
    order = model.rerun_temporal_from_output(
        output, order_name, temporal_action_scale=scale, temporal_reason_scale=scale
    )
    repeat = model.rerun_temporal_from_output(
        output, "repeated_last", temporal_action_scale=scale, temporal_reason_scale=scale
    )
    return {"order": order["terminal_error_history"], "repeat": repeat["terminal_error_history"]}


def _apply_initial_owner_firewall(model: TIDAOIAModel, temporal_scale: float) -> None:
    if float(temporal_scale) != 0.0:
        return
    owners = model.owner_parameters()
    for owner in ("temporal_action", "temporal_reason"):
        for parameter in owners[owner]:
            parameter.grad = None


def _checkpoint_payload(
    runtime: TIDARuntime,
    optimizer: torch.optim.Optimizer,
    ema: TIDATrainableEMA,
    epoch: int,
    update: int,
    best: dict[str, Any],
    train_generator: torch.Generator,
    *,
    total_updates: int,
    git_head: str,
    git_tree: str,
    config_path: Path,
    predicate_role_path: Path,
) -> dict[str, Any]:
    trainable_state = _trainable_state(runtime.model)
    return {
        "epoch": epoch,
        "global_update": update,
        "optimizer_update": update,
        "model": trainable_state,
        "tida_trainable_state": trainable_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": {"kind": "warmup_cosine_by_update", "update": update, "total_updates": total_updates},
        "ema": ema.state_dict(),
        "rng_state": capture_rng_state(),
        "train_generator_state": train_generator.get_state(),
        "sampler_state": {"epoch": epoch, "generator_state": train_generator.get_state()},
        "best": best,
        "git_head": git_head,
        "git_tree": git_tree,
        "config_sha256": file_sha256(config_path),
        "image_checkpoint": str(runtime.image_checkpoint.resolve()),
        "image_checkpoint_sha256": file_sha256(runtime.image_checkpoint),
        "clip_manifest": str(runtime.clip_manifest.resolve()),
        "clip_manifest_sha256": file_sha256(runtime.clip_manifest),
        "split_sha256": file_sha256(runtime.clip_manifest),
        "predicate_role_sha256": file_sha256(predicate_role_path),
    }


def _restore_training_state(
    runtime: TIDARuntime, optimizer, ema, path: Path, train_generator: torch.Generator,
    *, total_updates: int, config_path: Path, predicate_role_path: Path, git_head: str, git_tree: str,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "image_checkpoint_sha256": file_sha256(runtime.image_checkpoint),
        "clip_manifest_sha256": file_sha256(runtime.clip_manifest),
        "config_sha256": file_sha256(config_path), "predicate_role_sha256": file_sha256(predicate_role_path),
        "git_head": git_head, "git_tree": git_tree,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if payload.get("scheduler", {}).get("total_updates") != total_updates:
        mismatches.append("scheduler.total_updates")
    if mismatches:
        raise RuntimeError(f"resume run identity differs: {sorted(mismatches)}")
    named = dict(runtime.model.named_parameters())
    for name, value in payload["tida_trainable_state"].items():
        named[name].data.copy_(value.to(named[name]))
    optimizer.load_state_dict(payload["optimizer"])
    ema.load_state_dict(payload["ema"])
    restore_rng_state(payload["rng_state"])
    train_generator.set_state(payload["train_generator_state"])
    return int(payload["epoch"]) + 1, int(payload["optimizer_update"]), dict(payload["best"])


def _view_metrics(test_rows, calib_rows):
    thresholds = fit_train_calib_thresholds(calib_rows)
    raw = branch_metrics(test_rows)
    deploy = {
        "image": branch_metrics(test_rows, thresholds["image"])["image"],
        "video": branch_metrics(test_rows, thresholds["video"])["video"],
    }
    return {
        "raw_fixed": raw, "deploy": deploy, "thresholds": thresholds,
        "dynamic_slices": dynamic_slice_metrics(test_rows, thresholds["video"]),
    }


def completion_pass(
    run_kind: str,
    *,
    completed_epochs: int,
    optimizer_updates: int,
    max_optimizer_updates: int | None,
) -> bool:
    if run_kind == "full":
        return int(completed_epochs) == 10 and max_optimizer_updates is None
    return max_optimizer_updates is None or int(optimizer_updates) >= int(max_optimizer_updates)


def train(args: Any) -> None:
    runtime = build_runtime(args)
    config, model, device = runtime.config, runtime.model, runtime.device
    output_dir = Path(_arg(args, "output_dir")); output_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(_arg(args, "epochs", config["training"]["epochs"]))
    run_kind = _arg(args, "run_kind", "full")
    max_optimizer_updates = _arg(args, "max_optimizer_updates", None)
    if epochs != 10 and run_kind == "full":
        raise ValueError("formal full training requires exactly ten epochs")
    if run_kind == "full" and max_optimizer_updates is not None:
        raise ValueError("formal full training forbids max_optimizer_updates")
    grad_accum = int(_arg(args, "gradient_accumulation_steps", 15))
    optimizer = build_optimizer(model, config)
    ema = TIDATrainableEMA(model, decay=float(config["training"]["ema_decay"]))
    updates_per_epoch = math.ceil(len(runtime.loaders["train_core"]) / grad_accum)
    total_updates = max(int(max_optimizer_updates or (updates_per_epoch * epochs)), 1)
    train_generator = runtime.loaders["train_core"].generator
    start_epoch, optimizer_update, best = 0, 0, {}
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    config_path = Path(_arg(args, "config")).resolve()
    predicate_role_path = Path("configs/tida_predicate_roles.yaml").resolve()
    resume = _arg(args, "resume", None)
    if resume:
        start_epoch, optimizer_update, best = _restore_training_state(
            runtime, optimizer, ema, Path(resume), train_generator,
            total_updates=total_updates, config_path=config_path, predicate_role_path=predicate_role_path,
            git_head=git_head, git_tree=git_tree,
        )
    manifest = {
        "run_kind": run_kind, "git_head": git_head, "git_tree": git_tree,
        "config": str(Path(_arg(args, "config")).resolve()), "clip_manifest": str(runtime.clip_manifest.resolve()),
        "clip_manifest_sha256": file_sha256(runtime.clip_manifest), "image_checkpoint": str(runtime.image_checkpoint.resolve()),
        "image_checkpoint_sha256": file_sha256(runtime.image_checkpoint), "test_only_evaluation": True,
        "best_selection_split": "test", "feature_cache_enabled": False, "token_compression": "none",
        "foreground_only": True, "epochs": epochs, "gradient_accumulation_steps": grad_accum,
        "batch_size": int(_arg(args, "batch_size", 2)),
        "context_chunk_size": int(_arg(args, "context_chunk_size", config["model"]["context_chunk_size"])),
        "num_workers": int(_arg(args, "num_workers", config["data"]["num_workers"])),
        "command_line": [sys.executable, *sys.argv], "precision": config["training"]["precision"],
        "predicate_role_sha256": file_sha256(predicate_role_path), "config_sha256": file_sha256(config_path),
        "split_sha256": file_sha256(runtime.clip_manifest), "selected_layers": config["backbone"]["selected_layers"],
        "loss_weights": config["loss"], "learning_rates": config["training"]["lr"],
        "max_optimizer_updates": max_optimizer_updates,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    Path(output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_write_json(output_dir / "owner_map.json", {name: [id(parameter) for parameter in values] for name, values in model.owner_parameters().items()})

    baseline_path = output_dir / "TIDA_IMAGE_BASELINE_COVERED_SUBSET.json"
    if start_epoch == 0 and not baseline_path.exists():
        model.eval()
        baseline_calib_rows = collect_tida_outputs(model, runtime.loaders["train_calib"], device, temporal_scale=0.0)
        baseline_test_rows = collect_tida_outputs(model, runtime.loaders["test"], device, temporal_scale=0.0)
        baseline_thresholds = fit_train_calib_thresholds(baseline_calib_rows)
        atomic_write_json(baseline_path, {
            "pass": True, "covered_test_count": len(baseline_test_rows["file_names"]),
            "raw_fixed": branch_metrics(baseline_test_rows)["image"],
            "deploy": branch_metrics(baseline_test_rows, baseline_thresholds["image"])["image"],
            "threshold_fit_split": "train_calib", "test_labels_used_for_parameters": False,
        })

    scaler_enabled = device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rank_window: dict[str, list[torch.Tensor]] = {}
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        micro_count = 0
        for micro_step, batch in enumerate(runtime.loaders["train_core"]):
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            schedule = schedule_values(
                optimizer_update, total_updates,
                warmup_ratio=float(config["training"]["warmup_ratio"]),
                ramp_end_ratio=float(config["training"]["temporal_ramp_end_ratio"]),
                min_lr_ratio=float(config["training"]["min_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * schedule["lr_scale"]
            data_end = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
                output = model(
                    batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
                    temporal_action_scale=schedule["temporal_scale"], temporal_reason_scale=schedule["temporal_scale"],
                )
                counterfactual = _counterfactual_errors(model, output, optimizer_update, schedule["temporal_scale"])
                registry = build_tida_loss_registry(
                    output,
                    batch["action"],
                    batch["reason"],
                    counterfactual_errors=counterfactual,
                    rank_reference=rank_window_reference(rank_window),
                    weights=config["loss"],
                )
                loss = registry.total() / grad_accum
            loss.backward()
            _apply_initial_owner_firewall(model, schedule["temporal_scale"])
            contradiction = output.get("image_branch", {}).get("contradiction_score")
            append_rank_window(
                rank_window,
                output["video_action_logits"],
                batch["action"],
                output["video_reason_logits"],
                batch["reason"],
                reason_pu_weight(batch["reason"], contradiction),
            )
            micro_count += 1
            should_update = micro_count == grad_accum or micro_step + 1 == len(runtime.loaders["train_core"])
            if not should_update:
                continue
            if micro_count != grad_accum:
                correction = grad_accum / micro_count
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["training"]["global_grad_clip"]),
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model)
            optimizer_update += 1; micro_count = 0
            rank_window_samples = sum(value.shape[0] for value in rank_window.get("action_logits", []))
            row = {
                "epoch": epoch, "micro_step": micro_step, "optimizer_update": optimizer_update,
                "total_updates": total_updates, "temporal_scale": schedule["temporal_scale"],
                "initial_owner_firewall_active": schedule["temporal_scale"] == 0.0,
                "lr_scale": schedule["lr_scale"], "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "loss_total": float(registry.total().detach().cpu()), "losses": registry.artifact(),
                "grad_norm": float(grad_norm.detach().cpu()),
                "rho_mean": float(output["innovation_reliability"].mean().detach().cpu()),
                "rho_nonzero_rate": float((output["innovation_reliability"] > 0).float().mean().detach().cpu()),
                "action_delta_rms": float(output["action_temporal_delta"].float().square().mean().sqrt().detach().cpu()),
                "reason_delta_rms": float(output["reason_temporal_delta"].float().square().mean().sqrt().detach().cpu()),
                "action_null_mass": float(output["action_null_mass"].mean().detach().cpu()),
                "rank_window_samples": int(rank_window_samples),
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
            }
            append_jsonl(output_dir / "loss_components.jsonl", row)
            if optimizer_update % int(config["runtime"]["print_every_optimizer_updates"]) == 0:
                print(json.dumps({"event": "tida_batch", **row}, ensure_ascii=False), flush=True)
            clear_rank_window(rank_window)
            if max_optimizer_updates is not None and optimizer_update >= int(max_optimizer_updates):
                break

        model.eval()
        online_calib = collect_tida_outputs(model, runtime.loaders["train_calib"], device)
        online_test = collect_tida_outputs(
            model, runtime.loaders["test"], device,
            collect_mechanism=True, mechanism_samples=int(config["runtime"]["fixed_test_audit_samples"]),
            collect_audit_tensors=True,
        )
        online = _view_metrics(online_test, online_calib)
        mechanism = online_test.pop("_mechanism")
        with ema.average_parameters(model):
            ema_calib = collect_tida_outputs(model, runtime.loaders["train_calib"], device)
            ema_test = collect_tida_outputs(model, runtime.loaders["test"], device)
            ema_metrics = _view_metrics(ema_test, ema_calib)
        metrics = {"epoch": epoch, "online": online, "ema": ema_metrics, "mechanism": mechanism, "epoch_seconds": time.perf_counter() - epoch_start}
        append_jsonl(output_dir / "metrics_summary.jsonl", metrics)
        selected_name, selected = max(
            (("online", online), ("ema", ema_metrics)),
            key=lambda pair: float(pair[1]["deploy"]["video"]["joint"]),
        )
        payload = _checkpoint_payload(
            runtime, optimizer, ema, epoch, optimizer_update, best, train_generator,
            total_updates=total_updates, git_head=git_head, git_tree=git_tree,
            config_path=config_path, predicate_role_path=predicate_role_path,
        )
        payload.update({"metrics": metrics, "selected_view": selected_name})
        checkpoint_epoch = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        criteria = {
            "joint": float(selected["deploy"]["video"]["joint"]),
            "action_mf1": float(selected["deploy"]["video"]["Act_mF1"]),
            "action_map": float(selected["raw_fixed"]["video"]["Act_mAP"]),
            "exp_mf1": float(selected["deploy"]["video"]["Exp_mF1"]),
            "exp_map": float(selected["raw_fixed"]["video"]["Exp_mAP"]),
        }
        checkpoint_names = {
            "joint": "checkpoint_best_test_joint.pth", "action_mf1": "checkpoint_best_test_action_mf1.pth",
            "action_map": "checkpoint_best_test_action_map.pth", "exp_mf1": "checkpoint_best_test_exp_mf1.pth",
            "exp_map": "checkpoint_best_test_exp_map.pth",
        }
        for key, value in criteria.items():
            if value > float(best.get(key, {}).get("value", float("-inf"))):
                best[key] = {"value": value, "epoch": epoch, "view": selected_name}
                payload["best"] = best
                save_checkpoint_atomic(output_dir / checkpoint_names[key], payload)
        payload["best"] = best
        save_checkpoint_atomic(checkpoint_epoch, payload)
        save_checkpoint_atomic(output_dir / "checkpoint_latest.pth", payload)
        atomic_write_json(output_dir / "best_epoch_source.json", best)
        save_epoch_outputs(output_dir, epoch, online_test, metrics, online["thresholds"], mechanism)
        print(json.dumps({"event": "tida_epoch", "epoch": epoch, "selected_view": selected_name, **criteria}), flush=True)
        model.train()
        completed_epochs = epoch + 1
        if max_optimizer_updates is not None and optimizer_update >= int(max_optimizer_updates):
            break

    completion_name = "TRAIN_COMPLETED_TIDA_OIA_V1.json" if run_kind == "full" else "SMOKE_COMPLETED_TIDA_OIA_V1.json"
    atomic_write_json(output_dir / completion_name, {
        "pass": completion_pass(
            run_kind,
            completed_epochs=completed_epochs,
            optimizer_updates=optimizer_update,
            max_optimizer_updates=max_optimizer_updates,
        ),
        "run_kind": run_kind, "epochs_requested": epochs, "epochs_completed": completed_epochs,
        "optimizer_updates": optimizer_update, "best": best, "git_head": git_head, "git_tree": git_tree,
        "clip_manifest_sha256": file_sha256(runtime.clip_manifest), "image_checkpoint_sha256": file_sha256(runtime.image_checkpoint),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=15)
    parser.add_argument("--context-chunk-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-kind", choices=("smoke", "profile", "full"), default="full")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-optimizer-updates", type=int)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
