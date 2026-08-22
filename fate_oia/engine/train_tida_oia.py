from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
from fate_oia.losses.tida_losses import build_tida_loss_registry
from fate_oia.models.tida_oia_model import TIDAFrozenVETRAImageBase, TIDAOIAModel
from fate_oia.models.tida_predicate_differential import ROLE_NAMES
from fate_oia.utils.tida_artifacts import (
    TIDATrainableEMA,
    append_jsonl,
    atomic_write_json,
    capture_rng_state,
    file_sha256,
    restore_rng_state,
    save_checkpoint_atomic,
    seed_tida_run,
)
from fate_oia.utils.tida_contracts import (
    resolve_schedule_total_updates,
    schedule_values,
    validate_training_protocol,
)
from fate_oia.utils.tida_stateful_sampler import TIDAStatefulRandomSampler
from fate_oia.utils.vetra_stage_contracts import sha256_file


@dataclass
class TIDARuntime:
    config: dict[str, Any]
    model: TIDAOIAModel
    loaders: dict[str, DataLoader]
    device: torch.device
    image_checkpoint: Path
    clip_manifest: Path
    train_sampler: TIDAStatefulRandomSampler


def append_rank_window(
    window: dict[str, list[torch.Tensor]],
    action_logits: torch.Tensor,
    action_target: torch.Tensor,
) -> None:
    values = {
        "action_logits": action_logits,
        "action_target": action_target,
    }
    for key, value in values.items():
        window.setdefault(key, []).append(value.detach())


def rank_window_reference(window: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor] | None:
    if not window.get("action_logits"):
        return None
    return {key: torch.cat(values, dim=0).detach() for key, values in window.items()}


def clear_rank_window(window: dict[str, list[torch.Tensor]]) -> None:
    window.clear()


@torch.no_grad()
def owner_gradient_norms(owners: dict[str, list[torch.nn.Parameter]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for owner, parameters in owners.items():
        squared = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared = value if squared is None else squared + value
        result[owner] = 0.0 if squared is None else float(squared.sqrt().cpu())
    return result


@torch.no_grad()
def owner_parameter_snapshots(
    owners: dict[str, list[torch.nn.Parameter]],
) -> dict[str, list[torch.Tensor]]:
    return {
        owner: [parameter.detach().clone() for parameter in parameters]
        for owner, parameters in owners.items()
    }


@torch.no_grad()
def owner_parameter_update_norms(
    owners: dict[str, list[torch.nn.Parameter]],
    before: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    if set(owners) != set(before):
        raise ValueError("owner snapshot keys do not match current owners")
    for owner, parameters in owners.items():
        snapshots = before[owner]
        if len(parameters) != len(snapshots):
            raise ValueError(f"owner snapshot length changed: {owner}")
        squared = sum(
            (parameter.detach().float() - snapshot.to(parameter).float()).square().sum()
            for parameter, snapshot in zip(parameters, snapshots)
        )
        result[owner] = float(squared.sqrt().cpu())
    return result


@torch.no_grad()
def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().contiguous().view(torch.uint8).cpu()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(parameters, gradients) -> float:
    squared = None
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


class TIDAForwardTimer:
    """Collect synchronized stage timings only for selected supervision windows."""

    def __init__(self, model: TIDAOIAModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.active = False
        self.totals: dict[str, float] = {}
        self._starts: dict[str, list[float]] = {}
        self._context_query_before: list[float] = []
        self._forward_start = 0.0
        self._forward_before: dict[str, float] = {}
        dino = model.image_model.foundation.dino
        self.handles = [
            dino.register_forward_pre_hook(self._dino_pre),
            dino.register_forward_hook(self._dino_post),
            model.query_reader.register_forward_pre_hook(self._query_pre),
            model.query_reader.register_forward_hook(self._query_post),
            model.context_encoder.register_forward_pre_hook(self._context_pre),
            model.context_encoder.register_forward_hook(self._context_post),
        ]
        self.reset()

    def _sync(self) -> None:
        if self.active and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        self.totals = {
            "target_dino_time": 0.0,
            "context_dino_time": 0.0,
            "query_read_time": 0.0,
            "temporal_time": 0.0,
            "_context_total_time": 0.0,
        }
        self._starts.clear()
        self._context_query_before.clear()

    def _push(self, key: str) -> None:
        if not self.active:
            return
        self._sync()
        self._starts.setdefault(key, []).append(time.perf_counter())

    def _pop(self, key: str) -> None:
        if not self.active:
            return
        self._sync()
        self.totals[key] += time.perf_counter() - self._starts[key].pop()

    def _dino_pre(self, _module, inputs) -> None:
        self._push("target_dino_time")

    def _dino_post(self, _module, inputs, _output) -> None:
        self._pop("target_dino_time")

    def _query_pre(self, _module, _inputs) -> None:
        self._push("query_read_time")

    def _query_post(self, _module, _inputs, _output) -> None:
        self._pop("query_read_time")

    def _context_pre(self, _module, _inputs) -> None:
        if not self.active:
            return
        self._context_query_before.append(self.totals["query_read_time"])
        self._push("_context_total_time")

    def _context_post(self, _module, _inputs, _output) -> None:
        if not self.active:
            return
        before_total = self.totals["_context_total_time"]
        self._pop("_context_total_time")
        context_elapsed = self.totals["_context_total_time"] - before_total
        query_elapsed = self.totals["query_read_time"] - self._context_query_before.pop()
        self.totals["context_dino_time"] += max(0.0, context_elapsed - query_elapsed)

    def begin_forward(self) -> None:
        if not self.active:
            return
        self._sync()
        self._forward_before = dict(self.totals)
        self._forward_start = time.perf_counter()

    def end_forward(self) -> None:
        if not self.active:
            return
        self._sync()
        elapsed = time.perf_counter() - self._forward_start
        measured = sum(
            self.totals[key] - self._forward_before[key]
            for key in ("target_dino_time", "context_dino_time", "query_read_time")
        )
        self.totals["temporal_time"] += max(0.0, elapsed - measured)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def reason_firewall_gradient_audit(
    registry,
    model: TIDAOIAModel,
) -> dict[str, float]:
    reason_names = ("reason_partial", "reason_rank", "reason_soft_f1", "reason_delta")
    action_names = ("action_asl", "action_smooth_ap", "action_base_protect", "action_delta")
    reason_loss = sum(registry.rows[name].weight * registry.rows[name].value for name in reason_names)
    action_loss = sum(registry.rows[name].weight * registry.rows[name].value for name in action_names)
    action_parameters = list(model.action_reader.parameters())
    reason_parameters = list(model.reason_reader.parameters())
    reason_to_action = torch.autograd.grad(
        reason_loss, action_parameters, retain_graph=True, allow_unused=True
    )
    action_to_reason = torch.autograd.grad(
        action_loss, reason_parameters, retain_graph=True, allow_unused=True
    )
    return {
        "reason_loss_to_action_owner": gradient_norm(action_parameters, reason_to_action),
        "action_loss_to_reason_owner": gradient_norm(reason_parameters, action_to_reason),
    }


def append_supervision_tensors(
    store: dict[str, list[torch.Tensor]],
    output: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    action_sign = 2.0 * batch["action"].float() - 1.0
    values = {
        "rho": output["innovation_reliability"],
        "action_delta": output["action_temporal_delta"],
        "reason_delta": output["reason_temporal_delta"],
        "action_null_mass": output["action_null_mass"],
        "route_entropy": output["action_route_entropy"],
        "action_nonnull_mass": output["action_nonnull_mass"],
        "predicate_velocity": output["predicate_velocity_norm"],
        "predicate_acceleration": output["predicate_acceleration_norm"],
        "predicate_persistence": output["predicate_persistence"],
        "common_motion": output["common_motion_norm"],
        "region_mass_velocity": output["predicate_region_mass_velocity"],
        "terminal_history_error": output["terminal_error_history"],
        "terminal_no_history_error": output["terminal_error_no_history"],
        "history_frame_valid": output["frame_valid_mask"][:, :-1],
        "history_available": output["history_valid"],
        "image_action_margin": action_sign * output["image_action_logits"],
        "video_action_margin": action_sign * output["video_action_logits"],
    }
    for key, value in values.items():
        store.setdefault(key, []).append(value.detach().float().cpu())


def supervision_tensor_summary(
    store: dict[str, list[torch.Tensor]],
    predicate_role_ids: torch.Tensor,
) -> dict[str, Any]:
    values = {key: torch.cat(rows, dim=0) for key, rows in store.items()}
    rho = values["rho"]
    predicate_rho = rho[:, 4:]
    role_ids = predicate_role_ids.detach().cpu().long()
    rho_by_role = {
        role: float(predicate_rho[:, role_ids == role_index].mean())
        for role_index, role in enumerate(ROLE_NAMES)
    }
    image_margin = values["image_action_margin"]
    video_margin = values["video_action_margin"]
    nonnull = values["action_nonnull_mass"]
    return {
        "history_valid_rate": float(values["history_frame_valid"].mean()),
        "history_available_rate": float(values["history_available"].mean()),
        "terminal_history_error": float(values["terminal_history_error"].mean()),
        "terminal_no_history_error": float(values["terminal_no_history_error"].mean()),
        "reconstruction_gain": float(
            values["terminal_no_history_error"].mean() - values["terminal_history_error"].mean()
        ),
        "rho_quantiles": {
            "p10": float(torch.quantile(rho, 0.10)),
            "p50": float(torch.quantile(rho, 0.50)),
            "p90": float(torch.quantile(rho, 0.90)),
        },
        "rho_per_action": rho[:, :4].mean(0).tolist(),
        "rho_per_predicate_role": rho_by_role,
        "predicate_velocity_mean": float(values["predicate_velocity"].mean()),
        "predicate_acceleration_mean": float(values["predicate_acceleration"].mean()),
        "predicate_persistence_mean": float(values["predicate_persistence"].mean()),
        "common_motion_norm_mean": float(values["common_motion"].mean()),
        "region_mass_velocity_rms": float(values["region_mass_velocity"].square().mean().sqrt()),
        "action_delta_rms_window": float(values["action_delta"].square().mean().sqrt()),
        "reason_delta_rms_window": float(values["reason_delta"].square().mean().sqrt()),
        "action_null_mass_window": float(values["action_null_mass"].mean()),
        "route_entropy_window": float(values["route_entropy"].mean()),
        "per_action_route_coverage": (nonnull >= 0.01).float().mean(0).tolist(),
        "image_to_video_margin_inversion_rate": float(
            ((image_margin > 0) & (video_margin <= 0)).float().mean()
        ),
        "image_to_video_margin_recovery_rate": float(
            ((image_margin <= 0) & (video_margin > 0)).float().mean()
        ),
        "action_margin_delta_mean": float((video_margin - image_margin).mean()),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    validate_training_protocol(config)
    return config


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def make_loader(
    dataset, batch_size: int, shuffle: bool, workers: int, config: dict[str, Any],
    generator=None, sampler=None,
) -> DataLoader:
    data = config["data"]
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle) if sampler is None else False,
        "num_workers": int(workers),
        "pin_memory": bool(data["pin_memory"]),
        "persistent_workers": bool(data["persistent_workers"]) and workers > 0,
        "collate_fn": tida_video_collate,
        "worker_init_fn": _worker_init,
        "generator": generator,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
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
    seed = int(config["training"].get("seed", config["data"]["partition_seed"]))
    seed_tida_run(seed)
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
        action_evidence_trust_cap=float(config["model"].get("action_evidence_trust_cap", 0.25)),
        reason_evidence_trust_cap=float(config["model"].get("reason_evidence_trust_cap", 0.25)),
    ).to(device)
    batch_size = int(_arg(args, "batch_size", 2))
    workers = int(_arg(args, "num_workers", config["data"]["num_workers"]))
    max_samples = _arg(args, "max_samples", None)
    datasets = {
        partition: BDDOIAVideoDataset(
            clip_manifest,
            partition,
            training=partition == "train_core" and not evaluation_only,
            seed=20260821,
            max_samples=max_samples,
        )
        for partition in ("train_core", "train_calib", "train_audit", "test")
    }
    train_sampler = TIDAStatefulRandomSampler(datasets["train_core"], seed=20260821)
    loaders = {
        partition: make_loader(
            dataset,
            batch_size,
            shuffle=partition == "train_core" and not evaluation_only,
            workers=workers,
            config=config,
            sampler=train_sampler if partition == "train_core" and not evaluation_only else None,
        )
        for partition, dataset in datasets.items()
    }
    checkpoint = _arg(args, "checkpoint", None)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainable = dict(model.named_parameters())
        for name, value in payload["tida_trainable_state"].items():
            trainable[name].data.copy_(value.to(trainable[name]))
    return TIDARuntime(config, model, loaders, device, image_checkpoint, clip_manifest, train_sampler)


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


def _counterfactual_outputs(
    model: TIDAOIAModel,
    output: dict[str, Any],
    update: int,
    scale: float,
    *,
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    if not enabled:
        return {}, {}
    order_name = "time_shuffle" if update % 2 == 0 else "time_reverse"
    order = model.rerun_temporal_from_output(
        output, order_name, temporal_action_scale=scale, temporal_reason_scale=scale
    )
    repeat = model.rerun_temporal_from_output(
        output, "repeated_last", temporal_action_scale=scale, temporal_reason_scale=scale
    )
    history_off = model.rerun_temporal_from_output(
        output, "history_off", temporal_action_scale=scale, temporal_reason_scale=scale
    )
    return (
        {"order": order["terminal_error_history"], "repeat": repeat["terminal_error_history"]},
        {order_name: order, "repeated_last": repeat, "history_off": history_off},
    )


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
    train_sampler: TIDAStatefulRandomSampler,
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
        "sampler_state": train_sampler.state_dict(),
        "checkpoint_at_optimizer_boundary": True,
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
    runtime: TIDARuntime, optimizer, ema, path: Path,
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
    if not payload.get("checkpoint_at_optimizer_boundary", False):
        raise RuntimeError("resume checkpoint is not at an optimizer boundary")
    runtime.train_sampler.load_state_dict(payload["sampler_state"])
    return int(runtime.train_sampler.epoch), int(payload["optimizer_update"]), dict(payload["best"])


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
    schedule_override = _arg(args, "schedule_total_updates", None)
    if run_kind == "full" and schedule_override is not None:
        raise ValueError("formal full training forbids schedule_total_updates override")
    grad_accum = int(_arg(args, "gradient_accumulation_steps", 15))
    optimizer = build_optimizer(model, config)
    ema = TIDATrainableEMA(model, decay=float(config["training"]["ema_decay"]))
    updates_per_epoch = math.ceil(len(runtime.loaders["train_core"]) / grad_accum)
    total_updates = resolve_schedule_total_updates(
        updates_per_epoch=updates_per_epoch,
        configured_epochs=int(config["training"]["epochs"]),
        schedule_total_updates=schedule_override,
    )
    start_epoch, optimizer_update, best = 0, 0, {}
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    config_path = Path(_arg(args, "config")).resolve()
    predicate_role_path = Path("configs/tida_predicate_roles.yaml").resolve()
    resume = _arg(args, "resume", None)
    if resume:
        start_epoch, optimizer_update, best = _restore_training_state(
            runtime, optimizer, ema, Path(resume),
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
        "seed": int(config["training"].get("seed", config["data"]["partition_seed"])),
        "max_optimizer_updates": max_optimizer_updates,
        "schedule_total_updates": total_updates,
        "schedule_override": schedule_override is not None,
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
    forward_timer = TIDAForwardTimer(model, device)
    frozen_image_hash = module_state_sha256(model.image_model)
    print_interval = int(config["runtime"]["print_every_optimizer_updates"])
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        if runtime.train_sampler.epoch != epoch:
            raise RuntimeError("train sampler epoch and trainer epoch differ")
        epoch_start = time.perf_counter()
        epoch_batch_count = math.ceil(len(runtime.train_sampler) / int(_arg(args, "batch_size", 2)))
        micro_count = 0
        previous_iteration_end = time.perf_counter()
        telemetry_window = False
        telemetry_decode_time = 0.0
        telemetry_backward_time = 0.0
        telemetry_samples = 0
        telemetry_tensors: dict[str, list[torch.Tensor]] = {}
        firewall_gradients: dict[str, float] | None = None
        for micro_step, batch in enumerate(runtime.loaders["train_core"]):
            batch_arrival = time.perf_counter()
            if micro_count == 0:
                telemetry_window = optimizer_update == 0 or (optimizer_update + 1) % print_interval == 0
                forward_timer.active = telemetry_window
                forward_timer.reset()
                telemetry_decode_time = 0.0
                telemetry_backward_time = 0.0
                telemetry_samples = 0
                telemetry_tensors = {}
                firewall_gradients = None
            if telemetry_window:
                telemetry_decode_time += batch_arrival - previous_iteration_end
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            schedule = schedule_values(
                optimizer_update, total_updates,
                warmup_ratio=float(config["training"]["warmup_ratio"]),
                ramp_end_ratio=float(config["training"]["temporal_ramp_end_ratio"]),
                min_lr_ratio=float(config["training"]["min_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * schedule["lr_scale"]
            forward_timer.begin_forward()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
                output = model(
                    batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
                    temporal_action_scale=schedule["temporal_scale"], temporal_reason_scale=schedule["temporal_scale"],
                )
                counterfactual_enabled = micro_count + 1 == grad_accum or micro_step + 1 == epoch_batch_count
                counterfactual_errors, counterfactual_outputs = _counterfactual_outputs(
                    model, output, optimizer_update, schedule["temporal_scale"], enabled=counterfactual_enabled
                )
                registry = build_tida_loss_registry(
                    output,
                    batch["action"],
                    batch["reason"],
                    counterfactual_errors=counterfactual_errors,
                    counterfactual_outputs=counterfactual_outputs,
                    rank_reference=rank_window_reference(rank_window),
                    weights=config["loss"],
                )
                loss = registry.total() / grad_accum
            forward_timer.end_forward()
            if telemetry_window and firewall_gradients is None:
                firewall_gradients = reason_firewall_gradient_audit(registry, model)
            if telemetry_window and device.type == "cuda":
                torch.cuda.synchronize(device)
            backward_start = time.perf_counter()
            loss.backward()
            if telemetry_window and device.type == "cuda":
                torch.cuda.synchronize(device)
            if telemetry_window:
                telemetry_backward_time += time.perf_counter() - backward_start
                telemetry_samples += int(batch["target_image"].shape[0])
                append_supervision_tensors(telemetry_tensors, output, batch)
            _apply_initial_owner_firewall(model, schedule["temporal_scale"])
            append_rank_window(
                rank_window,
                output["video_action_logits"],
                batch["action"],
            )
            micro_count += 1
            runtime.train_sampler.mark_consumed(int(batch["target_image"].shape[0]))
            should_update = micro_count == grad_accum or micro_step + 1 == epoch_batch_count
            if not should_update:
                previous_iteration_end = time.perf_counter()
                continue
            if micro_count != grad_accum:
                correction = grad_accum / micro_count
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            owners = model.owner_parameters()
            owner_gradients = owner_gradient_norms(owners)
            owner_before = owner_parameter_snapshots(owners) if telemetry_window else None
            dino_parameters = list(model.image_model.foundation.dino.parameters())
            dino_gradient = gradient_norm(dino_parameters, [parameter.grad for parameter in dino_parameters])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config["training"]["global_grad_clip"]),
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True); ema.update(model)
            owner_updates = (
                owner_parameter_update_norms(owners, owner_before)
                if owner_before is not None else None
            )
            optimizer_update += 1; micro_count = 0
            rank_window_samples = sum(value.shape[0] for value in rank_window.get("action_logits", []))
            row = {
                "epoch": epoch, "micro_step": micro_step, "optimizer_update": optimizer_update,
                "total_updates": total_updates, "temporal_scale": schedule["temporal_scale"],
                "flow_phase": schedule["phase"],
                "initial_owner_firewall_active": schedule["temporal_scale"] == 0.0,
                "lr_scale": schedule["lr_scale"], "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "loss_total": float(registry.total().detach().cpu()), "losses": registry.artifact(),
                "grad_norm": float(grad_norm.detach().cpu()),
                "owner_gradient_norms": owner_gradients,
                "rho_mean": float(output["innovation_reliability"].mean().detach().cpu()),
                "rho_nonzero_rate": float((output["innovation_reliability"] > 0).float().mean().detach().cpu()),
                "action_delta_rms": float(output["action_temporal_delta"].float().square().mean().sqrt().detach().cpu()),
                "action_evidence_confidence_mean": float(output["action_evidence_confidence"].mean().detach().cpu()),
                "action_effective_trust_mean": float(output["action_effective_trust"].mean().detach().cpu()),
                "reason_delta_rms": float(output["reason_temporal_delta"].float().square().mean().sqrt().detach().cpu()),
                "reason_evidence_confidence_mean": float(output["reason_evidence_confidence"].mean().detach().cpu()),
                "reason_effective_trust_mean": float(output["reason_effective_trust"].mean().detach().cpu()),
                "action_null_mass": float(output["action_null_mass"].mean().detach().cpu()),
                "rank_window_samples": int(rank_window_samples),
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
            }
            if telemetry_window:
                stage_times = {
                    key: value for key, value in forward_timer.totals.items()
                    if not key.startswith("_")
                }
                stage_times.update({
                    "decode_time": telemetry_decode_time,
                    "backward_time": telemetry_backward_time,
                })
                elapsed = sum(stage_times.values())
                current_image_hash = module_state_sha256(model.image_model)
                row.update({
                    "supervision_telemetry_available": True,
                    "owner_parameter_update_norms": owner_updates,
                    "runtime_timing_seconds": stage_times,
                    "samples_per_second": telemetry_samples / max(elapsed, 1e-9),
                    "gpu_allocated_gib": torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                    "gpu_reserved_gib": torch.cuda.memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0,
                    "context_chunk_count": math.ceil(
                        int(config["data"]["history_frames"]) / model.context_encoder.context_chunk_size
                    ),
                    "reason_firewall_grad": firewall_gradients,
                    "dino_grad": dino_gradient,
                    "image_base_hash": current_image_hash,
                    "image_base_hash_delta": int(current_image_hash != frozen_image_hash),
                    **supervision_tensor_summary(
                        telemetry_tensors, output["predicate_role_ids"]
                    ),
                })
            else:
                row["supervision_telemetry_available"] = False
            append_jsonl(output_dir / "loss_components.jsonl", row)
            if optimizer_update % print_interval == 0:
                print(json.dumps({"event": "tida_batch", **row}, ensure_ascii=False), flush=True)
            forward_timer.active = False
            clear_rank_window(rank_window)
            previous_iteration_end = time.perf_counter()
            if max_optimizer_updates is not None and optimizer_update >= int(max_optimizer_updates):
                break

        epoch_exhausted = runtime.train_sampler.epoch_complete
        if epoch_exhausted:
            runtime.train_sampler.advance_epoch()
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
            runtime, optimizer, ema, epoch, optimizer_update, best, runtime.train_sampler,
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
        completed_epochs = epoch + 1 if epoch_exhausted else epoch
        if max_optimizer_updates is not None and optimizer_update >= int(max_optimizer_updates):
            break

    forward_timer.close()
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
    parser.add_argument("--schedule-total-updates", type=int)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
