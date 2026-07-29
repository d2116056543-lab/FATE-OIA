from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.meter_dataset import (
    METERDataset,
    fixed_meter_split_indices,
    meter_split_manifest,
)
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex
from fate_oia.engine.eval_acpr_meter_oia import (
    branch_metrics,
    collect_outputs,
    mechanism_stats_from_collected,
    metrics_summary,
)
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_counterfactual_losses import (
    dense_factor_intervention_loss,
    identity_corruption_loss,
)
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.losses.meter_pu_losses import (
    meter_hidden_positive_audit,
    meter_private_pu_loss,
    meter_pu_score,
)
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import (
    append_jsonl,
    combined_file_hash,
    file_hash,
    load_checkpoint,
    python_source_tree_hash,
    save_checkpoint,
    save_epoch_artifacts,
    state_hash,
    write_json,
)
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import (
    METERCalibrationResult,
    fit_train_calib_deploy_theta,
    guard_train_calib_deploy_theta,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def _loader(
    dataset: METERDataset,
    indices: list[int],
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    config: dict[str, Any],
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(config["data"].get("pin_memory", True)),
        "persistent_workers": workers > 0
        and bool(config["data"].get("persistent_workers", True)),
        "drop_last": bool(shuffle),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    return DataLoader(Subset(dataset, indices), **kwargs)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _parameter_groups(
    model: METEROIAModel, config: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Tensor]] = {
        "foundation": [],
        "typed_factor": [],
        "action_transport": [],
        "reason_global": [],
        "reason_correction": [],
    }
    owners: dict[int, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("foundation.dino."):
            continue
        if name.startswith("typed_factors."):
            owner = "typed_factor"
        elif name.startswith("action_transport."):
            owner = "action_transport"
        elif name.startswith("reason_decoder.correction_"):
            owner = "reason_correction"
        elif name.startswith("reason_decoder."):
            owner = "reason_global"
        else:
            owner = "foundation"
        if id(parameter) in owners:
            raise RuntimeError(f"Parameter {name} has duplicate optimizer ownership")
        owners[id(parameter)] = owner
        groups[owner].append(parameter)
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
        and all(parameter is not dino for dino in model.foundation.dino.parameters())
    }
    if set(owners) != expected:
        raise RuntimeError("Optimizer ownership does not cover trainable parameters exactly")
    training = config["training"]
    return [
        {
            "params": groups[name],
            "lr": float(training[f"lr_{name}"]),
            "group_name": name,
        }
        for name in groups
        if groups[name]
    ]


def _scheduler(optimizer: AdamW, total_updates: int, warmup_ratio: float) -> LambdaLR:
    warmup = max(1, int(round(total_updates * warmup_ratio)))

    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def _mechanism_ramps(step: int, total_updates: int) -> tuple[float, float]:
    r5 = min(max(step / max(total_updates * 0.05, 1.0), 0.0), 1.0)
    r10 = min(max(step / max(total_updates * 0.10, 1.0), 0.0), 1.0)
    return 0.25 + 0.75 * r5, r10


def _identity_output(
    model: METEROIAModel, output: dict[str, Any], progress: float
) -> dict[str, Tensor]:
    corrupt_token = torch.roll(output["factor_typed_token"], 1, 1)
    return model.action_transport(
        output["action_logits_visual"],
        output["action_nodes"],
        corrupt_token,
        output["factor_reliability"],
        output["factor_action_ownership"],
        progress=progress,
        update_running_stats=False,
    )


def _compute_losses(
    model: METEROIAModel,
    output: dict[str, Any],
    batch: dict[str, Any],
    *,
    config: dict[str, Any],
    grounding_ramp: float,
    mechanism_ramp: float,
    pu_lambda: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    action_target = batch["action"]
    reason_target = batch["reason"]
    dense = dense_factor_intervention_loss(
        output["action_logits_final"],
        output["action_factor_contributions"],
        action_target,
    )
    corrupt = _identity_output(model, output, mechanism_ramp)
    identity = identity_corruption_loss(
        output["action_factor_contributions"],
        corrupt["action_factor_contributions"],
        action_target,
    )
    output["dense_specificity_loss"] = dense["specificity"] * mechanism_ramp
    output["dense_identity_loss"] = identity * mechanism_ramp
    action = meter_action_loss(output, action_target, config["loss_weights"])
    confidence = output["factor_reliability"].detach()
    observability = output["factor_observability"].detach()
    reason = meter_reason_loss(
        output,
        reason_target,
        confidence,
        config["loss_weights"],
        observability=observability,
    )
    state_positive = output["factor_state_prob"][..., 0]
    pu_score = meter_pu_score(
        torch.sigmoid(output["reason_logits_global"]),
        state_positive,
        output["factor_reliability"],
        output["factor_observability"],
    )
    pu = meter_private_pu_loss(
        output["reason_logits_pu_private"],
        reason_target,
        pu_score,
        pu_lambda,
    )
    if "meter_grounding" in batch:
        grounding = meter_grounding_loss(output, batch["meter_grounding"])
        grounding_total = grounding["total"] * grounding_ramp
    else:
        zero = output["action_logits_final"].new_zeros(())
        grounding = {
            key: zero
            for key in (
                "anchor",
                "state",
                "observability",
                "discrimination",
                "mirror",
                "total",
            )
        }
        grounding_total = zero
    dense_weight = float(config["loss_weights"].get("dense_intervention", 0.05))
    total = (
        action["total"]
        + reason["total"]
        + grounding_total
        + dense_weight * mechanism_ramp * dense["total"]
        + pu
    )
    return total, {
        "action": action,
        "reason": reason,
        "grounding": grounding,
        "dense": dense,
        "identity": identity,
        "pu": pu,
        "pu_score": pu_score,
    }


@torch.no_grad()
def _collect_calibration(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    progress: float,
) -> dict[str, Tensor]:
    value = collect_outputs(
        model,
        loader,
        device,
        progress=progress,
        sequential_modes=False,
    )
    return {
        "action_logits": value["action_final"],
        "reason_logits": value["reason_final"],
        "action_labels": value["labels_action"],
        "reason_labels": value["labels_reason"],
        "state_probability": value["mechanism"]["factor_state_prob"],
        "reliability": value["mechanism"]["factor_reliability"],
        "observability": value["mechanism"]["factor_observability"],
    }


def _fit_calibration(
    model: METEROIAModel, collected: dict[str, Tensor]
) -> METERCalibrationResult:
    logits = torch.cat(
        [collected["action_logits"], collected["reason_logits"]], dim=1
    )
    labels = torch.cat(
        [collected["action_labels"], collected["reason_labels"]], dim=1
    )
    groups = tuple([0, 0, 1, 1] + [2] * 5 + [3] * 8 + [4] * 8)
    candidate = fit_train_calib_deploy_theta(
        logits,
        labels,
        model_state_hash=state_hash(model),
        fit_split="train_calib",
        label_groups=groups,
    )
    return guard_train_calib_deploy_theta(
        collected["action_logits"],
        collected["action_labels"],
        collected["reason_logits"],
        collected["reason_labels"],
        candidate,
    )


def _calibration_payload(value: METERCalibrationResult | None) -> dict[str, Any]:
    if value is None:
        return {}
    return {
        "theta": value.theta.detach().cpu(),
        "temperature": (
            None if value.temperature is None else value.temperature.detach().cpu()
        ),
        "strategy": value.strategy,
        "accepted": value.accepted,
        "fallback_reason": value.fallback_reason,
        "fit_split": value.fit_split,
        "representation_updated": value.representation_updated,
        "train_calib_raw_joint": value.train_calib_raw_joint,
        "train_calib_deploy_joint": value.train_calib_deploy_joint,
    }


def _update_pu(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    progress: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    value = _collect_calibration(model, loader, device, progress)
    audit = meter_hidden_positive_audit(
        torch.sigmoid(value["reason_logits"]),
        value["state_probability"][..., 0]
        * value["reliability"]
        * value["observability"],
        value["reason_labels"],
        min_positive_count=int(config["pu"].get("min_positive_count", 20)),
        seed=int(config["splits"]["seed"]),
    )
    maximum = float(config["pu"].get("max_lambda", 0.15))
    audit["lambda"] = [min(float(item), maximum) for item in audit["lambda"]]
    audit["active_labels"] = [
        index for index, item in enumerate(audit["lambda"]) if item > 0
    ]
    return audit


def _save_test_epoch(
    output_dir: Path,
    epoch: int,
    collected: dict[str, Any],
    calibration: METERCalibrationResult,
    *,
    mechanism: dict[str, Any],
    pu_state: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    summaries = metrics_summary(collected, calibration)
    branches = branch_metrics(collected)
    directory = save_epoch_artifacts(
        output_dir,
        epoch,
        metrics_raw=summaries["metrics_raw"],
        metrics_deploy=summaries["metrics_deploy"],
        branch_metrics=branches,
        logits={
            "action_final_raw_test": collected["action_final"],
            "action_visual_test": collected["action_visual"],
            "reason_final_raw_test": collected["reason_final"],
            "reason_global_test": collected["reason_global"],
        },
        labels={
            "action_test": collected["labels_action"],
            "reason_test": collected["labels_reason"],
        },
        file_names=collected["file_names"],
        diagnostics={
            "typed_evidence.json": mechanism,
            "pu_stats.json": pu_state,
            "calibration.json": _calibration_payload(calibration),
            "runtime.json": runtime,
        },
    )
    return directory, summaries, branches


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed or config["splits"]["seed"])
    _seed_everything(seed)
    device = torch.device(args.device)
    workers = int(args.num_workers if args.num_workers is not None else config["data"]["num_workers"])
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    grad_accum = int(
        args.gradient_accumulation_steps
        or config["training"]["gradient_accumulation_steps"]
    )
    epochs = int(args.epochs or config["training"]["epochs"])
    schema_path = Path("configs/meter_factor_schema.yaml")
    grounding_index = METERGroundingIndex(
        config["data"]["bdd100k_root"], schema_path=schema_path
    )
    train_dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="train",
        transform=meter_image_transform(training=True),
        grounding_index=grounding_index,
        include_grounding=True,
        mirror_probability=float(config["data"].get("mirror_probability", 0.25)),
    )
    plain_train_dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="train",
        transform=meter_image_transform(),
    )
    test_dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="test",
        transform=meter_image_transform(),
    )
    names = [sample.file_name for sample in train_dataset.base.samples]
    full_split = fixed_meter_split_indices(
        names,
        audit_fraction=float(config["splits"]["audit_fraction"]),
        calib_fraction=float(config["splits"]["calib_fraction"]),
        seed=seed,
    )
    split = {name: list(indices) for name, indices in full_split.items()}
    if args.max_train_samples:
        split["main"] = split["main"][: args.max_train_samples]
    if args.max_audit_samples:
        split["audit"] = split["audit"][: args.max_audit_samples]
    if args.max_calib_samples:
        split["calib"] = split["calib"][: args.max_calib_samples]
    test_indices = list(range(len(test_dataset)))
    if args.max_test_samples:
        test_indices = test_indices[: args.max_test_samples]
    train_loader = _loader(
        train_dataset,
        split["main"],
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        config=config,
    )
    audit_loader = _loader(
        plain_train_dataset,
        split["audit"],
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        config=config,
    )
    calib_loader = _loader(
        plain_train_dataset,
        split["calib"],
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        config=config,
    )
    test_loader = _loader(
        test_dataset,
        test_indices,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        config=config,
    )
    model = METEROIAModel(
        dim=int(config["model"]["dim"]),
        action_dim=int(config["model"]["action_dim"]),
        reason_dim=int(config["model"]["reason_dim"]),
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=bool(args.use_mock_dino),
        factor_rank=int(config["model"].get("factor_rank", 16)),
    ).to(device)
    optimizer = AdamW(
        _parameter_groups(model, config),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = max(epochs * updates_per_epoch, 1)
    scheduler = _scheduler(
        optimizer, total_updates, float(config["training"]["warmup_ratio"])
    )
    config_hash = combined_file_hash(args.config)
    source_hash = python_source_tree_hash(Path.cwd())
    schema_hash = file_hash(schema_path)
    start_epoch = 0
    optimizer_step = 0
    pu_state: dict[str, Any] = {
        "lambda": [0.0] * 21,
        "active_labels": [],
        "labels": [],
    }
    calibration: METERCalibrationResult | None = None
    if args.resume:
        payload = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_hash=config_hash,
            expected_source_hash=source_hash,
            expected_schema_hash=schema_hash,
        )
        start_epoch = int(payload["epoch"]) + 1
        optimizer_step = int(payload["optimizer_step"])
        pu_state = dict(payload.get("pu_state") or pu_state)
    manifest = {
        "git_head": _git_head(),
        "command_line": sys.argv,
        "seed": seed,
        "direct_image": True,
        "one_dino_call_per_ordinary_batch": True,
        "feature_cache_enabled": False,
        "token_compression": "none",
        "eval_splits": "test",
        "best_selection_split": "test",
        "internal_test_selected": True,
        "publication_eligible": False,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_workers": workers,
        "split_manifest": meter_split_manifest(names, full_split),
        "runtime_subset_counts": {
            name: len(indices) for name, indices in split.items()
        },
        "config": config,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    best: dict[str, float] = {
        "deploy_joint": -1.0,
        "raw_action_map": -1.0,
        "raw_action_mf1": -1.0,
        "raw_exp_map": -1.0,
        "deploy_exp_mf1": -1.0,
    }
    precision = str(config["training"].get("precision", "bf16")).lower()
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_start = time.perf_counter()
        epoch_rows: list[dict[str, Any]] = []
        for micro_step, raw_batch in enumerate(train_loader):
            data_time = time.perf_counter() - data_start
            batch = _move(raw_batch, device)
            grounding_ramp, mechanism_ramp = _mechanism_ramps(
                optimizer_step, total_updates
            )
            dino_start = time.perf_counter()
            with autocast():
                field = model.encode_images(batch["image"])
                dino_time = time.perf_counter() - dino_start
                output = model.decode_from_field(
                    field,
                    progress=optimizer_step / max(total_updates, 1),
                    collect_timing=True,
                    update_semantic_stats=True,
                )
                total, parts = _compute_losses(
                    model,
                    output,
                    batch,
                    config=config,
                    grounding_ramp=grounding_ramp,
                    mechanism_ramp=mechanism_ramp,
                    pu_lambda=torch.tensor(
                        pu_state["lambda"], device=device, dtype=output["reason_logits_final"].dtype
                    ),
                )
                scaled = total / grad_accum
            backward_start = time.perf_counter()
            scaled.backward()
            backward_time = time.perf_counter() - backward_start
            is_update = (micro_step + 1) % grad_accum == 0 or micro_step + 1 == len(train_loader)
            grad_norm = 0.0
            if is_update:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["training"]["grad_clip"])
                    )
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            timing = output["runtime_timing"]
            row = {
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "loss_total": float(total.detach()),
                "loss_action": float(parts["action"]["total"].detach()),
                "loss_reason": float(parts["reason"]["total"].detach()),
                "loss_anchor": float(parts["grounding"]["anchor"].detach()),
                "loss_state": float(parts["grounding"]["state"].detach()),
                "loss_observability": float(parts["grounding"]["observability"].detach()),
                "loss_discrimination": float(parts["grounding"]["discrimination"].detach()),
                "loss_mirror": float(parts["grounding"]["mirror"].detach()),
                "loss_dense_intervention": float(parts["dense"]["total"].detach()),
                "loss_identity": float(parts["identity"].detach()),
                "loss_pu": float(parts["pu"].detach()),
                "grounding_ramp": grounding_ramp,
                "mechanism_ramp": mechanism_ramp,
                "grad_norm": grad_norm,
                "data_time": data_time,
                "dino_time": dino_time,
                "foundation_time": timing.get("foundation_time", 0.0),
                "factor_time": timing.get("factor_time", 0.0),
                "action_time": timing.get("action_time", 0.0),
                "reason_time": timing.get("reason_time", 0.0),
                "backward_time": backward_time,
                "allocated_gb": (
                    torch.cuda.memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
                "reserved_gb": (
                    torch.cuda.memory_reserved(device) / 2**30
                    if device.type == "cuda"
                    else 0.0
                ),
                "dino_call_count": 1,
                "action_correction_rms_ratio": output[
                    "action_correction_rms_ratio"
                ].detach().cpu().tolist(),
                "factor_null_mean": float(output["factor_null_mass"].mean().detach()),
                "factor_observability_mean": float(
                    output["factor_observability"].mean().detach()
                ),
                "dense_action_coverage": int(parts["dense"]["action_coverage"]),
                "dense_factor_coverage": int(parts["dense"]["factor_coverage"]),
            }
            epoch_rows.append(row)
            append_jsonl(output_dir / "loss_components.jsonl", row)
            if (micro_step + 1) % 200 == 0:
                print("meter_batch " + json.dumps(row, sort_keys=True), flush=True)
            data_start = time.perf_counter()
            del output, field, total, scaled
        progress = min(1.0, optimizer_step / max(total_updates, 1))
        pu_state = _update_pu(model, audit_loader, device, progress, config)
        calibration_data = _collect_calibration(
            model, calib_loader, device, progress
        )
        calibration = _fit_calibration(model, calibration_data)
        test = collect_outputs(
            model, test_loader, device, progress=progress, sequential_modes=True
        )
        mechanism = mechanism_stats_from_collected(test)
        runtime = {
            "epoch": epoch,
            "train_rows": len(epoch_rows),
            "mean_data_time": sum(row["data_time"] for row in epoch_rows)
            / max(len(epoch_rows), 1),
            "mean_dino_time": sum(row["dino_time"] for row in epoch_rows)
            / max(len(epoch_rows), 1),
            "peak_reserved_gb": max(
                (row["reserved_gb"] for row in epoch_rows), default=0.0
            ),
            "eval_mode_time": mechanism.get("eval_mode_time", {}),
            "dino_call_count": mechanism.get("dino_call_count", {}),
        }
        _, summaries, branches = _save_test_epoch(
            output_dir,
            epoch,
            test,
            calibration,
            mechanism=mechanism,
            pu_state=pu_state,
            runtime=runtime,
        )
        metric_row = {
            "epoch": epoch,
            **summaries["metrics_raw"],
            **summaries["metrics_deploy"],
            "visual_Act_mAP": branches["action_visual"]["Act_mAP"],
            "visual_Act_mF1": branches["action_visual"]["Act_mF1"],
            "global_Exp_mAP": branches["reason_global"]["Exp_mAP"],
            "global_Exp_mF1": branches["reason_global"]["Exp_mF1"],
            "factor_off_Act_mAP": branches["factor_off"]["Act_mAP"],
            "reason_correction_off_Exp_mAP": branches["reason_correction_off"][
                "Exp_mAP"
            ],
        }
        append_jsonl(output_dir / "metrics_summary.jsonl", metric_row)
        print("meter_epoch " + json.dumps(metric_row, sort_keys=True), flush=True)
        checkpoint_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch,
            "micro_step": len(train_loader),
            "optimizer_step": optimizer_step,
            "runtime_profile": runtime,
            "meta_state": {"training_enabled": False, "audit_only": True},
            "pu_state": pu_state,
            "calibration": _calibration_payload(calibration),
            "config_hash": config_hash,
            "source_hash": source_hash,
            "schema_hash": schema_hash,
        }
        save_checkpoint(output_dir / "checkpoint_latest.pth", **checkpoint_args)
        candidates = {
            "deploy_joint": float(summaries["metrics_deploy"]["deploy_joint"]),
            "raw_action_map": float(summaries["metrics_raw"]["Act_mAP"]),
            "raw_action_mf1": float(summaries["metrics_raw"]["Act_mF1"]),
            "raw_exp_map": float(summaries["metrics_raw"]["Exp_mAP"]),
            "deploy_exp_mf1": float(summaries["metrics_deploy"]["Exp_mF1"]),
        }
        for name, value in candidates.items():
            if value > best[name]:
                best[name] = value
                save_checkpoint(
                    output_dir / f"checkpoint_best_{name}.pth", **checkpoint_args
                )
        write_json(output_dir / "best_metrics.json", best)
    write_json(
        output_dir / "GOAL_COMPLETED_METER_OIA_V2_TESA.json",
        {
            "completed": True,
            "epochs": epochs,
            "git_head": _git_head(),
            "best": best,
            "internal_test_selected": True,
            "publication_eligible": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/fate_oia_train_360x640_acpr_meter_oia_v2_tesa.yaml",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_audit_samples", type=int, default=0)
    parser.add_argument("--max_calib_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--use_mock_dino", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--no_feature_cache", action="store_true")
    parser.add_argument("--require_no_token_compression", action="store_true")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    if args.require_no_token_compression and config["model"]["token_compression"] != "none":
        raise RuntimeError("Token compression is forbidden")
    if args.no_feature_cache and config["model"]["feature_cache_enabled"]:
        raise RuntimeError("Feature cache is forbidden")
    train(config, args)


if __name__ == "__main__":
    main()
