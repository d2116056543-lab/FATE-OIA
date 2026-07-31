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
from torch import Tensor, nn
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
from fate_oia.engine.tesa_diagnostics import run_stratified_patch_audit
from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_counterfactual_losses import (
    dense_factor_intervention_loss,
    identity_corruption_loss,
    reason_identity_corruption_loss,
)
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.losses.meter_pu_losses import (
    meter_hidden_positive_audit,
    meter_private_pu_loss,
    meter_pu_score,
)
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.metrics import binary_average_precision, binary_roc_auc
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
from fate_oia.utils.tesa_contracts import build_runtime_subset_counts
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


def initialize_model_from_checkpoint(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load model weights only so a diverged optimizer cannot be revived."""
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain a model state dict")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(
            f"Incompatible checkpoint {path}: missing={sorted(incompatible.missing_keys)}, unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return {
        "mode": "weights_only",
        "source_epoch": int(payload.get("epoch", -1)),
        "source_optimizer_step": int(payload.get("optimizer_step", -1)),
        "path": str(path),
    }





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


def _slice_encoded_field(
    field: dict[str, Any], start: int, end: int, encoded_batch: int
) -> dict[str, Any]:
    return {
        key: (
            value[start:end]
            if isinstance(value, Tensor)
            and value.ndim > 0
            and value.shape[0] == encoded_batch
            else value
        )
        for key, value in field.items()
    }


def _forward_training_batch(
    model: METEROIAModel,
    images: Tensor,
    *,
    progress: float,
    mirror_due: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, float]:
    """Decode an optional paired mirror while keeping one DINO encode call."""
    batch_size = images.shape[0]
    encoded_images = (
        torch.cat([images, torch.flip(images[:1], dims=[-1])], dim=0)
        if mirror_due
        else images
    )
    encoded_batch = encoded_images.shape[0]
    encode_start = time.perf_counter()
    encoded_field = model.encode_images(encoded_images)
    encode_seconds = time.perf_counter() - encode_start
    output = model.decode_from_field(
        _slice_encoded_field(encoded_field, 0, batch_size, encoded_batch),
        progress=progress,
        collect_timing=True,
        update_semantic_stats=True,
    )
    mirror_output = (
        model.decode_from_field(
            _slice_encoded_field(
                encoded_field, batch_size, encoded_batch, encoded_batch
            ),
            progress=progress,
            collect_timing=False,
            update_semantic_stats=False,
        )
        if mirror_due
        else None
    )
    return output, mirror_output, encode_seconds


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


def _diagnostic_due(epoch: int, total_epochs: int, interval: int) -> bool:
    """Run expensive branch/deletion audits at a fixed interval and the final epoch."""
    if total_epochs <= 0:
        return False
    return epoch == total_epochs - 1 or (interval > 0 and (epoch + 1) % interval == 0)


def _identity_output(
    model: METEROIAModel,
    output: dict[str, Any],
    progress: float,
    mode: str,
) -> dict[str, Tensor]:
    reliability = output["factor_reliability"]
    factor_source = output["factor_observability"]
    corrupt_value_token = output["factor_action_value_token"]
    if mode == "schema":
        corrupt_token = torch.roll(output["factor_action_token"], 1, 1)
        corrupt_value_token = torch.roll(corrupt_value_token, 1, 1)
    elif mode == "cross_sample":
        if output["factor_action_token"].shape[0] < 2:
            corrupt_token = output["factor_action_token"]
        else:
            corrupt_token = torch.roll(output["factor_action_token"], 1, 0)
            corrupt_value_token = torch.roll(corrupt_value_token, 1, 0)
            reliability = torch.roll(reliability, 1, 0)
            factor_source = torch.roll(factor_source, 1, 0)
    elif mode == "state":
        corrupt_state = torch.roll(output["factor_state_prob"], 1, -1)
        corrupt_token = model.typed_factors.compose_action_token(
            output["factor_anchor_token"], corrupt_state
        )
    else:
        raise ValueError(f"Unknown identity corruption mode: {mode}")
    return model.action_transport(
        output["action_logits_visual"],
        output["action_nodes"],
        corrupt_token,
        reliability,
        output["factor_action_ownership"],
        factor_value_token=corrupt_value_token,
        factor_source=factor_source,
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
    mirror_output: dict[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    action_target = batch["action"]
    reason_target = batch["reason"]
    dense = dense_factor_intervention_loss(
        output["action_logits_final"],
        output["action_factor_contributions"],
        action_target,
    )
    identity_terms: dict[str, Tensor] = {}
    reason_identity_terms: dict[str, Tensor] = {}
    for mode in ("schema", "cross_sample", "state"):
        if mode == "cross_sample" and output["factor_typed_token"].shape[0] < 2:
            zero = output["action_logits_final"].new_zeros(())
            identity_terms[mode] = zero
            reason_identity_terms[mode] = zero
            continue
        corrupt = _identity_output(model, output, mechanism_ramp, mode)
        identity_terms[mode] = identity_corruption_loss(
            output["action_factor_contributions"],
            corrupt["action_factor_contributions"],
            action_target,
        )
        corrupt_reliability = output["factor_reliability"]
        if mode == "schema":
            corrupt_token = torch.roll(output["factor_typed_token"], 1, 1)
        elif mode == "cross_sample":
            corrupt_token = (
                torch.roll(output["factor_typed_token"], 1, 0)
                if output["factor_typed_token"].shape[0] > 1
                else output["factor_typed_token"]
            )
            if output["factor_typed_token"].shape[0] > 1:
                corrupt_reliability = torch.roll(corrupt_reliability, 1, 0)
        else:
            corrupt_token = model.typed_factors.compose_typed_token(
                output["factor_global_token"],
                output["factor_anchor_token"],
                torch.roll(output["factor_state_prob"], 1, -1),
            )
        corrupt_reason = model.reason_decoder(
            patch_tokens_by_layer=output["patch_tokens_by_layer"],
            reason_logits_calalign=output["reason_logits_calalign"],
            factor_typed_token=corrupt_token,
            factor_reliability=corrupt_reliability,
            factor_groundable_mask=output["factor_groundable_mask"],
            progress=mechanism_ramp,
        )
        reason_identity_terms[mode] = reason_identity_corruption_loss(
            output["reason_logits_final"],
            corrupt_reason["reason_logits_final"],
            reason_target,
        )
    identity = torch.stack(tuple(identity_terms.values())).mean()
    reason_identity = torch.stack(tuple(reason_identity_terms.values())).mean()
    output["dense_specificity_loss"] = dense["specificity"] * mechanism_ramp
    output["action_specificity_loss"] = dense["specificity"] * mechanism_ramp
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
        grounding = meter_grounding_loss(
            output,
            batch["meter_grounding"],
            mirrored_output=mirror_output,
            mirror_pairs=model.typed_factors.mirror_pairs,
            weights=config["loss_weights"],
        )
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
        + dense_weight * mechanism_ramp * dense["necessity"]
        + float(config["loss_weights"].get("reason_identity", 0.03))
        * mechanism_ramp
        * reason_identity
        + pu
    )
    return total, {
        "action": action,
        "reason": reason,
        "grounding": grounding,
        "dense": dense,
        "identity": identity,
        "identity_terms": identity_terms,
        "reason_identity": reason_identity,
        "reason_identity_terms": reason_identity_terms,
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
        "reason_global_logits": value["reason_global"],
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


def _identity_ap_diagnostics(
    branches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clean = [float(value) for value in branches["action_final"]["Act_per_label_ap"]]
    matrix: list[list[float]] = []
    for target in range(4):
        corrupt = [
            float(value)
            for value in branches[f"schema_target_{target}"]["Act_per_label_ap"]
        ]
        matrix.append(
            [round(clean[action] - corrupt[action], 10) for action in range(4)]
        )
    target_delta = [matrix[action][action] for action in range(4)]
    wrong_delta = [
        round(
            sum(abs(matrix[target][action]) for action in range(4) if action != target)
            / 3.0,
            10,
        )
        for target in range(4)
    ]
    return {
        "identity_ap_delta_matrix": matrix,
        "identity_target_delta": target_delta,
        "identity_wrong_delta": wrong_delta,
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
        torch.sigmoid(value["reason_global_logits"]),
        value["state_probability"][..., 0],
        value["reason_labels"],
        reliability=value["reliability"],
        observability=value["observability"],
        min_positive_count=int(config["pu"].get("min_positive_count", 20)),
        seed=int(config["splits"]["seed"]),
    )
    maximum = float(config["pu"].get("max_lambda", 0.15))
    audit["lambda"] = [min(float(item), maximum) for item in audit["lambda"]]
    audit["active_labels"] = [
        index for index, item in enumerate(audit["lambda"]) if item > 0
    ]
    return audit


@torch.no_grad()
def _typed_factor_audit(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    progress: float,
) -> dict[str, Any]:
    anchor_score: list[list[float]] = [[] for _ in range(21)]
    wrong_score: list[list[float]] = [[] for _ in range(21)]
    state_probability: list[list[float]] = [[] for _ in range(21)]
    state_target: list[list[float]] = [[] for _ in range(21)]
    observability: list[list[float]] = [[] for _ in range(21)]
    observability_target: list[list[float]] = [[] for _ in range(21)]
    state_confusion = torch.zeros(21, 3, 3, dtype=torch.long)
    source_count = [0] * 21
    mirror_margin: list[list[float]] = [[] for _ in range(21)]
    mirror_partner = {
        9: 15, 10: 16, 11: 17, 12: 18, 13: 19,
        15: 9, 16: 10, 17: 11, 18: 12, 19: 13,
    }
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        mirror_pair = model.forward_mirror_pair(batch["image"], progress=progress)
        output = mirror_pair["original"]
        for factor, value in mirror_pair["mirror_equivariance"][
            "per_factor_margin"
        ].items():
            mirror_margin[int(factor)].append(float(value))
        target = batch["meter_grounding"]
        predicted_anchor = output["factor_anchor_map"]
        target_anchor = target["factor_anchor_map"].flatten(2)
        for factor in range(21):
            valid_anchor = target["factor_anchor_valid"][:, factor].bool()
            valid_state = target["factor_state_valid"][:, factor].bool()
            valid_obs = target["factor_observability_valid"][:, factor].bool()
            if bool(valid_anchor.any()):
                score = (
                    predicted_anchor[:, factor] * target_anchor[:, factor]
                ).sum(-1)
                wrong_factor = mirror_partner.get(factor, (factor + 1) % 21)
                wrong = (
                    predicted_anchor[:, wrong_factor] * target_anchor[:, factor]
                ).sum(-1)
                anchor_score[factor].extend(score[valid_anchor].cpu().tolist())
                wrong_score[factor].extend(wrong[valid_anchor].cpu().tolist())
                source_count[factor] += int(valid_anchor.sum())
            if bool(valid_state.any()):
                predicted_state = output["factor_state_prob"][:, factor].argmax(-1)
                for truth, prediction in zip(
                    target["factor_state_target"][valid_state, factor].cpu(),
                    predicted_state[valid_state].cpu(),
                ):
                    state_confusion[factor, int(truth), int(prediction)] += 1
                state_probability[factor].extend(
                    output["factor_state_prob"][valid_state, factor, 0]
                    .cpu()
                    .tolist()
                )
                state_target[factor].extend(
                    (target["factor_state_target"][valid_state, factor] == 0)
                    .float()
                    .cpu()
                    .tolist()
                )
            if bool(valid_obs.any()):
                observability[factor].extend(
                    output["factor_observability"][valid_obs, factor]
                    .cpu()
                    .tolist()
                )
                observability_target[factor].extend(
                    target["factor_observability"][valid_obs, factor]
                    .cpu()
                    .tolist()
                )
    rows: list[dict[str, Any]] = []
    for factor in range(21):
        state_p = torch.tensor(state_probability[factor])
        state_y = torch.tensor(state_target[factor])
        obs_p = torch.tensor(observability[factor])
        obs_y = torch.tensor(observability_target[factor])
        rows.append(
            {
                "factor_id": factor,
                "source_count": source_count[factor],
                "anchor_overlap_mean": (
                    sum(anchor_score[factor]) / len(anchor_score[factor])
                    if anchor_score[factor]
                    else None
                ),
                "same_type_wrong_overlap_mean": (
                    sum(wrong_score[factor]) / len(wrong_score[factor])
                    if wrong_score[factor]
                    else None
                ),
                "same_type_margin": (
                    (
                        sum(anchor_score[factor]) / len(anchor_score[factor])
                        - sum(wrong_score[factor]) / len(wrong_score[factor])
                    )
                    if anchor_score[factor] and wrong_score[factor]
                    else None
                ),
                "state_auprc": (
                    binary_average_precision(state_p, state_y)
                    if state_p.numel()
                    else None
                ),
                "state_frequency_baseline": (
                    float(state_y.mean()) if state_y.numel() else None
                ),
                "state_auc": (
                    binary_roc_auc(state_p, state_y) if state_p.numel() else None
                ),
                "state_confusion_matrix": state_confusion[factor].tolist(),
                "observability_auc": (
                    binary_roc_auc(obs_p, obs_y) if obs_p.numel() else None
                ),
                "mirror_equivariance": (
                    sum(mirror_margin[factor]) / len(mirror_margin[factor])
                    if mirror_margin[factor]
                    else None
                ),
            }
        )
    return {
        "per_factor": rows,
        "source_coverage": source_count,
        "state_confusion_matrix": state_confusion.tolist(),
        "factors_with_anchor_source": sum(count > 0 for count in source_count),
    }


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
    grounded_audit_dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="train",
        transform=meter_image_transform(),
        grounding_index=grounding_index,
        include_grounding=True,
    )
    factor_audit_loader = _loader(
        grounded_audit_dataset,
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
        action_correction_fraction=float(
            config["model"].get("action_correction_fraction", 0.20)
        ),
        action_max_visual_rms=float(
            config["model"].get("action_max_visual_rms", 5.0)
        ),
        action_max_delta=float(config["model"].get("action_max_delta", 1.0)),
        action_logit_norm_cap=float(
            config["model"].get("action_logit_norm_cap", 20.0)
        ),
    ).to(device)
    initialization: dict[str, Any] | None = None
    if args.init_model_checkpoint:
        initialization = initialize_model_from_checkpoint(
            model, args.init_model_checkpoint
        )
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
        "config_hash": config_hash,
        "source_hash": source_hash,
        "schema_hash": schema_hash,
        "command_line": sys.argv,
        "seed": seed,
        "use_mock_dino": bool(args.use_mock_dino),
        "pretrained_weights": config["backbone"]["pretrained_weights"],
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
        "initialization": initialization,
        "split_manifest": meter_split_manifest(names, full_split),
        "runtime_subset_counts": build_runtime_subset_counts(
            split, test_count=len(test_indices)
        ),
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
    cumulative_patch_ids: set[str] = set()
    cumulative_path = output_dir / "patch_audit_cumulative.json"
    if cumulative_path.exists():
        cumulative_patch_ids.update(
            json.loads(cumulative_path.read_text(encoding="utf-8")).get(
                "sample_ids", []
            )
        )
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
            dino_calls_before = model._encode_call_count
            mirror_interval = int(
                config["training"].get("mirror_training_interval", 8)
            )
            mirror_due = mirror_interval > 0 and micro_step % mirror_interval == 0
            with autocast():
                output, mirror_output, dino_time = _forward_training_batch(
                    model,
                    batch["image"],
                    progress=optimizer_step / max(total_updates, 1),
                    mirror_due=mirror_due,
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
                    mirror_output=mirror_output,
                )
                scaled = total / grad_accum
            backward_start = time.perf_counter()
            scaled.backward()
            backward_time = time.perf_counter() - backward_start
            is_update = (micro_step + 1) % grad_accum == 0 or micro_step + 1 == len(train_loader)
            grad_norm = 0.0
            foundation_grad_norm = 0.0
            if is_update:
                foundation_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.foundation.parameters(),
                        float(
                            config["training"].get(
                                "foundation_grad_clip",
                                config["training"]["grad_clip"],
                            )
                        ),
                    )
                )
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
                "loss_null": float(parts["grounding"]["null"].detach()),
                "loss_observability": float(parts["grounding"]["observability"].detach()),
                "loss_discrimination": float(parts["grounding"]["discrimination"].detach()),
                "loss_mirror": float(parts["grounding"]["mirror"].detach()),
                "loss_dense_intervention": float(parts["dense"]["total"].detach()),
                "loss_dense_necessity": float(parts["dense"]["necessity"].detach()),
                "loss_action_specificity": float(
                    parts["action"]["specificity"].detach()
                ),
                "loss_action_anti_monopoly": float(
                    parts["action"]["anti_monopoly"].detach()
                ),
                "loss_action_near_boundary": float(
                    parts["action"]["near_boundary"].detach()
                ),
                "loss_action_delta_ranking": float(
                    parts["action"]["delta_ranking"].detach()
                ),
                "loss_identity": float(parts["identity"].detach()),
                "loss_pu": float(parts["pu"].detach()),
                "grounding_ramp": grounding_ramp,
                "mechanism_ramp": mechanism_ramp,
                "grad_norm": grad_norm,
                "foundation_grad_norm": foundation_grad_norm,
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
                "dino_call_count": model._encode_call_count - dino_calls_before,
                "action_correction_rms_ratio": output[
                    "action_correction_rms_ratio"
                ].detach().cpu().tolist(),
                "action_correction_kappa": output[
                    "action_correction_kappa"
                ].detach().cpu().tolist(),
                "action_credit_ramp": float(output["action_credit_ramp"].detach()),
                "action_visual_logit_abs_max": float(
                    output["action_logits_visual"].detach().abs().max()
                ),
                "action_final_logit_abs_max": float(
                    output["action_logits_final"].detach().abs().max()
                ),
                "action_direct_preclip_norm_max": float(
                    output["action_direct_preclip_norm"].detach().max()
                ),
                "factor_null_mean": float(output["factor_null_mass"].mean().detach()),
                "factor_observability_mean": float(
                    output["factor_observability"].mean().detach()
                ),
                "dense_action_coverage": int(parts["dense"]["action_coverage"]),
                "dense_factor_coverage": int(parts["dense"]["factor_coverage"]),
                "dense_correct_effect_abs": float(
                    parts["dense"]["correct_effect"].abs().mean()
                ),
                "dense_wrong_effect_abs": float(
                    parts["dense"]["wrong_effect"].abs().mean()
                ),
                "dense_correct_effect_abs_per_action": parts["dense"][
                    "correct_effect"
                ]
                .abs()
                .mean(0)
                .tolist(),
                "dense_wrong_effect_abs_per_action": parts["dense"][
                    "wrong_effect"
                ]
                .abs()
                .mean(0)
                .tolist(),
                "loss_reason_identity": float(parts["reason_identity"].detach()),
                "loss_identity_schema": float(
                    parts["identity_terms"]["schema"].detach()
                ),
                "loss_identity_cross_sample": float(
                    parts["identity_terms"]["cross_sample"].detach()
                ),
                "loss_identity_state": float(
                    parts["identity_terms"]["state"].detach()
                ),
                "loss_reason_identity_schema": float(
                    parts["reason_identity_terms"]["schema"].detach()
                ),
                "loss_reason_identity_cross_sample": float(
                    parts["reason_identity_terms"]["cross_sample"].detach()
                ),
                "loss_reason_identity_state": float(
                    parts["reason_identity_terms"]["state"].detach()
                ),
            }
            epoch_rows.append(row)
            append_jsonl(output_dir / "loss_components.jsonl", row)
            if (micro_step + 1) % 200 == 0:
                print("meter_batch " + json.dumps(row, sort_keys=True), flush=True)
            data_start = time.perf_counter()
            del output, mirror_output, total, scaled
        progress = min(1.0, optimizer_step / max(total_updates, 1))
        pre_eval_runtime = {
            "epoch": epoch,
            "train_rows": len(epoch_rows),
            "mean_data_time": sum(row["data_time"] for row in epoch_rows)
            / max(len(epoch_rows), 1),
            "mean_dino_time": sum(row["dino_time"] for row in epoch_rows)
            / max(len(epoch_rows), 1),
            "peak_reserved_gb": max(
                (row["reserved_gb"] for row in epoch_rows), default=0.0
            ),
            "evaluation_complete": False,
        }
        save_checkpoint(
            output_dir / "checkpoint_pre_eval.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            micro_step=len(train_loader),
            optimizer_step=optimizer_step,
            runtime_profile=pre_eval_runtime,
            meta_state={"training_enabled": False, "audit_only": True},
            pu_state=pu_state,
            calibration=_calibration_payload(calibration),
            config_hash=config_hash,
            source_hash=source_hash,
            schema_hash=schema_hash,
        )
        pu_state = _update_pu(model, audit_loader, device, progress, config)
        calibration_data = _collect_calibration(
            model, calib_loader, device, progress
        )
        calibration = _fit_calibration(model, calibration_data)
        diagnostic_due = _diagnostic_due(
            epoch,
            epochs,
            int(config.get("evaluation", {}).get("diagnostic_interval_epochs", 5)),
        )
        test = collect_outputs(
            model,
            test_loader,
            device,
            progress=progress,
            sequential_modes=diagnostic_due,
        )
        test_branches = branch_metrics(test)
        mechanism = mechanism_stats_from_collected(test)
        mechanism.update(
            {
                "diagnostic_due": diagnostic_due,
                "analytic_coverage": {
                    "actions": max(
                        (row["dense_action_coverage"] for row in epoch_rows),
                        default=0,
                    ),
                    "factors": max(
                        (row["dense_factor_coverage"] for row in epoch_rows),
                        default=0,
                    ),
                },
                "correct_factor_effect": sum(
                    row["dense_correct_effect_abs"] for row in epoch_rows
                )
                / max(len(epoch_rows), 1),
                "wrong_factor_effect": sum(
                    row["dense_wrong_effect_abs"] for row in epoch_rows
                )
                / max(len(epoch_rows), 1),
            }
        )
        if diagnostic_due:
            train_audit = _typed_factor_audit(
                model, factor_audit_loader, device, progress
            )
            patch_audit = run_stratified_patch_audit(
                model,
                factor_audit_loader,
                device,
                progress=progress,
                max_unique=min(
                    int(config.get("evaluation", {}).get("patch_audit_max_unique", 16)),
                    len(split["audit"]),
                ),
                previous_sample_ids=cumulative_patch_ids,
            )
            cumulative_patch_ids.update(patch_audit.get("sample_ids", []))
            write_json(
                cumulative_path,
                {
                    "sample_ids": sorted(cumulative_patch_ids),
                    "cumulative_unique_count": len(cumulative_patch_ids),
                },
            )
            mechanism.update(
                {
                    "train_audit": train_audit,
                    "patch_audit": patch_audit,
                    "state_confusion_matrix": train_audit["state_confusion_matrix"],
                    "source_coverage": train_audit["source_coverage"],
                    "same_type_margin": [
                        row["same_type_margin"] for row in train_audit["per_factor"]
                    ],
                    "mirror_equivariance": [
                        row["mirror_equivariance"] for row in train_audit["per_factor"]
                    ],
                    **_identity_ap_diagnostics(test_branches),
                    "reason_identity_delta_per_label": [
                        round(
                            float(test_branches["reason_final"]["Exp_per_label_ap"][label])
                            - float(
                                test_branches["schema_corruption"]["Exp_per_label_ap"][
                                    label
                                ]
                            ),
                            10,
                        )
                        for label in range(21)
                    ],
                    "factor_off_delta": mechanism.get(
                        "factor_off_delta_per_action", []
                    ),
                    "state_off_delta": mechanism.get(
                        "state_off_delta_per_action", []
                    ),
                    "cross_sample_swap_effect": mechanism.get(
                        "cross_sample_swap_delta_per_action", []
                    ),
                    "patch_selected_effect": patch_audit.get(
                        "selected_effect_mean", 0.0
                    ),
                    "patch_control_effect": patch_audit.get(
                        "control_effect_mean", 0.0
                    ),
                    "unique_sample_count": patch_audit.get(
                        "unique_sample_count", 0
                    ),
                    "cumulative_unique_count": patch_audit.get(
                        "cumulative_unique_count", 0
                    ),
                    "action_coverage": patch_audit.get("action_coverage", []),
                    "factor_coverage": patch_audit.get("factor_coverage", []),
                }
            )
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
            "factor_off_Act_mAP": test_branches.get("factor_off", {}).get("Act_mAP"),
            "reason_correction_off_Exp_mAP": test_branches.get(
                "reason_correction_off", {}
            ).get("Exp_mAP"),
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
    parser.add_argument("--init_model_checkpoint", default="")
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
