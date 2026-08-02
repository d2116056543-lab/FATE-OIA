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
    REASON_MIRROR_PAIRS,
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
from fate_oia.losses.meter_counterfactual_losses import identity_corruption_loss
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.losses.meter_pu_losses import (
    meter_hidden_positive_audit,
    meter_private_pu_loss,
    meter_pu_score,
)
from fate_oia.losses.meter_reason_losses import (
    cross_view_consistency,
    meter_reason_loss,
    noisy_zero_trust,
)
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.optim.heca_optimization import (
    HECAExcessRiskBalancer,
    HECALossRegistry,
    HECAScheduleState,
    ReasonProbabilityEMA,
    correction_fraction_for_run,
    identity_corruption_mode,
    validate_formal_protocol,
)
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
    validate_heca_pilot_bundle,
    write_json,
)
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.engine.evaluate_meter_oia_v3_heca_pilot import (
    validate_heca_pilot_recomputation,
)
from fate_oia.utils.tesa_contracts import build_runtime_subset_counts
from fate_oia.utils.meter_posthoc_calibration import (
    METERCalibrationResult,
    fit_train_calib_deploy_theta,
    guard_train_calib_deploy_theta,
)
from fate_oia.utils.heca_clean_head import worktree_admission_failures


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


def _heca_worktree_admission_failures() -> list[str]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], text=True
        )
    except Exception:
        return ["git_status_unavailable"]
    return worktree_admission_failures(status)


def _validated_gate_pass(path: str) -> bool:
    if not path:
        return False
    if _heca_worktree_admission_failures():
        return False
    source = Path(path)
    if not source.exists() or source.name != "HECA_GATE_C.json":
        return False
    head = _git_head()
    if validate_heca_pilot_bundle(source.parent, expected_git_head=head):
        return False
    if validate_heca_pilot_recomputation(source.parent, expected_git_head=head):
        return False
    payload = json.loads(source.read_text(encoding="utf-8"))
    pilot = json.loads(
        (source.parent / "HECA_PILOT_PASS.json").read_text(encoding="utf-8")
    )
    return (
        payload.get("pass") is True
        and payload.get("gate") in {"C", "HECA_GATE_C"}
        and pilot.get("gate_payloads", {}).get("C") == payload
        and all(pilot.get("gates", {}).get(letter) is True for letter in "ABCDEFG")
    )


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


def _autograd_norm(loss: Tensor, parameters: Iterable[Tensor]) -> float:
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = loss.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.detach().float().square().sum()
    return float(total.sqrt().cpu())


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
    view_kind: str = "mirror",
    shared_action_gradient_scale: float = 1.0,
    shared_reason_gradient_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any] | None, float]:
    """Decode one paired geometric or light view in the same DINO call."""
    batch_size = images.shape[0]
    if view_kind not in {"mirror", "light"}:
        raise ValueError(f"Unknown HECA paired view: {view_kind}")
    paired = (
        torch.flip(images, dims=[-1])
        if view_kind == "mirror"
        else (images * 0.92 + 0.03).clamp(images.amin(), images.amax())
    )
    encoded_images = (
        torch.cat([images, paired], dim=0)
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
        shared_action_gradient_scale=shared_action_gradient_scale,
        shared_reason_gradient_scale=shared_reason_gradient_scale,
    )
    mirror_output = (
        model.decode_from_field(
            _slice_encoded_field(
                encoded_field, batch_size, encoded_batch, encoded_batch
            ),
            progress=progress,
            collect_timing=False,
            update_semantic_stats=False,
            shared_action_gradient_scale=shared_action_gradient_scale,
            shared_reason_gradient_scale=shared_reason_gradient_scale,
        )
        if mirror_due
        else None
    )
    return output, mirror_output, encode_seconds


def _mirror_reason_tensor(value: Tensor) -> Tensor:
    result = value.clone()
    for left, right in REASON_MIRROR_PAIRS:
        result[:, left], result[:, right] = value[:, right], value[:, left]
    return result


def _parameter_groups(
    model: METEROIAModel, config: dict[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Tensor]] = {
        "foundation": [],
        "shared_adapter": [],
        "measurement": [],
        "action_credit": [],
        "reason_global": [],
        "reason_correction": [],
    }
    owners: dict[int, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("foundation.dino."):
            continue
        if name.startswith("heca_adapters.shared_adapter."):
            owner = "shared_adapter"
        elif name.startswith("heca_adapters.action_private_adapter."):
            owner = "action_credit"
        elif name.startswith("heca_adapters.reason_private_adapter."):
            owner = "reason_global"
        elif name.startswith("heca_adapters.pu_private_head."):
            owner = "reason_global"
        elif name.startswith("typed_factors.action_bridge_proj."):
            owner = "action_credit"
        elif name.startswith("typed_factors."):
            owner = "measurement"
        elif name.startswith("action_transport."):
            owner = "action_credit"
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
    learning_rates = {
        "foundation": float(training["foundation_target"]),
        "shared_adapter": float(training["lr_shared_adapter"]),
        "measurement": float(training["lr_measurement"]),
        "action_credit": float(training["lr_action_credit"]),
        "reason_global": float(training["lr_reason_global"]),
        "reason_correction": float(training["lr_reason_correction"]),
    }
    return [
        {
            "params": groups[name],
            "lr": learning_rates[name],
            "group_name": name,
        }
        for name in groups
        if groups[name]
    ]


def _action_credit_parameters(model: METEROIAModel) -> tuple[Tensor, ...]:
    """Return the whole action-only optimizer owner for firewall probes."""
    prefixes = (
        "heca_adapters.action_private_adapter.",
        "typed_factors.action_bridge_proj.",
        "action_transport.",
    )
    return tuple(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(prefixes)
    )


def _state_measurement_parameters(model: METEROIAModel) -> tuple[Tensor, ...]:
    """Return every shared typed-state parameter reached by action credit."""
    return (
        model.typed_factors.state_weight,
        model.typed_factors.state_bias,
        model.typed_factors.action_state_embeddings,
        *tuple(model.typed_factors.state_text_proj.parameters()),
        *tuple(model.typed_factors.global_proj.parameters()),
    )


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
    bridge = output["factor_action_bridge_token"]
    state_prob_credit = output["factor_state_prob_credit"]
    if mode == "schema":
        bridge = torch.roll(bridge, 1, 1)
        state_prob_credit = torch.roll(state_prob_credit, 1, 1)
    elif mode == "cross_sample":
        if bridge.shape[0] > 1:
            bridge = torch.roll(bridge, 1, 0)
            state_prob_credit = torch.roll(state_prob_credit, 1, 0)
            reliability = torch.roll(reliability, 1, 0)
    elif mode == "state":
        corrupt_state = torch.roll(output["factor_state_prob_action"], 1, -1)
        bridge, state_prob_credit = model.typed_factors.compose_action_bridge_token(
            output["factor_anchor_token"],
            corrupt_state,
            output["factor_global_token"],
        )
    else:
        raise ValueError(f"Unknown identity corruption mode: {mode}")
    return model.action_transport(
        output["action_logits_visual"],
        output["action_nodes"],
        bridge,
        state_prob_credit,
        reliability,
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
    forward_progress: float,
    pu_lambda: Tensor,
    mirror_output: dict[str, Any] | None = None,
    corruption_step: int,
    view_kind: str = "mirror",
) -> tuple[Tensor, dict[str, Any]]:
    action_target = batch["action"]
    reason_target = batch["reason"]
    mode = identity_corruption_mode(corruption_step)
    corrupt = _identity_output(model, output, forward_progress, mode)
    identity = identity_corruption_loss(
        output["action_evidence_delta"],
        corrupt["action_evidence_delta"].detach(),
        action_target,
    )
    # The target-effectiveness term receives this same-image intervention as a
    # detached control. It cannot improve the loss by damaging the control.
    output["action_counterfactual_delta"] = corrupt["action_evidence_delta"].detach()
    output["action_specificity_loss"] = identity * mechanism_ramp
    action = meter_action_loss(output, action_target, config["loss_weights"])
    state_positive = output["factor_state_prob"][..., 0]
    view_consistency = torch.ones_like(state_positive)
    loss_view_output = mirror_output
    if mirror_output is not None:
        paired_reason = mirror_output["reason_logits_global"]
        paired_reliability = mirror_output["factor_reliability"]
        if view_kind == "mirror":
            paired_reason = _mirror_reason_tensor(paired_reason)
            paired_reliability = _mirror_reason_tensor(paired_reliability)
        view_consistency = cross_view_consistency(
            output["reason_logits_global"].detach(),
            paired_reason.detach(),
            output["factor_reliability"].detach(),
            paired_reliability.detach(),
        )
        loss_view_output = dict(mirror_output)
        loss_view_output["reason_logits_final"] = (
            _mirror_reason_tensor(mirror_output["reason_logits_final"])
            if view_kind == "mirror"
            else mirror_output["reason_logits_final"]
        )
        loss_view_output["factor_reliability"] = paired_reliability
    ema_probability = output.get(
        "reason_ema_probability",
        torch.sigmoid(output["reason_logits_global"].detach()),
    )
    pu_score, negative_weight = noisy_zero_trust(
        ema_probability,
        state_positive,
        output["factor_reliability"],
        view_consistency,
    )
    reason = meter_reason_loss(
        output,
        reason_target,
        pu_score,
        config["loss_weights"],
        observability=output["factor_observability"].detach(),
        soft_positive_weight=None,
        view_output=loss_view_output,
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
            mirrored_output=mirror_output if view_kind == "mirror" else None,
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
                "null",
                "discrimination",
                "mirror",
                "ontology_identity",
                "total",
            )
        }
        grounding_total = zero
    registry = HECALossRegistry()
    action_map = {
        "action_final": "final", "action_visual": "visual",
        "action_credit_rank": "credit_rank", "action_necessity": "necessity",
        "action_specificity": "specificity", "action_nonreg": "nonreg",
        "action_soft_f1": "soft_f1", "action_cardinality": "cardinality",
        "action_logit_scale": "logit_scale",
    }
    reason_map = {
        "reason_final": "final", "reason_global": "global",
        "reason_rank": "rank", "reason_soft_f1": "soft_f1",
        "reason_correction_sign": "correction_sign",
        "reason_view_consistency": "view_consistency",
    }
    for key, part in action_map.items():
        registry.add(key, action[part], config["loss_weights"][key], owner="action")
    for key, part in reason_map.items():
        registry.add(key, reason[part], config["loss_weights"][key], owner="reason")
    if "meter_grounding" in batch:
        registry.add("anchor", grounding["anchor"], config["loss_weights"]["anchor"] * grounding_ramp, owner="measurement")
        registry.add("state", grounding["state"], config["loss_weights"]["state"] * grounding_ramp, owner="measurement")
        registry.add("observability_null", grounding["observability"] + grounding["null"], config["loss_weights"]["observability_null"] * grounding_ramp, owner="measurement")
        registry.add("discrimination", grounding["discrimination"], config["loss_weights"]["discrimination"] * grounding_ramp, owner="measurement")
        registry.add("mirror", grounding["mirror"], config["loss_weights"]["mirror"] * grounding_ramp, owner="measurement")
        registry.add("ontology_identity", grounding["ontology_identity"], config["loss_weights"]["ontology_identity"] * grounding_ramp, owner="measurement")
    registry.add("pu_private", pu, 1.0, owner="reason_private")
    total = registry.total()
    return total, {
        "action": action,
        "reason": reason,
        "grounding": grounding,
        "identity": identity,
        "identity_mode": mode,
        "pu": pu,
        "pu_score": pu_score,
        "negative_weight": negative_weight,
        "loss_registry": registry,
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


@torch.no_grad()
def _apply_pu_gate(
    audit: dict[str, Any],
    view_consistency: Tensor,
    config: dict[str, Any],
    previous_state: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    maximum = float(config["pu"].get("max_lambda", 0.15))
    minimum_lift = float(config["pu"].get("min_auprc_lift", 0.02))
    minimum_view = float(config["pu"].get("min_view_consistency", 0.50))
    required_streak = int(config["pu"].get("required_pass_streak", 2))
    start_after = int(config["pu"].get("start_after_epoch", 1))
    previous_streak = list(previous_state.get("pass_streak", [0] * 21))
    streak: list[int] = []
    gated_lambda: list[float] = []
    for index, row in enumerate(audit["labels"]):
        passed = bool(
            row.get("eligible") is True
            and float(row.get("auprc_delta", float("-inf"))) >= minimum_lift
            and float(view_consistency[index]) >= minimum_view
        )
        current_streak = previous_streak[index] + 1 if passed else 0
        streak.append(current_streak)
        active = epoch >= start_after and current_streak >= required_streak
        gated_lambda.append(min(float(audit["lambda"][index]), maximum) if active else 0.0)
        row["view_consistency"] = float(view_consistency[index])
        row["pass_streak"] = current_streak
        row["gate_active"] = active
    audit["lambda"] = gated_lambda
    audit["pass_streak"] = streak
    audit["active_labels"] = [index for index, value in enumerate(gated_lambda) if value > 0]
    return audit


@torch.no_grad()
def _update_pu(
    model: METEROIAModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    progress: float,
    config: dict[str, Any],
    *,
    previous_state: dict[str, Any],
    epoch: int,
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
    consistency_sum = torch.zeros(21)
    consistency_rows = 0
    for raw_batch in loader:
        images = raw_batch["image"].to(device, non_blocking=True)
        pair = model.forward_view_pair(
            images, torch.flip(images, dims=[-1]), progress=progress
        )
        mirrored_logits = _mirror_reason_tensor(pair["view"]["reason_logits_global"])
        mirrored_reliability = _mirror_reason_tensor(pair["view"]["factor_reliability"])
        consistency = cross_view_consistency(
            pair["original"]["reason_logits_global"],
            mirrored_logits,
            pair["original"]["factor_reliability"],
            mirrored_reliability,
        )
        consistency_sum += consistency.detach().float().cpu().sum(0)
        consistency_rows += consistency.shape[0]
    view_consistency = consistency_sum / max(consistency_rows, 1)
    return _apply_pu_gate(audit, view_consistency, config, previous_state, epoch)


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
    visual_confidence: list[list[float]] = [[] for _ in range(21)]
    provenance_valid_count = [0] * 21
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
            valid_provenance = target.get(
                "factor_provenance_valid", target["factor_observability_valid"]
            )[:, factor].bool()
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
            if bool(valid_provenance.any()):
                provenance_valid_count[factor] += int(valid_provenance.sum())
                visual_confidence[factor].extend(
                    output["factor_visual_confidence"][valid_provenance, factor]
                    .cpu()
                    .tolist()
                )
    rows: list[dict[str, Any]] = []
    for factor in range(21):
        state_p = torch.tensor(state_probability[factor])
        state_y = torch.tensor(state_target[factor])
        confidence_p = torch.tensor(visual_confidence[factor])
        state_positive_count = int(state_y.sum()) if state_y.numel() else 0
        state_negative_count = int(state_y.numel()) - state_positive_count
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
                "state_positive_count": state_positive_count,
                "state_negative_count": state_negative_count,
                "state_identifiable": bool(
                    state_positive_count >= 20 and state_negative_count >= 20
                ),
                "audit_split": "train_audit",
                "state_confusion_matrix": state_confusion[factor].tolist(),
                "provenance_valid_count": provenance_valid_count[factor],
                "visual_confidence_mean": (
                    float(confidence_p.mean()) if confidence_p.numel() else None
                ),
                "visual_confidence_std": (
                    float(confidence_p.std(unbiased=False)) if confidence_p.numel() else None
                ),
                "observability_visually_unidentifiable": True,
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
    prototype_factor_path = config["model"].get("factor_text_prototype_path")
    prototype_state_path = config["model"].get("state_text_prototype_path")
    provenance_stats_path = config["model"].get("provenance_stats_path")
    if not args.use_mock_dino:
        for required in (prototype_factor_path, prototype_state_path, provenance_stats_path):
            if not required or not Path(required).exists():
                raise FileNotFoundError(f"Missing formal HECA static artifact: {required}")
    model = METEROIAModel(
        dim=int(config["model"]["dim"]),
        action_dim=int(config["model"]["action_dim"]),
        reason_dim=int(config["model"]["reason_dim"]),
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=bool(args.use_mock_dino),
        factor_rank=int(config["model"].get("factor_rank", 16)),
        state_effect_rank=int(config["model"].get("state_effect_rank", 64)),
        schema_path=str(schema_path),
        action_correction_fraction=correction_fraction_for_run(
            args.run_kind,
            gate_c_pass=_validated_gate_pass(args.gate_c_pass),
        ),
        action_max_visual_rms=float(
            config["model"].get("action_max_visual_rms", 5.0)
        ),
        action_max_delta=float(config["model"].get("action_max_delta", 1.0)),
        action_logit_norm_cap=float(
            config["model"].get("action_logit_norm_cap", 20.0)
        ),
        action_measurement_grad_scale=float(
            config["model"].get("action_measurement_grad_scale", 0.05)
        ),
        action_allocation_logit_scale=float(
            config["model"].get("action_allocation_logit_scale", 4.0)
        ),
        action_max_rms_ratio=float(
            config["model"].get("action_max_rms_ratio", 0.20)
        ),
        reason_global_delta_cap=float(
            config["model"].get("reason_global_delta_cap", 0.05)
        ),
        factor_text_prototype_path=(
            prototype_factor_path
            if prototype_factor_path and Path(prototype_factor_path).exists()
            else None
        ),
        state_text_prototype_path=(
            prototype_state_path
            if prototype_state_path and Path(prototype_state_path).exists()
            else None
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
    schedule_state = HECAScheduleState(update=0, total_updates=total_updates)
    excess_risk = HECAExcessRiskBalancer()
    reason_probability_ema = ReasonProbabilityEMA(momentum=0.90)
    config_hash = combined_file_hash(args.config)
    source_hash = python_source_tree_hash(Path.cwd())
    schema_hash = file_hash(schema_path)
    start_epoch = 0
    optimizer_step = 0
    pu_state: dict[str, Any] = {
        "lambda": [0.0] * 21,
        "active_labels": [],
        "labels": [],
        "pass_streak": [0] * 21,
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
        heca_state = dict((payload.get("meta_state") or {}).get("heca", {}))
        if heca_state:
            schedule_state = HECAScheduleState.from_state_dict(heca_state["schedule"])
            excess_risk.load_state_dict(heca_state["excess_risk"])
            reason_probability_ema.load_state_dict(heca_state["reason_probability_ema"])
        if schedule_state.update != optimizer_step:
            raise RuntimeError("HECA resume optimizer/schedule update mismatch")
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
        "run_kind": args.run_kind,
        "gate_c_pass": str(args.gate_c_pass),
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
    best_path = output_dir / "best_metrics.json"
    if best_path.exists():
        previous_best = json.loads(best_path.read_text(encoding="utf-8"))
        for name in best:
            if name in previous_best:
                best[name] = float(previous_best[name])
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
    shared_parameters = list(model.heca_adapters.shared_adapter.parameters())
    probe_interval = int(config["training"]["gradient_ownership_probe_interval_updates"])
    if probe_interval <= 0:
        raise ValueError("HECA gradient ownership probe interval must be positive")
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_start = time.perf_counter()
        epoch_rows: list[dict[str, Any]] = []
        window_action_loss = 0.0
        window_reason_loss = 0.0
        window_microbatches = 0
        ownership_logged_epoch = False
        epoch_component_start = dict(model._component_call_counts)
        epoch_dino_start = model._encode_call_count
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
            view_kind = "mirror" if optimizer_step % 2 == 0 else "light"
            active_balance = schedule_state.shared_balance()
            with autocast():
                output, mirror_output, dino_time = _forward_training_batch(
                    model,
                    batch["image"],
                    progress=optimizer_step / max(total_updates, 1),
                    mirror_due=mirror_due,
                    view_kind=view_kind,
                    shared_action_gradient_scale=active_balance["action"],
                    shared_reason_gradient_scale=active_balance["reason"],
                )
                reason_fallback = torch.sigmoid(output["reason_logits_global"].detach())
                output["reason_ema_probability"] = reason_probability_ema.values_for(
                    [str(name) for name in batch["file_name"]], reason_fallback
                )
                corruption_step = schedule_state.corruption_microbatch_index
                total, parts = _compute_losses(
                    model,
                    output,
                    batch,
                    config=config,
                    grounding_ramp=grounding_ramp,
                    mechanism_ramp=mechanism_ramp,
                    forward_progress=optimizer_step / max(total_updates, 1),
                    pu_lambda=torch.tensor(
                        pu_state["lambda"], device=device, dtype=output["reason_logits_final"].dtype
                    ),
                    mirror_output=mirror_output,
                    corruption_step=corruption_step,
                    view_kind=view_kind,
                )
                scaled = total / grad_accum
            window_action_loss += float(parts["action"]["total"].detach())
            window_reason_loss += float(parts["reason"]["total"].detach())
            window_microbatches += 1
            is_update = (micro_step + 1) % grad_accum == 0 or micro_step + 1 == len(train_loader)
            ownership_probe: dict[str, float] | None = None
            probe_due = (
                is_update
                and optimizer_step > 0
                and (
                    not ownership_logged_epoch
                    or optimizer_step % probe_interval == 0
                )
            )
            if probe_due:
                # Ownership probes are expensive but remain exact and run at the
                # required audit cadence, never on ordinary micro-batches.
                action_shared = parts["loss_registry"].owner_total({"action"}) / grad_accum
                reason_shared = parts["loss_registry"].owner_total({"reason"}) / grad_accum
                action_credit_parameters = _action_credit_parameters(model)
                anchor_parameters = tuple(model.typed_factors.anchor_query.parameters())
                state_measurement_parameters = _state_measurement_parameters(model)
                factor_parameters = tuple(model.typed_factors.parameters())
                foundation_parameters = tuple(
                    parameter
                    for name, parameter in model.foundation.named_parameters()
                    if parameter.requires_grad and not name.startswith("dino.")
                )
                action_to_state = _autograd_norm(
                    action_shared, state_measurement_parameters
                )
                action_grads = torch.autograd.grad(
                    action_shared, shared_parameters, retain_graph=True, allow_unused=True
                )
                reason_grads = torch.autograd.grad(
                    reason_shared, shared_parameters, retain_graph=True, allow_unused=True
                )
                flat_action = torch.cat(
                    [gradient.flatten() for gradient in action_grads if gradient is not None]
                )
                flat_reason = torch.cat(
                    [gradient.flatten() for gradient in reason_grads if gradient is not None]
                )
                ownership_probe = {
                    "action_to_anchor_query": _autograd_norm(
                        action_shared, anchor_parameters
                    ),
                    "action_to_state_bridge_ratio": action_to_state
                    / max(
                        _autograd_norm(action_shared, action_credit_parameters),
                        1e-12,
                    ),
                    "action_to_state_measurement": action_to_state,
                    "action_to_credit_adapter": _autograd_norm(
                        action_shared, action_credit_parameters
                    ),
                    "reason_to_action_credit": _autograd_norm(
                        reason_shared, action_credit_parameters
                    ),
                    "pu_to_action_factor": _autograd_norm(
                        parts["pu"], (*action_credit_parameters, *factor_parameters)
                    ),
                    "measurement_to_foundation": _autograd_norm(
                        parts["loss_registry"].owner_total({"measurement"}),
                        foundation_parameters,
                    ),
                    "shared_action_reason_grad_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            flat_action, flat_reason, dim=0, eps=1e-8
                        )
                    ),
                }
            backward_start = time.perf_counter()
            scaled.backward()
            schedule_state.corruption_microbatch_index += 1
            reason_probability_ema.update(
                [str(name) for name in batch["file_name"]], reason_fallback
            )
            backward_time = time.perf_counter() - backward_start
            grad_norm = 0.0
            foundation_grad_norm = 0.0
            shared_grad_cosine = 0.0
            foundation_grad_cap = float(config["training"]["foundation_grad_cap_max"])
            next_balance = active_balance
            if is_update:
                window_action = output["action_logits_final"].new_tensor(
                    window_action_loss / max(window_microbatches, 1)
                )
                window_reason = output["reason_logits_final"].new_tensor(
                    window_reason_loss / max(window_microbatches, 1)
                )
                excess_risk.update_floors(window_action, window_reason)
                next_balance = excess_risk.weights(window_action, window_reason)
                schedule_state.set_next_shared_balance(next_balance)
                if ownership_probe is not None:
                    shared_grad_cosine = ownership_probe["shared_action_reason_grad_cosine"]
                raw_foundation_norm = torch.nn.utils.clip_grad_norm_(
                    model.foundation.parameters(), float("inf")
                )
                schedule_state.foundation_grad_ema = (
                    0.90 * schedule_state.foundation_grad_ema
                    + 0.10 * float(raw_foundation_norm)
                )
                foundation_grad_cap = schedule_state.foundation_grad_cap(
                    float(config["training"]["foundation_grad_cap_min"]),
                    float(config["training"]["foundation_grad_cap_max"]),
                )
                foundation_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.foundation.parameters(),
                        foundation_grad_cap,
                    )
                )
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["training"]["grad_clip_global"])
                    )
                )
                # Keep this with the probe's pre-step gradients, not the
                # updated weights produced by optimizer.step().
                action_state_effect_norm = float(
                    model.action_transport.state_effect_embedding.detach()
                    .float()
                    .norm()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                window_action_loss = 0.0
                window_reason_loss = 0.0
                window_microbatches = 0
                optimizer_step += 1
                schedule_state.update = optimizer_step
                schedule_state.corruption_phase = (
                    schedule_state.corruption_microbatch_index % 3
                )
                schedule_state.action_floor = excess_risk.action_floor
                schedule_state.reason_floor = excess_risk.reason_floor
                visual_rms = output["action_logits_visual"].detach().float().square().mean(0).sqrt()
                schedule_state.visual_rms_ema = (
                    0.95 * torch.tensor(schedule_state.visual_rms_ema)
                    + 0.05 * visual_rms.cpu()
                ).tolist()
                logit_rms = float(output["action_logits_visual"].detach().float().square().mean().sqrt())
                for group, scheduled_lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                    if group.get("group_name") == "foundation":
                        group["lr"] = scheduled_lr * schedule_state.foundation_lr_multiplier(logit_rms=logit_rms)
                if ownership_probe is not None:
                    append_jsonl(
                        output_dir / "heca_gradient_ownership.jsonl",
                        {
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "action_credit_ramp": float(
                                output["action_credit_ramp"].detach()
                            ),
                            "action_state_effect_norm": action_state_effect_norm,
                            "shared_action_weight": active_balance["action"],
                            "shared_reason_weight": active_balance["reason"],
                            "next_shared_action_weight": next_balance["action"],
                            "next_shared_reason_weight": next_balance["reason"],
                            "shared_action_reason_grad_cosine": shared_grad_cosine,
                            "foundation_grad_norm": foundation_grad_norm,
                            "foundation_grad_cap": foundation_grad_cap,
                            **ownership_probe,
                        },
                    )
                    ownership_logged_epoch = True
                write_json(
                    output_dir / "heca_schedule_state.json",
                    {
                        "optimizer_step": optimizer_step,
                        "progress": schedule_state.progress(),
                        "credit_ramp": float(output["action_credit_ramp"].detach()),
                        "foundation_grad_cap": foundation_grad_cap,
                        "excess_risk": {
                            "action": float(excess_risk.action_floor or 0.0),
                            "reason": float(excess_risk.reason_floor or 0.0),
                        },
                        "state": schedule_state.state_dict(),
                    },
                )
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
                "loss_action_specificity": float(
                    parts["action"]["specificity"].detach()
                ),
                "loss_action_credit_rank": float(parts["action"]["credit_rank"].detach()),
                "loss_action_necessity": float(parts["action"]["necessity"].detach()),
                "action_target_effect_active_count": float(
                    parts["action"]["necessity_active_count"].detach()
                ),
                "action_target_effect_active_fraction": float(
                    parts["action"]["necessity_active_fraction"].detach()
                ),
                "action_target_effect_support_mean": float(
                    parts["action"]["necessity_support_mean"].detach()
                ),
                "action_target_effect_required_margin_mean": float(
                    parts["action"]["necessity_required_margin_mean"].detach()
                ),
                "action_target_effect_mean": float(
                    parts["action"]["necessity_target_effect_mean"].detach()
                ),
                "action_target_directional_effect_mean": float(
                    parts["action"]["necessity_directional_effect_mean"].detach()
                ),
                "loss_action_target_contrastive": float(
                    parts["action"]["necessity_contrastive_loss"].detach()
                ),
                "loss_action_target_directional": float(
                    parts["action"]["necessity_directional_loss"].detach()
                ),
                "loss_action_nonreg": float(parts["action"]["nonreg"].detach()),
                "loss_action_logit_scale": float(parts["action"]["logit_scale"].detach()),
                "loss_identity": float(parts["identity"].detach()),
                "identity_mode": parts["identity_mode"],
                "loss_pu": float(parts["pu"].detach()),
                "grounding_ramp": grounding_ramp,
                "mechanism_ramp": mechanism_ramp,
                "grad_norm": grad_norm,
                "foundation_grad_norm": foundation_grad_norm,
                "foundation_grad_cap": foundation_grad_cap,
                "foundation_grad_ema": schedule_state.foundation_grad_ema,
                "shared_action_weight": active_balance["action"],
                "shared_reason_weight": active_balance["reason"],
                "next_shared_action_weight": next_balance["action"],
                "next_shared_reason_weight": next_balance["reason"],
                "shared_action_reason_grad_cosine": shared_grad_cosine,
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
                "action_visual_preclip_norm_max": float(output["action_visual_preclip_norm"].detach().max()),
                "action_emergency_cap_rate": float(
                    output["action_emergency_cap_active"].float().mean().detach()
                ),
                "factor_null_mean": float(output["factor_null_mass"].mean().detach()),
                "factor_visual_confidence_mean": float(
                    output["factor_visual_confidence"].mean().detach()
                ),
                "loss_wiring": parts["loss_registry"].artifact(),
            }
            epoch_rows.append(row)
            append_jsonl(output_dir / "loss_components.jsonl", row)
            if not (output_dir / "heca_loss_wiring.json").exists():
                loss_rows = parts["loss_registry"].artifact()
                registry = [item["term"] for item in loss_rows]
                write_json(
                    output_dir / "heca_loss_wiring.json",
                    {
                        "registry": registry,
                        "counts": {
                            item["term"]: int(item["call_count"])
                            for item in loss_rows
                        },
                        "duplicates": [],
                        "pass": len(registry) == len(set(registry))
                        and all(item["call_count"] == 1 for item in loss_rows),
                        "terms": loss_rows,
                    },
                )
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
            meta_state={
                "training_enabled": True,
                "audit_only": False,
                "heca": {
                    "schedule": schedule_state.state_dict(),
                    "excess_risk": excess_risk.state_dict(),
                    "reason_probability_ema": reason_probability_ema.state_dict(),
                },
            },
            pu_state=pu_state,
            calibration=_calibration_payload(calibration),
            config_hash=config_hash,
            source_hash=source_hash,
            schema_hash=schema_hash,
        )
        pu_state = _update_pu(
            model,
            audit_loader,
            device,
            progress,
            config,
            previous_state=pu_state,
            epoch=epoch,
        )
        schedule_state.pu_pass_streak = list(pu_state.get("pass_streak", [0] * 21))
        calibration_data = _collect_calibration(
            model, calib_loader, device, progress
        )
        calibration = _fit_calibration(model, calibration_data)
        diagnostic_due = _diagnostic_due(
            epoch,
            epochs,
            int(config.get("evaluation", {}).get("patch_deletion_interval_epochs", 4)),
        )
        test = collect_outputs(
            model,
            test_loader,
            device,
            progress=progress,
            sequential_modes=True,
            extra_diagnostic_modes=diagnostic_due,
        )
        test_branches = branch_metrics(test)
        mechanism = mechanism_stats_from_collected(test)
        mechanism.update(
            {
                "diagnostic_due": diagnostic_due,
                "factor_visual_confidence_mean": sum(
                    row["factor_visual_confidence_mean"] for row in epoch_rows
                ) / max(len(epoch_rows), 1),
                "action_correction_rms_ratio_mean": [
                    sum(row["action_correction_rms_ratio"][action] for row in epoch_rows)
                    / max(len(epoch_rows), 1)
                    for action in range(4)
                ],
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
                    int(config.get("evaluation", {}).get("patch_audit_max_unique", 128)),
                    len(split["audit"]),
                ),
                patches_per_factor=int(
                    config.get("evaluation", {}).get(
                        "patch_audit_patches_per_factor", 12
                    )
                ),
                factors_per_action=int(
                    config.get("evaluation", {}).get(
                        "patch_audit_factors_per_action", 2
                    )
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
        component_delta = {
            name: int(model._component_call_counts[name] - epoch_component_start[name])
            for name in epoch_component_start
        }
        write_json(
            output_dir / "heca_component_call_counters.json",
            {
                "components": {
                    "dino_encode": int(model._encode_call_count - epoch_dino_start),
                    **component_delta,
                },
                "one_dino_encode_per_batch": (
                    model._encode_call_count
                    == int(getattr(model.foundation, "ordinary_dino_calls", -1))
                ),
            },
        )
        contribution = test["mechanism"]["action_factor_contributions"].float()
        credit = contribution.sum(-1)
        for action_index in range(4):
            summed = float(contribution[:, action_index].sum(-1).mean())
            expected = float(credit[:, action_index].mean())
            append_jsonl(
                output_dir / "heca_contribution_conservation.jsonl",
                {
                    "epoch": epoch,
                    "action": action_index,
                    "sum_contribution": summed,
                    "action_credit_sum": expected,
                    "abs_error": abs(summed - expected),
                },
            )
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
            "meta_state": {
                "training_enabled": True,
                "audit_only": False,
                "heca": {
                    "schedule": schedule_state.state_dict(),
                    "excess_risk": excess_risk.state_dict(),
                    "reason_probability_ema": reason_probability_ema.state_dict(),
                },
            },
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
        output_dir / "GOAL_COMPLETED_METER_OIA_V3_HECA.json",
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
        default="configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml",
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
    parser.add_argument("--run_kind", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--gate_c_pass", default="")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    if args.require_no_token_compression and config["model"]["token_compression"] != "none":
        raise RuntimeError("Token compression is forbidden")
    if args.no_feature_cache and config["model"]["feature_cache_enabled"]:
        raise RuntimeError("Feature cache is forbidden")
    if args.run_kind == "full":
        validate_formal_protocol(
            {
                "from_scratch": not bool(args.init_model_checkpoint),
                "epochs": int(args.epochs or config["training"]["epochs"]),
                "pilot_checkpoint": args.init_model_checkpoint,
            }
        )
    train(config, args)


if __name__ == "__main__":
    main()
