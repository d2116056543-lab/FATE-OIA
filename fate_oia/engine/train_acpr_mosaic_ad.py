from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.datasets.mosaic_train_calib_split import make_multilabel_train_calib_indices
from fate_oia.engine.eval_acpr_mosaic_ad import evaluate_mosaic
from fate_oia.engine.export_mosaic_visual_audit import _weak_factor_score
from fate_oia.engine.mosaic_schedule import MOSAICPhaseControls, mosaic_phase_controls
from fate_oia.losses.mosaic_action_losses import build_mosaic_action_loss
from fate_oia.losses.mosaic_factor_losses import build_mosaic_factor_loss
from fate_oia.losses.mosaic_posterior_ranking import (
    action_cross_image_ranking_loss,
    posterior_weighted_reason_ranking_loss,
)
from fate_oia.losses.mosaic_reason_observation_losses import build_mosaic_reason_loss
from fate_oia.losses.mosaic_state_losses import build_mosaic_state_loss
from fate_oia.metrics import binary_average_precision
from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel
from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead
from fate_oia.models.mosaic_selective_observation import MOSAICSelectiveObservationModel
from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue
from fate_oia.transforms import AspectRatioLetterboxTransform, IMAGENET_MEAN, IMAGENET_STD
from fate_oia.utils.mosaic_artifacts import (
    EPOCH_JSONL_FILES,
    EPOCH_JSON_FILES,
    LOGIT_FILES,
    append_jsonl,
    initialize_run_artifacts,
    write_epoch_artifacts,
    write_json,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if config.get("training", {}).get("epochs") != 15:
        raise ValueError("formal MOSAIC-AD training requires exactly 15 epochs")
    if config.get("evaluation", {}).get("eval_splits") != ["test"]:
        raise ValueError("formal MOSAIC-AD evaluation is test-only")
    if config.get("backbone", {}).get("feature_cache") is not False:
        raise ValueError("feature cache must be disabled")
    if config.get("backbone", {}).get("token_compression") != "none":
        raise ValueError("token compression must be none")
    expected_contract = {
        "experiment": {
            "name": "acpr_mosaic_ad_v1",
            "direct_image": True,
            "initialization": "public_dino_vits8_teacher_only",
            "no_metric_early_stop": True,
            "eval_splits": ["test"],
            "best_selection_split": "test",
            "best_selection_metric": "deploy_fixed_joint",
        },
        "data": {
            "image_height": 360,
            "image_width": 640,
            "patch_size": 8,
            "action_dim": 4,
            "reason_dim": 21,
            "train_calib_fraction": 0.10,
            "train_calib_seed": 20260710,
            "train_calib_min_positive_per_label": 20,
            "num_workers_candidates": [2, 4, 0],
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
        },
        "backbone": {
            "arch": "vit_small",
            "patch_size": 8,
            "selected_layers": [3, 7, 11],
            "checkpoint_key": "teacher",
            "freeze_backbone": True,
            "no_grad_backbone": True,
            "feature_cache": False,
            "token_compression": "none",
        },
        "model": {
            "formal_class": "MOSAICADModel",
            "dim": 384,
            "decoder_layers": 2,
            "self_attention_heads": 4,
            "highres_topk": 256,
            "midres_topk": 128,
            "anchors_per_factor": 2,
            "typed_attention_heads": 4,
            "point_samples": 8,
            "curve_samples": 12,
            "region_samples": 12,
            "spatial_prior_scale_init": 0.05,
            "spatial_prior_scale_max": 0.20,
            "spatial_prior_dropout": 0.50,
            "content_temperature_init": 0.07,
            "action_state_gate_cap_max": 0.25,
            "state_residual_cap": 0.20,
            "reason_state_contribution_cap_max": 0.20,
        },
        "selective_observation": {
            "pi_min": 0.20,
            "pi_max": 0.95,
            "false_positive_max": 0.05,
            "group_names": ["traffic_control", "obstacle", "lane", "other"],
            "synthetic_missing_positive_fraction": 0.20,
            "posterior_queue_size": 2048,
        },
        "loss": {
            "action": {
                "gamma_pos": 0.0,
                "gamma_neg": 4.0,
                "clip": 0.05,
                "rank_weight": 0.10,
                "cardinality_weight": 0.02,
            },
            "factor": {
                "geometry_mask_weight": 0.10,
                "view_consistency_weight": 0.05,
                "flip_equivariance_weight": 0.05,
                "prototype_occupancy_weight": 0.02,
                "prototype_repulsion_weight": 0.01,
                "prior_scale_weight": 0.01,
                "contradiction_weight": 0.02,
            },
            "state": {
                "sparsity_weight": 0.02,
                "residual_weight": 0.02,
                "uncertainty_weight": 0.02,
            },
            "reason": {
                "posterior_bce_weight": 0.30,
                "posterior_rank_weight": 0.08,
                "missing_recovery_weight": 0.10,
                "latent_rate_range_weight": 0.03,
                "propensity_regularization_weight": 0.01,
            },
        },
        "optimizer": {
            "name": "AdamW",
            "fused_when_supported": True,
            "weight_decay": 0.05,
            "grad_clip": 1.0,
            "scheduler": "warmup_cosine",
            "warmup_epochs": 1,
            "min_lr_ratio": 0.05,
            "aux_shared_lambda_max": 0.25,
            "action_anchor_kappa": 0.70,
            "lr": {
                "visual_pyramid": 0.00010,
                "action_adapter": 0.00015,
                "reason_adapter": 0.00015,
                "factor_measurement": 0.00020,
                "prototypes": 0.00010,
                "state_composer": 0.00020,
                "action_decoder": 0.00020,
                "reason_decoder": 0.00020,
                "propensity": 0.00010,
                "threshold": 0.00050,
            },
        },
        "training": {
            "epochs": 15,
            "bf16": True,
            "tf32": True,
            "torch_compile": False,
            "no_metric_early_stop": True,
            "factor_audit_epochs": [5, 8, 11],
            "factor_audit_samples": 128,
            "runtime_candidates": [
                {"batch_size": 8, "grad_accum": 4, "effective_batch": 32},
                {"batch_size": 6, "grad_accum": 5, "effective_batch": 30},
                {"batch_size": 5, "grad_accum": 6, "effective_batch": 30},
                {"batch_size": 4, "grad_accum": 8, "effective_batch": 32},
            ],
            "max_reserved_vram_gb": 43.0,
        },
        "calibration": {
            "enabled": True,
            "steps_per_epoch": 100,
            "batch_size": 256,
            "surrogate_temperature": 0.20,
            "soft_f1_weight": 1.00,
            "bce_weight": 0.05,
            "rate_weight": 0.02,
            "delta_weight": 0.01,
            "cardinality_weight": 0.02,
            "label_delta_max": 1.0,
            "groups": ["actions", "common_reasons", "tail_reasons"],
            "tail_reason_indices": [12, 9, 5, 14, 6, 11, 10, 13],
            "train_calib_only": True,
            "representation_frozen": True,
            "deploy_equation": "raw_minus_theta",
            "test_oracle_diagnostic_only": True,
        },
        "evaluation": {
            "eval_splits": ["test"],
            "eval_every_epoch": True,
            "best_selection_split": "test",
            "best_selection_metric": "deploy_fixed_joint",
            "threshold": 0.5,
            "report_raw": True,
            "report_deploy_fixed": True,
            "report_train_calib_teacher_diagnostic": True,
            "report_test_oracle_diagnostic": True,
        },
        "runtime": {
            "foreground_only": True,
            "no_feature_cache": True,
            "require_no_token_compression": True,
            "require_runtime_selection": True,
            "runtime_selection_path": ".review/mosaic_runtime_selection.json",
            "warmup_profile_steps": 20,
            "timed_profile_steps": 100,
            "dataloader_stability_minutes": 15,
        },
    }

    def require_contract(actual: dict[str, Any], expected: dict[str, Any], prefix: str = "") -> None:
        for key, expected_value in expected.items():
            path_name = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                raise ValueError(f"formal MOSAIC config missing {path_name}")
            actual_value = actual[key]
            if isinstance(expected_value, dict):
                if not isinstance(actual_value, dict):
                    raise ValueError(f"formal MOSAIC config {path_name} must be a mapping")
                require_contract(actual_value, expected_value, path_name)
            elif actual_value != expected_value:
                raise ValueError(
                    f"formal MOSAIC config drift at {path_name}: expected {expected_value!r}, got {actual_value!r}"
                )

    require_contract(config, expected_contract)
    calibration_steps = config.get("calibration", {}).get("steps_per_epoch")
    if type(calibration_steps) is not int or calibration_steps <= 0:
        raise ValueError("calibration.steps_per_epoch must be a positive fixed integer")
    return config


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "action": torch.stack([row["action"] for row in batch]),
        "reason": torch.stack([row["reason"] for row in batch]),
        "file_name": [row["file_name"] for row in batch],
        "image_path": [row["image_path"] for row in batch],
        "split": [row["split"] for row in batch],
    }


def _loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    config: dict[str, Any],
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(config["data"].get("pin_memory", True)),
        "collate_fn": _collate,
        "drop_last": bool(shuffle),
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config["data"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
        kwargs["timeout"] = 900
    return DataLoader(dataset, **kwargs)


def _detach_artifact_value(value: Any) -> Any:
    """Detach per-step diagnostics immediately so epoch logs cannot retain autograd graphs."""
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, dict):
        return {str(key): _detach_artifact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach_artifact_value(item) for item in value]
    return value


def _sample_multilabel_vector(sample: Any) -> tuple[int, ...]:
    action = sample["action"] if isinstance(sample, dict) else sample.action
    reason = sample["reason"] if isinstance(sample, dict) else sample.reason
    values = tuple(action) + tuple(reason)
    if len(values) != 25 or any(float(value) not in {0.0, 1.0} for value in values):
        raise ValueError("MOSAIC stratification requires binary 4+21 metadata labels")
    return tuple(int(value) for value in values)


def _stratified_subset_indices(
    dataset,
    indices: list[int],
    limit: int | None,
    *,
    seed: int,
) -> list[int]:
    """Build a deterministic multi-label subset without decoding images."""
    if limit is None or limit >= len(indices):
        return list(indices)
    if type(limit) is not int or limit <= 0:
        raise ValueError("stratified subset limit must be a positive integer")
    samples = dataset.samples
    labels = {index: _sample_multilabel_vector(samples[index]) for index in indices}
    availability = [sum(labels[index][label] for index in indices) for label in range(25)]
    fraction = limit / len(indices)
    targets = [min(count, max(1, int(round(count * fraction)))) if count else 0 for count in availability]

    def stable_rank(index: int) -> tuple[int, int]:
        sample = samples[index]
        file_name = sample["file_name"] if isinstance(sample, dict) else sample.file_name
        digest = hashlib.sha256(f"{seed}:{file_name}".encode("utf-8")).digest()
        return int.from_bytes(digest, "big", signed=False), index

    ordered_by_label = [
        sorted((index for index in indices if labels[index][label]), key=stable_rank)
        for label in range(25)
    ]
    pointers = [0] * 25
    selected: set[int] = set()
    counts = [0] * 25
    while len(selected) < limit:
        deficits = [max(targets[label] - counts[label], 0) for label in range(25)]
        candidates = [label for label in range(25) if deficits[label] > 0]
        if not candidates:
            break
        label = max(candidates, key=lambda item: (deficits[item] / max(targets[item], 1), deficits[item], -item))
        pool = ordered_by_label[label]
        while pointers[label] < len(pool) and pool[pointers[label]] in selected:
            pointers[label] += 1
        if pointers[label] >= len(pool):
            targets[label] = counts[label]
            continue
        chosen = pool[pointers[label]]
        pointers[label] += 1
        selected.add(chosen)
        for item, value in enumerate(labels[chosen]):
            counts[item] += value
    if len(selected) < limit:
        for index in sorted(indices, key=stable_rank):
            if index not in selected:
                selected.add(index)
                if len(selected) == limit:
                    break
    result = sorted(selected)
    if len(result) != limit:
        raise RuntimeError("deterministic MOSAIC stratification returned the wrong subset size")
    if any(targets[label] > 0 and not any(labels[index][label] for index in result) for label in range(25)):
        raise RuntimeError("deterministic MOSAIC stratification lost supported labels")
    return result


def _positive_counts(dataset, indices: list[int]) -> list[int]:
    return [
        sum(_sample_multilabel_vector(dataset.samples[index])[label] for index in indices)
        for label in range(25)
    ]


def build_loaders(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    max_train_samples: int | None = None,
    max_calib_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 20260710,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    data = config["data"]
    transform = AspectRatioLetterboxTransform(
        data["image_height"], data["image_width"], patch_size=data["patch_size"], normalize=True
    )
    train = BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split="train", action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )
    test = BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split="test", action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )
    split_dir = Path(output_dir) / "split"
    main_indices, calib_indices = make_multilabel_train_calib_indices(
        train,
        calib_fraction=float(data["train_calib_fraction"]),
        seed=int(data["train_calib_seed"]),
        min_calib_positives=int(data["train_calib_min_positive_per_label"]),
        output_dir=split_dir,
    )
    main_indices = _stratified_subset_indices(train, main_indices, max_train_samples, seed=seed)
    calib_indices = _stratified_subset_indices(train, calib_indices, max_calib_samples, seed=seed + 1)
    test_indices = list(range(len(test)))
    if max_test_samples is not None:
        test_indices = test_indices[: max(1, min(max_test_samples, len(test_indices)))]
    train_main = Subset(train, main_indices)
    train_calib = Subset(train, calib_indices)
    test_subset = Subset(test, test_indices)
    split_hash = json.loads((split_dir / "train_calib_split_hash.json").read_text(encoding="utf-8"))
    split_stats = json.loads((split_dir / "train_calib_split_stats.json").read_text(encoding="utf-8"))
    split_stats["split_hash"] = split_hash["split_sha256"]
    split_stats["effective_main_count"] = len(train_main)
    split_stats["effective_calib_count"] = len(train_calib)
    split_stats["effective_test_count"] = len(test_subset)
    split_stats["effective_main_positive_counts"] = _positive_counts(train, main_indices)
    split_stats["effective_calib_positive_counts"] = _positive_counts(train, calib_indices)
    split_stats["effective_calib_all_supported"] = all(
        count > 0 for count in split_stats["effective_calib_positive_counts"]
    )
    return (
        _loader(train_main, batch_size=batch_size, shuffle=True, num_workers=num_workers, config=config, generator=torch.Generator().manual_seed(seed)),
        _loader(
            train_calib,
            batch_size=int(config["calibration"]["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            config=config,
        ),
        _loader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, config=config),
        split_stats,
    )


def _normalize_view(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    pixels = (image * std + mean).clamp(0.0, 1.0)
    return pixels, mean, std


def _make_weak_views(
    images: torch.Tensor,
    multiview: MOSAICWeakMultiView,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    first, second, first_meta, second_meta = [], [], [], []
    for image in images:
        pixels, mean, std = _normalize_view(image)
        output = multiview(pixels)
        first.append((output["images"][0] - mean) / std)
        second.append((output["images"][1] - mean) / std)
        first_meta.append(output["metadata"][0])
        second_meta.append(output["metadata"][1])
    return torch.stack(first), torch.stack(second), first_meta, second_meta


def _restore_factor_tensor(
    tensor: torch.Tensor,
    metadata: list[dict[str, Any]],
    multiview: MOSAICWeakMultiView,
    *,
    is_mask: bool = False,
) -> torch.Tensor:
    restored = []
    for index, item in enumerate(metadata):
        value = tensor[index]
        if is_mask:
            value = multiview.invert_factor_masks(value, item, factor_dim=0)
        else:
            value = multiview._restore_factor_axis(value, item, 0)
        restored.append(value)
    return torch.stack(restored)


def _restore_label_logits(
    logits: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    reason: bool,
) -> torch.Tensor:
    if reason:
        permutation = list(range(21))
        for left, right in ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20)):
            permutation[left], permutation[right] = permutation[right], permutation[left]
    else:
        permutation = [0, 1, 3, 2]
    restored = [
        logits[index, permutation] if item["horizontal_flip"] else logits[index]
        for index, item in enumerate(metadata)
    ]
    return torch.stack(restored)


def _canonical_factor_predictions(
    output: dict[str, Any],
    metadata: list[dict[str, Any]],
    multiview: MOSAICWeakMultiView,
) -> dict[str, Any]:
    result = dict(output)
    for name in (
        "factor_presence_logits", "factor_presence_prob", "factor_visibility_logits",
        "factor_visibility_prob", "factor_positive_evidence", "factor_negative_evidence",
        "factor_uncertainty", "factor_features", "prototype_weights",
    ):
        result[name] = _restore_factor_tensor(output[name], metadata, multiview)
    result["factor_soft_masks"] = _restore_factor_tensor(
        output["factor_soft_masks"], metadata, multiview, is_mask=True
    )
    return result


def build_model_components(config: dict[str, Any], config_path: str | Path, device: torch.device):
    model_cfg = config["model"]
    model = MOSAICADModel(
        config_root=Path(config_path).parent,
        backbone_arch=str(config["backbone"]["arch"]),
        backbone_patch_size=int(config["backbone"]["patch_size"]),
        selected_layers=tuple(int(value) for value in config["backbone"]["selected_layers"]),
        checkpoint_key=str(config["backbone"]["checkpoint_key"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        decoder_layers=int(model_cfg["decoder_layers"]),
        self_attention_heads=int(model_cfg["self_attention_heads"]),
        highres_topk=int(model_cfg["highres_topk"]),
        midres_topk=int(model_cfg["midres_topk"]),
        anchors_per_factor=int(model_cfg["anchors_per_factor"]),
        typed_attention_heads=int(model_cfg["typed_attention_heads"]),
        point_samples=int(model_cfg["point_samples"]),
        curve_samples=int(model_cfg["curve_samples"]),
        region_samples=int(model_cfg["region_samples"]),
        spatial_prior_scale_init=float(model_cfg["spatial_prior_scale_init"]),
        spatial_prior_scale_max=float(model_cfg["spatial_prior_scale_max"]),
        spatial_prior_dropout=float(model_cfg["spatial_prior_dropout"]),
        content_temperature_init=float(model_cfg["content_temperature_init"]),
        state_residual_cap=float(model_cfg["state_residual_cap"]),
    ).to(device)
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    selective = MOSAICSelectiveObservationModel(
        factor_names,
        model.schema_bundle["reason_observation"],
        pi_min=float(config["selective_observation"]["pi_min"]),
        pi_max=float(config["selective_observation"]["pi_max"]),
        global_false_positive_max=float(config["selective_observation"]["false_positive_max"]),
    ).to(device)
    if tuple(config["selective_observation"]["group_names"]) != selective.GROUPS:
        raise ValueError("selective-observation group names do not match the formal model")
    threshold = MOSAICGroupThresholdHead(
        tail_reason_indices=config["calibration"]["tail_reason_indices"],
        label_delta_max=float(config["calibration"]["label_delta_max"]),
    ).to(device)
    action_queue = MOSAICSoftRankQueue(
        4, capacity=int(config["selective_observation"]["posterior_queue_size"])
    ).to(device)
    reason_queue = MOSAICSoftRankQueue(
        21, capacity=int(config["selective_observation"]["posterior_queue_size"])
    ).to(device)
    return model, selective, threshold, action_queue, reason_queue


def _unique_parameters(modules: Iterable[torch.nn.Module], *, exclude: set[int] | None = None):
    seen = set(exclude or ())
    result = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def build_optimizers(model, selective, threshold, config):
    lr = config["optimizer"]["lr"]
    prototype = [model.observable_predicates.prototype_bank.prototypes]
    prototype_ids = {id(parameter) for parameter in prototype}
    groups = [
        {"params": model.visual_pyramid.parameters(), "lr": lr["visual_pyramid"], "name": "visual_pyramid"},
        {"params": model.action_adapter.parameters(), "lr": lr["action_adapter"], "name": "action_adapter"},
        {"params": model.reason_adapter.parameters(), "lr": lr["reason_adapter"], "name": "reason_adapter"},
        {"params": _unique_parameters([model.observable_predicates], exclude=prototype_ids), "lr": lr["factor_measurement"], "name": "factor_measurement"},
        {"params": prototype, "lr": lr["prototypes"], "name": "prototypes"},
        {"params": model.state_composer.parameters(), "lr": lr["state_composer"], "name": "state_composer"},
        {"params": model.action_decoder.parameters(), "lr": lr["action_decoder"], "name": "action_decoder"},
        {"params": model.reason_decoder.parameters(), "lr": lr["reason_decoder"], "name": "reason_decoder"},
        {"params": selective.parameters(), "lr": lr["propensity"], "name": "propensity"},
    ]
    for group in groups:
        group["params"] = list(group["params"])
        group["base_lr"] = float(group["lr"])
    kwargs = {"lr": 1e-4, "weight_decay": float(config["optimizer"]["weight_decay"])}
    if torch.cuda.is_available():
        kwargs["fused"] = bool(config["optimizer"].get("fused_when_supported", True))
    try:
        representation = torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        representation = torch.optim.AdamW(groups, **kwargs)
    calibration = torch.optim.AdamW(
        threshold.parameters(), lr=float(lr["threshold"]), weight_decay=0.0
    )
    return representation, calibration


def _set_representation_lrs(
    optimizer,
    update: int,
    total_updates: int,
    warmup_updates: int,
    phase_scale: float,
    min_lr_ratio: float,
):
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0,1]")
    if update < warmup_updates:
        multiplier = (update + 1) / max(warmup_updates, 1)
    else:
        progress = (update - warmup_updates) / max(total_updates - warmup_updates, 1)
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier * phase_scale


def _positive_anchor_reason_loss(logits: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    positive = observed > 0.5
    values = F.softplus(-logits)
    count = positive.sum()
    return torch.where(count > 0, (values * positive).sum() / count.clamp_min(1), logits.sum() * 0.0)


def _prototype_cosine(model: MOSAICADModel) -> torch.Tensor:
    prototypes = F.normalize(model.observable_predicates.prototype_bank.prototypes, dim=-1, eps=1e-6)
    return torch.einsum("fkd,fjd->fkj", prototypes, prototypes)


def _grounding_records(index: BDD100KGroundingIndex, file_names: list[str]) -> list[dict[str, Any]]:
    records = []
    for file_name in file_names:
        paths = index.lookup(file_name)
        records.append(
            {
                "label_json": paths.label_json,
                "drivable_map": paths.drivable_map,
                "image_size": (720, 1280),
            }
            if paths.label_json or paths.drivable_map
            else None
        )
    return records


def _partition_parameters(model: MOSAICADModel, selective: MOSAICSelectiveObservationModel):
    shared = _unique_parameters(
        [model.visual_pyramid, model.reason_adapter, model.observable_predicates, model.state_composer]
    )
    shared_ids = {id(parameter) for parameter in shared}
    action_only = _unique_parameters([model.action_adapter, model.action_decoder], exclude=shared_ids)
    explanation_only = _unique_parameters(
        [model.reason_adapter, model.reason_decoder, selective], exclude=shared_ids | {id(p) for p in action_only}
    )
    return shared, action_only, explanation_only


def _apply_phase(model, selective, optimizer, controls: MOSAICPhaseControls):
    model.set_phase_controls(
        state_residual_scale=controls.state_residual_scale,
        action_state_gate_cap=controls.action_state_gate_cap,
        reason_state_contribution_cap=controls.reason_state_contribution_cap,
    )
    for parameter in selective.parameters():
        parameter.requires_grad_(controls.learned_propensity and not controls.calibration_only)
    model.observable_predicates.prototype_bank.prototypes.requires_grad_(not controls.freeze_factor_prototypes and not controls.calibration_only)
    prototype_id = id(model.observable_predicates.prototype_bank.prototypes)
    for parameter in model.parameters():
        if id(parameter) != prototype_id:
            parameter.requires_grad_(not controls.calibration_only)
    for parameter in model.dino.parameters():
        parameter.requires_grad_(False)
    if controls.freeze_factor_prototypes:
        model.observable_predicates.prototype_bank.prototypes.requires_grad_(False)
    for group in optimizer.param_groups:
        if group["name"] == "propensity" and controls.freeze_propensity_groups:
            for parameter in group["params"]:
                parameter.requires_grad_(False)


def train_representation_epoch(
    *,
    model,
    selective,
    action_queue,
    reason_queue,
    loader,
    optimizer,
    action_anchor,
    grounding_builder,
    grounding_index,
    multiview,
    controls,
    config,
    device,
    epoch,
    grad_accum,
    global_update,
    total_updates,
    profile_timing: bool = False,
    dataloader_stall_threshold_sec: float = 5.0,
):
    model.train()
    selective.train()
    optimizer.zero_grad(set_to_none=True)
    rows = {name: [] for name in EPOCH_JSONL_FILES}
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    shared, action_only, explanation_only = _partition_parameters(model, selective)
    warmup_updates = max(1, len(loader) * int(config["optimizer"]["warmup_epochs"]) // grad_accum)
    recovery_scores: list[torch.Tensor] = []
    recovery_targets: list[torch.Tensor] = []
    iterator = iter(loader)
    for step in range(len(loader)):
        row_offsets = {name: len(values) for name, values in rows.items()}
        load_started = time.perf_counter()
        batch = next(iterator)
        load_time_sec = time.perf_counter() - load_started
        step_started = time.perf_counter()
        cuda_start = cuda_end = None
        if profile_timing and device.type == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        images = batch["image"].to(device, non_blocking=True)
        actions = batch["action"].to(device, non_blocking=True)
        reasons = batch["reason"].to(device, non_blocking=True)
        view1, view2, metadata1, metadata2 = _make_weak_views(images, multiview)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and bool(config["training"]["bf16"]),
        ):
            raw1 = model(view1, return_masks=True)
            raw2 = model(view2, return_masks=True)
            out1 = _canonical_factor_predictions(raw1, metadata1, multiview)
            out2 = _canonical_factor_predictions(raw2, metadata2, multiview)
            action_logits = _restore_label_logits(raw1["action_logits_raw"], metadata1, reason=False)
            reason_logits = _restore_label_logits(raw1["reason_logits_latent"], metadata1, reason=True)
            action_rank, action_rank_stats = action_cross_image_ranking_loss(
                action_logits, actions, batch["file_name"], action_queue
            )
            action_loss_config = config["loss"]["action"]
            action_losses = build_mosaic_action_loss(
                action_logits.float(),
                actions,
                rank_loss=action_rank.float(),
                gamma_pos=float(action_loss_config["gamma_pos"]),
                gamma_neg=float(action_loss_config["gamma_neg"]),
                clip=float(action_loss_config["clip"]),
                rank_weight=float(action_loss_config["rank_weight"]),
                cardinality_weight=float(action_loss_config["cardinality_weight"]),
            )
            observations = grounding_builder(
                reasons, _grounding_records(grounding_index, batch["file_name"]), split="train"
            )
            factor_predictions = {
                **out1,
                "view_factor_presence_prob": out2["factor_presence_prob"],
                "view_factor_visibility_prob": out2["factor_visibility_prob"],
                "view_consistency_valid": torch.ones_like(out1["factor_presence_prob"], dtype=torch.bool),
                "flip_factor_soft_masks_aligned": out2["factor_soft_masks"],
                "flip_equivariance_valid": torch.ones_like(out1["factor_soft_masks"], dtype=torch.bool),
                "prototype_valid_mask": model.observable_predicates.prototype_bank.prototype_valid_mask,
                "prototype_pairwise_cosine": _prototype_cosine(model),
                "factor_contradiction_mask": model.factor_contradiction_mask,
            }
            factor_loss_config = config["loss"]["factor"]
            factor_losses = build_mosaic_factor_loss(
                factor_predictions,
                observations,
                **{name: float(value) for name, value in factor_loss_config.items()},
            )
            state_loss_config = config["loss"]["state"]
            state_losses = build_mosaic_state_loss(
                raw1,
                **{name: float(value) for name, value in state_loss_config.items()},
            )
            if controls.synthetic_missing_positive:
                observed_for_likelihood, hidden_mask = selective.hide_observed_positives(
                    reasons, hide_fraction=float(config["selective_observation"]["synthetic_missing_positive_fraction"])
                )
            else:
                observed_for_likelihood = reasons
                hidden_mask = torch.zeros_like(reasons, dtype=torch.bool)
            selective_output = selective(
                reason_logits.float(),
                observed_for_likelihood,
                out1["factor_visibility_prob"].float(),
                out1["factor_uncertainty"].float(),
            )
            if controls.posterior_enabled:
                reason_rank, reason_rank_stats = posterior_weighted_reason_ranking_loss(
                    reason_logits.float(), selective_output["reason_latent_posterior"], batch["file_name"], reason_queue
                )
                reason_losses = build_mosaic_reason_loss(
                    reason_logits.float(), observed_for_likelihood,
                    selective_output["reason_observation_prob"].float(),
                    selective_output["reason_latent_posterior"],
                    selective_output["reason_latent_posterior_live"].float(),
                    selective_output["reason_propensity"].float(), hidden_mask,
                    propensity_visibility_slopes=selective_output["propensity_visibility_slopes"].float(),
                    propensity_uncertainty_slopes=selective_output["propensity_uncertainty_slopes"].float(),
                    propensity_pi_min=selective_output["propensity_pi_min"].float(),
                    propensity_pi_max=selective_output["propensity_pi_max"].float(),
                    reason_false_positive_rate=selective_output["reason_false_positive_rate"].float(),
                    reason_false_positive_max=selective_output["reason_false_positive_max"].float(),
                    rank_loss=reason_rank.float() * controls.posterior_rank_weight_scale,
                    prevalence_observed_targets=reasons,
                    **{
                        name: float(value)
                        for name, value in config["loss"]["reason"].items()
                    },
                )
                reason_total = reason_losses["loss_reason_total"]
            else:
                reason_rank = reason_logits.sum() * 0.0
                reason_rank_stats = {"pair_weight_sum": reason_rank.detach(), "queue_count": reason_rank.detach()}
                reason_total = _positive_anchor_reason_loss(reason_logits.float(), reasons)
                reason_losses = {
                    "loss_reason_total": reason_total,
                    "loss_observation_nll": reason_total,
                    "loss_posterior_bce": reason_total.detach() * 0,
                    "loss_posterior_rank": reason_total.detach() * 0,
                    "loss_missing_recovery": reason_total.detach() * 0,
                    "loss_latent_rate_range": reason_total.detach() * 0,
                    "loss_propensity_regularization": reason_total.detach() * 0,
                }
            action_loss = action_losses["loss_action_total"]
            explanation_loss = factor_losses["loss_factor_total"] + state_losses["loss_state_total"] + reason_total
            if not torch.isfinite(action_loss) or not torch.isfinite(explanation_loss):
                raise FloatingPointError(
                    f"non-finite MOSAIC loss at epoch={epoch} step={step}: "
                    f"action={float(action_loss.detach().cpu())} explanation={float(explanation_loss.detach().cpu())}"
                )

        should_step = (step + 1) % grad_accum == 0 or step + 1 == len(loader)
        if controls.action_anchor_enabled:
            action_anchor.accumulate(
                action_loss,
                explanation_loss,
                shared,
                action_only,
                explanation_only,
                loss_scale=1.0 / grad_accum,
            )
            anchor_stats = (
                action_anchor.finalize(step=global_update)
                if should_step
                else {"step": global_update, "available": False, "accumulating": True}
            )
        else:
            (action_loss / grad_accum).backward(retain_graph=True)
            (explanation_loss / grad_accum).backward()
            anchor_stats = {
                "step": global_update, "constraint_pass": True, "lambda_star": 0.0,
                "dot_action_aux": 0.0, "action_grad_norm": 0.0, "aux_grad_norm": 0.0,
                "halfspace_lhs": 0.0, "halfspace_rhs": 0.0, "shared_param_count": len(shared),
                "available": False,
            }
        if should_step:
            _set_representation_lrs(
                optimizer,
                global_update,
                total_updates,
                warmup_updates,
                controls.representation_lr_scale,
                float(config["optimizer"]["min_lr_ratio"]),
            )
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None],
                float(config["optimizer"]["grad_clip"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
        action_queue.enqueue(action_logits.detach(), actions, batch["file_name"])
        if controls.posterior_enabled:
            reason_queue.enqueue(
                reason_logits.detach(), selective_output["reason_latent_posterior"], batch["file_name"]
            )
        if profile_timing and device.type == "cuda":
            cuda_end.record()
            cuda_end.synchronize()
            device_step_time_sec = cuda_start.elapsed_time(cuda_end) / 1000.0
        else:
            device_step_time_sec = time.perf_counter() - step_started
        hidden_count = int(hidden_mask.sum().detach().cpu())
        hidden_recovery = (
            float(selective_output["reason_latent_posterior"][hidden_mask].mean().detach().cpu())
            if hidden_count else 0.0
        )
        loss_row = {
            "epoch": epoch, "step": step, "phase": controls.phase,
            **{name: value for name, value in action_losses.items()},
            **{name: value for name, value in factor_losses.items() if name.startswith("loss_")},
            **{name: value for name, value in state_losses.items() if name.startswith("loss_")},
            **{name: value for name, value in reason_losses.items() if name.startswith("loss_")},
            "action_only_loss": action_loss,
            "explanation_only_loss": explanation_loss,
            "effective_batch": images.shape[0] * grad_accum,
            "step_time_sec": time.perf_counter() - step_started,
            "device_step_time_sec": device_step_time_sec,
            "dataloader_load_time_sec": load_time_sec,
            "dataloader_stall": load_time_sec > dataloader_stall_threshold_sec,
        }
        rows["loss_components.jsonl"].append(loss_row)
        rows["action_rank_stats.jsonl"].append({"epoch": epoch, "step": step, **action_rank_stats})
        rows["reason_rank_stats.jsonl"].append({"epoch": epoch, "step": step, **reason_rank_stats})
        if anchor_stats.get("available", True):
            rows["action_anchor_stats.jsonl"].append({"epoch": epoch, **anchor_stats})
        rows["selective_observation_stats.jsonl"].append(
            {
                "epoch": epoch, "step": step,
                "posterior_available": controls.posterior_enabled,
                "propensity_mean": selective_output["reason_propensity"].mean(),
                "propensity_bound_rate": (
                    (selective_output["reason_propensity"] <= config["selective_observation"]["pi_min"] + 1e-3)
                    | (selective_output["reason_propensity"] >= config["selective_observation"]["pi_max"] - 1e-3)
                ).float().mean(),
                "posterior_mean": selective_output["reason_latent_posterior"].mean(),
                "posterior_all_on_rate": (selective_output["reason_latent_posterior"] > 0.95).float().mean(),
                "posterior_all_off_rate": (selective_output["reason_latent_posterior"] < 0.05).float().mean(),
            }
        )
        rows["posterior_recovery_stats.jsonl"].append(
            {"epoch": epoch, "step": step, "hidden_count": hidden_count, "hidden_posterior_mean": hidden_recovery}
        )
        recovery_domain = observed_for_likelihood <= 0.5
        recovery_scores.append(selective_output["reason_latent_posterior"][recovery_domain].detach().float().cpu())
        recovery_targets.append(hidden_mask[recovery_domain].detach().float().cpu())
        rows["factor_grounding_stats.jsonl"].append(
            {
                "epoch": epoch, "step": step,
                "presence_valid_count": factor_losses["count_presence"],
                "visibility_valid_count": factor_losses["count_visibility"],
                "geometry_valid_count": factor_losses["count_geometry_mask"],
                "source_coverage_rate": (observations["source_code"] > 0).float().mean(),
            }
        )
        for name, offset in row_offsets.items():
            if len(rows[name]) > offset:
                rows[name][offset:] = [
                    _detach_artifact_value(row) for row in rows[name][offset:]
                ]
        if step % int(config["training"].get("print_every", 200)) == 0:
            payload = {
                "event": "mosaic_batch", "epoch": epoch, "phase": controls.phase,
                "step": step, "total_steps": len(loader),
                "action_loss": float(action_loss.detach().cpu()),
                "reason_loss": float(reason_total.detach().cpu()),
                "factor_loss": float(factor_losses["loss_factor_total"].detach().cpu()),
                "posterior_mean": float(selective_output["reason_latent_posterior"].mean().detach().cpu()),
            }
            print(json.dumps(payload), flush=True)
    all_recovery_scores = torch.cat(recovery_scores) if recovery_scores else torch.zeros(0)
    all_recovery_targets = torch.cat(recovery_targets) if recovery_targets else torch.zeros(0)
    if all_recovery_targets.sum() > 0:
        posterior_auprc = binary_average_precision(all_recovery_scores, all_recovery_targets)
        zero_baseline_auprc = binary_average_precision(torch.zeros_like(all_recovery_scores), all_recovery_targets)
    else:
        posterior_auprc = 0.0
        zero_baseline_auprc = 0.0
    rows["posterior_recovery_stats.jsonl"].append(
        {
            "epoch": epoch,
            "summary": True,
            "available": bool(all_recovery_targets.sum() > 0),
            "posterior_recovery_auprc": posterior_auprc,
            "zero_as_negative_auprc": zero_baseline_auprc,
            "improvement": posterior_auprc - zero_baseline_auprc,
            "hidden_positive_count": int(all_recovery_targets.sum()),
            "recovery_domain_count": int(all_recovery_targets.numel()),
        }
    )
    return rows, global_update


def fit_calibrator(
    model,
    threshold,
    loader,
    optimizer,
    device,
    *,
    epoch: int,
    max_steps: int,
    calibration_config: dict[str, Any],
):
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("calibration max_steps must be a positive integer")
    model.eval()
    threshold.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_calib_action_logits = []
    train_calib_reason_logits = []
    train_calib_actions = []
    train_calib_reasons = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            output = model(images)
        train_calib_action_logits.append(output["action_logits_raw"].detach())
        train_calib_reason_logits.append(output["reason_logits_latent"].detach())
        train_calib_actions.append(batch["action"].to(device, non_blocking=True))
        train_calib_reasons.append(batch["reason"].to(device, non_blocking=True))
    if not train_calib_actions:
        raise RuntimeError("train_calib loader is empty")
    action_logits = torch.cat(train_calib_action_logits, dim=0)
    reason_logits = torch.cat(train_calib_reason_logits, dim=0)
    actions = torch.cat(train_calib_actions, dim=0)
    reasons = torch.cat(train_calib_reasons, dim=0)
    supported_labels = torch.cat((actions, reasons), dim=-1).sum(dim=0) > 0
    if not supported_labels.all():
        missing = torch.nonzero(~supported_labels, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"train_calib has no positive support for labels {missing}")

    batch_size = int(calibration_config["batch_size"])
    if batch_size != 256:
        raise ValueError("formal calibration batch_size must remain 256")
    rows = []
    sample_count = action_logits.shape[0]
    fixed_order = torch.arange(sample_count, device=device)
    for step in range(max_steps):
        start = step * batch_size
        batch_indices = fixed_order[(torch.arange(batch_size, device=device) + start) % sample_count]
        objective = threshold.calibration_objective(
            action_logits[batch_indices],
            reason_logits[batch_indices],
            actions[batch_indices],
            reasons[batch_indices],
            surrogate_temperature=float(calibration_config["surrogate_temperature"]),
            soft_f1_weight=float(calibration_config["soft_f1_weight"]),
            bce_weight=float(calibration_config["bce_weight"]),
            rate_weight=float(calibration_config["rate_weight"]),
            delta_weight=float(calibration_config["delta_weight"]),
            cardinality_weight=float(calibration_config["cardinality_weight"]),
        )
        loss = objective["loss_calibration_total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        theta = threshold.compose_theta().detach()
        rows.append(
            {
                "epoch": epoch, "step": step, "source": "train_calib", "loss": loss,
                "loss_calibration_soft_f1": objective["loss_calibration_soft_f1"],
                "loss_calibration_bce": objective["loss_calibration_bce"],
                "loss_calibration_rate": objective["loss_calibration_rate"],
                "loss_calibration_delta": objective["loss_calibration_delta"],
                "loss_calibration_cardinality": objective["loss_calibration_cardinality"],
                "soft_f1_valid_label_count": objective["soft_f1_valid_label_count"],
                "calibration_batch_size": batch_size,
                "threshold_logit": theta,
                "threshold_prob": torch.sigmoid(theta),
                "theta_group": threshold.theta_group.detach(),
                "theta_delta": threshold.label_delta.detach(),
            }
        )
    return rows


@torch.no_grad()
def evaluate_factor_modes(
    model,
    loader,
    grounding_builder,
    grounding_index,
    device,
    *,
    epoch: int,
    max_samples: int,
) -> dict[str, Any]:
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("factor-mode audit max_samples must be positive")
    model.eval()
    scores = {mode: [] for mode in ("full", "content_only", "prior_only")}
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        reasons = batch["reason"].to(device, non_blocking=True)
        remaining = max_samples - sample_count
        if remaining <= 0:
            break
        if images.shape[0] > remaining:
            images = images[:remaining]
            reasons = reasons[:remaining]
            file_names = batch["file_name"][:remaining]
        else:
            file_names = batch["file_name"]
        observations = grounding_builder(
            reasons,
            _grounding_records(grounding_index, file_names),
            split="train",
        )
        for mode in scores:
            output = model(images, prior_mode=mode, return_masks=True)
            scores[mode].append(_weak_factor_score(output, observations)["score"])
        sample_count += images.shape[0]
    means = {
        mode: sum(values) / len(values) if values else 0.0
        for mode, values in scores.items()
    }
    full = means["full"]
    return {
        "epoch": epoch,
        "split": "train_calib_audit",
        "available": sample_count > 0,
        "sample_count": sample_count,
        "full_factor_metric": full,
        "content_only_factor_metric": means["content_only"],
        "prior_only_factor_metric": means["prior_only"],
        "content_only_retention": means["content_only"] / max(full, 1e-9),
        "prior_to_full_ratio": means["prior_only"] / max(full, 1e-9),
    }


def _build_failure_cases(
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    action_targets: torch.Tensor,
    reason_targets: torch.Tensor,
    sample_ids: list[str],
    *,
    epoch: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if action_logits.shape != action_targets.shape or reason_logits.shape != reason_targets.shape:
        raise ValueError("failure-case logits and targets must have matching shapes")
    if action_logits.shape[0] != reason_logits.shape[0] or action_logits.shape[0] != len(sample_ids):
        raise ValueError("failure-case tensors and sample IDs must have matching batch sizes")
    action_predictions = action_logits >= 0
    reason_predictions = reason_logits >= 0
    action_truth = action_targets > 0.5
    reason_truth = reason_targets > 0.5
    action_errors = action_predictions != action_truth
    reason_errors = reason_predictions != reason_truth
    total_errors = action_errors.sum(-1) + reason_errors.sum(-1)
    failing = torch.nonzero(total_errors > 0, as_tuple=False).flatten()
    if failing.numel() == 0:
        return [{"epoch": epoch, "split": "test", "available": False, "reason": "no_misclassified_samples"}]
    ordered = failing[torch.argsort(total_errors[failing], descending=True)][:limit]
    rows: list[dict[str, Any]] = []
    for tensor_index in ordered:
        index = int(tensor_index)
        rows.append(
            {
                "epoch": epoch,
                "split": "test",
                "available": True,
                "file_name": sample_ids[index],
                "action_error_count": int(action_errors[index].sum()),
                "reason_error_count": int(reason_errors[index].sum()),
                "action_false_positive": torch.nonzero(
                    action_predictions[index] & ~action_truth[index], as_tuple=False
                ).flatten().tolist(),
                "action_false_negative": torch.nonzero(
                    ~action_predictions[index] & action_truth[index], as_tuple=False
                ).flatten().tolist(),
                "reason_false_positive": torch.nonzero(
                    reason_predictions[index] & ~reason_truth[index], as_tuple=False
                ).flatten().tolist(),
                "reason_false_negative": torch.nonzero(
                    ~reason_predictions[index] & reason_truth[index], as_tuple=False
                ).flatten().tolist(),
            }
        )
    return rows


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _checkpoint_payload(model, selective, threshold, action_queue, reason_queue, representation_optimizer, calibration_optimizer, epoch, controls, config, split_stats, runtime_profile, best):
    return {
        "model": model.state_dict(), "selective_observation": selective.state_dict(),
        "calibrator": threshold.state_dict(), "action_queue": action_queue.state_dict(),
        "reason_queue": reason_queue.state_dict(),
        "optimizer": {
            "representation": representation_optimizer.state_dict(),
            "calibration": calibration_optimizer.state_dict(),
        },
        "epoch": epoch, "phase": controls.phase, "config": config, "git_head": _git_head(),
        "split_hash": split_stats["split_hash"], "runtime_profile_hash": runtime_profile["sha256"],
        "best_metrics": best,
    }


def _save_checkpoints(output_dir, payload, metrics, best):
    output_dir = Path(output_dir)
    candidates = {
        "joint": metrics["deploy_fixed"]["joint"],
        "action": metrics["deploy_fixed"]["Act_mF1"],
        "reason": metrics["deploy_fixed"]["Exp_mF1"],
        "reason_map": metrics["deploy_fixed"]["Exp_mAP"],
    }
    files = {
        "joint": "checkpoint_best_test_joint.pth",
        "action": "checkpoint_best_test_action.pth",
        "reason": "checkpoint_best_test_reason.pth",
        "reason_map": "checkpoint_best_test_reason_map.pth",
    }
    for name, score in candidates.items():
        if score > best.get(name, float("-inf")):
            best[name] = score
            best[f"{name}_epoch"] = payload["epoch"]
            torch.save(payload, output_dir / files[name])
    torch.save(payload, output_dir / "checkpoint_latest.pth")
    write_json(output_dir / "best_checkpoints.json", {"records": best})


def _epoch_stop_reasons(
    epoch: int,
    evaluation: dict[str, Any],
    history: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[str]:
    summary = evaluation["metrics_summary"]
    current = {
        "action_raw": summary["raw"]["Act_mF1"],
        "action_oracle": summary["test_oracle_diagnostic"]["Act_mF1"],
        "exp_fixed": summary["deploy_fixed"]["Exp_mF1"],
        "exp_map": summary["raw"]["Exp_mAP"],
        "action_ap": evaluation["action_branch_metrics"]["raw"]["Act_per_label_ap"],
        "action_visual": evaluation["action_branch_metrics"]["visual"]["Act_mF1"],
        **diagnostics,
    }
    history.append(current)
    if epoch < 8:
        return []
    reasons: list[str] = []
    if len(history) >= 2:
        prior = history[:-1]
        best_raw = max(row["action_raw"] for row in prior)
        best_oracle = max(row["action_oracle"] for row in prior)
        last_two = history[-2:]
        if all(best_raw - row["action_raw"] > 0.02 or best_oracle - row["action_oracle"] > 0.02 for row in last_two):
            reasons.append("raw_or_oracle_action_mf1_declined_gt_0p02_two_epochs")
        best_ap = [max(row["action_ap"][index] for row in prior) for index in range(4)]
        if all(all(row["action_ap"][index] < best_ap[index] for index in range(4)) for row in last_two):
            reasons.append("all_four_per_action_ap_declined_two_epochs")
        explanation_tradeoff = len(history) >= 3 and all(
            row["exp_fixed"] > history[-3]["exp_fixed"]
            and row["exp_map"] < history[-3]["exp_map"]
            for row in last_two
        )
        if explanation_tradeoff:
            reasons.append("explanation_fixed_f1_up_while_map_down_two_epochs")
        if all(row["action_raw"] + 0.005 < row["action_visual"] for row in last_two):
            reasons.append("state_branch_consistently_harms_action_visual")
        if all(row.get("propensity_bound_rate", 0.0) > 0.90 for row in last_two):
            reasons.append("propensity_at_bounds_gt_0p90_two_epochs")
        if all(
            row.get("posterior_all_on_rate", 0.0) > 0.90
            or row.get("posterior_all_off_rate", 0.0) > 0.90
            for row in last_two
        ):
            reasons.append("latent_reason_posterior_all_on_or_all_off_two_epochs")
        factor_audits = [row for row in history if row.get("factor_audit_available")]
        if len(factor_audits) >= 2 and all(
            row.get("prior_to_full_ratio", 0.0) >= 0.90 for row in factor_audits[-2:]
        ):
            reasons.append("factor_prior_only_ge_0p90_full_two_audits")
    if diagnostics.get("posterior_recovery_available") and diagnostics.get("posterior_recovery_improvement", 0.0) <= 0:
        reasons.append("posterior_recovery_not_better_than_zero_as_negative")
    return reasons


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        tf32_enabled = bool(config["training"]["tf32"])
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
    runtime_profile_path = Path(args.runtime_selection)
    if not runtime_profile_path.exists() and not args.allow_missing_runtime_profile:
        raise FileNotFoundError("MOSAIC runtime selection is required before training")
    runtime_profile = (
        json.loads(runtime_profile_path.read_text(encoding="utf-8"))
        if runtime_profile_path.exists()
        else {"selected": {"batch_size": args.batch_size, "grad_accum": args.grad_accum}, "smoke_override": True}
    )
    runtime_profile["sha256"] = _sha256(runtime_profile_path) if runtime_profile_path.exists() else "SMOKE"
    selected = runtime_profile["selected"]
    batch_size = int(args.batch_size or selected["batch_size"])
    grad_accum = int(args.grad_accum or selected["grad_accum"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, calib_loader, test_loader, split_stats = build_loaders(
        config, output_dir, batch_size=batch_size, num_workers=args.num_workers,
        max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples,
        max_calib_samples=args.max_calib_samples,
        seed=args.seed,
    )
    model, selective, threshold, action_queue, reason_queue = build_model_components(config, args.config, device)
    representation_optimizer, calibration_optimizer = build_optimizers(model, selective, threshold, config)
    grounding_builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    multiview = MOSAICWeakMultiView(factor_names, seed=args.seed)
    action_anchor = MOSAICActionAnchoredGradient(
        aux_shared_lambda_max=float(config["optimizer"]["aux_shared_lambda_max"]),
        action_anchor_kappa=float(config["optimizer"]["action_anchor_kappa"]),
    )
    manifest = {
        "method": "ACPR-MOSAIC-AD V1", "git_head": _git_head(), "command": " ".join(args.command_line),
        "direct_image": True, "feature_cache": False, "token_compression": "none",
        "config_sha256": _sha256(args.config),
        "pretrained_weights": config["backbone"]["pretrained_weights"],
        "pretrained_sha256": _sha256(config["backbone"]["pretrained_weights"]),
        "backbone_arch": config["backbone"]["arch"],
        "selected_layers": config["backbone"]["selected_layers"],
        "checkpoint_key": config["backbone"]["checkpoint_key"],
        "backbone_frozen": config["backbone"]["freeze_backbone"],
        "data_root": config["data"]["data_root"],
        "raw_root": config["data"]["raw_root"],
        "bdd100k_root": config["data"]["bdd100k_root"],
        "eval_splits": ["test"], "best_selection_split": "test",
        "batch_size": batch_size, "grad_accum": grad_accum, "num_workers": args.num_workers,
        "effective_batch": batch_size * grad_accum,
        "loss_weights": config["loss"],
        "optimizer": config["optimizer"],
        "calibration": config["calibration"],
        "split_hash": split_stats["split_hash"],
        "foreground_only": True, "runtime_profile_hash": runtime_profile["sha256"],
    }
    initialize_run_artifacts(
        output_dir, manifest=manifest, config=config,
        git_state={"head": manifest["git_head"], "branch": "acpr_mosaic_ad_v1_direct_image"},
        runtime_profile=runtime_profile, split_stats=split_stats,
    )
    total_repr_updates = max(1, math.ceil(len(train_loader) / grad_accum) * 13)
    global_update = 0
    best: dict[str, Any] = {}
    failure_history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        controls = mosaic_phase_controls(epoch)
        _apply_phase(model, selective, representation_optimizer, controls)
        if controls.calibration_only:
            rows = {
                name: [
                    {
                        "epoch": epoch,
                        "phase": controls.phase,
                        "available": False,
                        "reason": "calibration_only_phase",
                    }
                ]
                for name in EPOCH_JSONL_FILES
            }
        else:
            rows, global_update = train_representation_epoch(
                model=model, selective=selective, action_queue=action_queue, reason_queue=reason_queue,
                loader=train_loader, optimizer=representation_optimizer, action_anchor=action_anchor,
                grounding_builder=grounding_builder, grounding_index=grounding_index, multiview=multiview,
                controls=controls, config=config, device=device, epoch=epoch, grad_accum=grad_accum,
                global_update=global_update, total_updates=total_repr_updates,
            )
        threshold_rows = fit_calibrator(
            model,
            threshold,
            calib_loader,
            calibration_optimizer,
            device,
            epoch=epoch,
            max_steps=int(config["calibration"]["steps_per_epoch"]),
            calibration_config=config["calibration"],
        )
        rows["threshold_stats.jsonl"] = threshold_rows or [
            {
                "epoch": epoch,
                "source": "train_calib",
                "available": False,
                "reason": "no_train_calib_batches",
            }
        ]
        if epoch in config["training"]["factor_audit_epochs"]:
            factor_mode_audit = evaluate_factor_modes(
                model,
                calib_loader,
                grounding_builder,
                grounding_index,
                device,
                epoch=epoch,
                max_samples=int(config["training"]["factor_audit_samples"]),
            )
        else:
            factor_mode_audit = {
                "epoch": epoch,
                "available": False,
                "reason": "factor_mode_audit_not_scheduled",
            }
        rows["factor_mode_audit.jsonl"] = [factor_mode_audit]
        evaluation = evaluate_mosaic(model, threshold, test_loader, device, epoch=epoch)
        rows["observable_factor_stats.jsonl"] = evaluation["factor_rows"]
        rows["factor_stats_by_factor.jsonl"] = evaluation["factor_rows"]
        rows["prototype_usage_stats.jsonl"] = evaluation["prototype_rows"]
        rows["decision_state_stats.jsonl"] = evaluation["state_rows"]
        rows["failure_cases.jsonl"] = _build_failure_cases(
            evaluation["logits"]["action_deploy"],
            evaluation["logits"]["reason_deploy"],
            evaluation["logits"]["labels_action"],
            evaluation["logits"]["labels_reason"],
            evaluation["sample_ids"],
            epoch=epoch,
        )
        for name in EPOCH_JSONL_FILES:
            rows.setdefault(name, [{"epoch": epoch, "available": False, "reason": "phase_not_active"}])
        logit_map = {
            "action_visual.pt": evaluation["logits"]["action_visual"],
            "action_state.pt": evaluation["logits"]["action_state"],
            "action_raw.pt": evaluation["logits"]["action_raw"],
            "action_deploy.pt": evaluation["logits"]["action_deploy"],
            "reason_latent.pt": evaluation["logits"]["reason_latent"],
            "reason_deploy.pt": evaluation["logits"]["reason_deploy"],
            "labels_action.pt": evaluation["logits"]["labels_action"],
            "labels_reason.pt": evaluation["logits"]["labels_reason"],
        }
        json_payloads = {
            "metrics_summary.json": evaluation["metrics_summary"],
            "per_label_metrics.json": evaluation["per_label_metrics"],
            "action_branch_metrics.json": evaluation["action_branch_metrics"],
            "reason_branch_metrics.json": evaluation["reason_branch_metrics"],
        }
        write_epoch_artifacts(
            output_dir, epoch=epoch, json_payloads=json_payloads, jsonl_payloads=rows,
            logits=logit_map, sample_ids=evaluation["sample_ids"],
        )
        payload = _checkpoint_payload(
            model, selective, threshold, action_queue, reason_queue, representation_optimizer,
            calibration_optimizer,
            epoch, controls, config, split_stats, runtime_profile, best,
        )
        _save_checkpoints(output_dir, payload, evaluation["metrics_summary"], best)
        print(json.dumps({"event": "mosaic_epoch", **evaluation["metrics_summary"]}), flush=True)
        active_anchor_rows = [
            row for row in rows["action_anchor_stats.jsonl"] if row.get("available", True)
        ]
        epoch_anchor_pass_rate = (
            sum(bool(row.get("constraint_pass")) for row in active_anchor_rows) / len(active_anchor_rows)
            if active_anchor_rows
            else 1.0
        )
        diagnostics: dict[str, Any] = {}
        if controls.action_anchor_enabled and epoch_anchor_pass_rate < 0.95:
            reasons = ["action_anchor_constraint_pass_rate_below_0p95"]
        else:
            selective_rows = [
                row
                for row in rows["selective_observation_stats.jsonl"]
                if row.get("posterior_available")
            ]
            recovery_summary = next(
                (
                    row
                    for row in reversed(rows["posterior_recovery_stats.jsonl"])
                    if row.get("summary")
                ),
                {},
            )
            diagnostics = {
                "propensity_bound_rate": (
                    sum(float(row["propensity_bound_rate"]) for row in selective_rows)
                    / len(selective_rows)
                    if selective_rows
                    else 0.0
                ),
                "posterior_all_on_rate": (
                    sum(float(row["posterior_all_on_rate"]) for row in selective_rows)
                    / len(selective_rows)
                    if selective_rows
                    else 0.0
                ),
                "posterior_all_off_rate": (
                    sum(float(row["posterior_all_off_rate"]) for row in selective_rows)
                    / len(selective_rows)
                    if selective_rows
                    else 0.0
                ),
                "posterior_recovery_available": bool(recovery_summary.get("available", False)),
                "posterior_recovery_improvement": float(recovery_summary.get("improvement", 0.0)),
                "factor_audit_available": bool(factor_mode_audit.get("available", False)),
                "prior_to_full_ratio": float(factor_mode_audit.get("prior_to_full_ratio", 0.0)),
            }
            reasons = _epoch_stop_reasons(epoch, evaluation, failure_history, diagnostics)
        decision = {
            "epoch": epoch,
            "event": "continue" if not reasons else "hard_stop",
            "reasons": reasons,
            "epoch_action_anchor_pass_rate": epoch_anchor_pass_rate,
            "diagnostics": diagnostics,
        }
        append_jsonl(output_dir / "supervisor_decisions.jsonl", decision)
        if reasons:
            raise RuntimeError(f"MOSAIC scientific stop rule triggered: {reasons}")
    completion_name = (
        "GOAL_COMPLETED_ACPR_MOSAIC_AD_V1.json"
        if args.epochs == 15 and args.max_train_samples is None
        else "BOUNDED_RUN_COMPLETED_ACPR_MOSAIC_AD_V1.json"
    )
    write_json(
        output_dir / completion_name,
        {"completed": True, "formal_full_run": completion_name.startswith("GOAL_"), "epochs": args.epochs, "git_head": _git_head(), "best_metrics": best},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--runtime_selection", default=".review/mosaic_runtime_selection.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--grad_accum", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_calib_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--allow_missing_runtime_profile", action="store_true")
    args = parser.parse_args()
    if args.epochs != 15 and not (args.max_train_samples and args.epochs > 0):
        raise ValueError("only bounded smoke may override the 15-epoch formal schedule")
    args.command_line = list(__import__("sys").argv)
    return args


if __name__ == "__main__":
    run(parse_args())
