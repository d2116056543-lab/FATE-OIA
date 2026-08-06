from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.losses import acpr_losses as base_losses
from fate_oia.losses.aie_loss_registry import AIELossRegistry, exact_owner_parameter_groups
from fate_oia.losses.aie_losses import (
    action_cardinality_loss,
    contribution_effect_loss,
    counterfactual_necessity_loss,
    evidence_censored_reason_asl_loss,
    naming_alignment_loss,
    predicate_map_compactness_loss,
    predicate_masked_asl_loss,
    predicate_reason_alignment_pu_loss,
    predicate_map_loss,
    probe_duplicate_loss,
    reason_negative_weight,
    reason_ranking_loss,
    soft_f1_loss,
)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar
from fate_oia.models.aie_oia_model import AIEOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.aie_artifacts import append_jsonl, capture_rng_state, json_safe, restore_rng_state, write_json
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds
from fate_oia.utils.aie_contracts import gradient_norm
from fate_oia.utils.aie_counterfactual import AIECounterfactualConfig, AIECounterfactualEngine
from fate_oia.utils.aie_hashes import aie_source_tree_sha256, file_sha256, object_sha256, state_dict_sha256
from fate_oia.utils.aie_metrics import aie_branch_metrics, counterfactual_case_metrics, counterfactual_metrics, probe_health_metrics


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def current_git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def canonical_model_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove the legacy DINO projection alias after proving it is redundant."""
    clean: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if ".attn.vproj." not in key:
            clean[key] = value
            continue
        canonical = key.replace(".attn.vproj.", ".attn.proj.")
        counterpart = state.get(canonical)
        if counterpart is None or counterpart.shape != value.shape or not torch.equal(counterpart, value):
            raise RuntimeError(f"unmatched DINO vproj alias: {key}")
    return clean


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "action": torch.stack([row["action"] for row in batch]),
        "reason": torch.stack([row["reason"] for row in batch]),
        "file_name": [row["file_name"] for row in batch],
        "image_path": [row["image_path"] for row in batch],
    }


def make_dataset(cfg: dict[str, Any], split: str) -> BDDOIAMultiTaskDataset:
    data = cfg["data"]
    transform = AspectRatioLetterboxTransform(
        int(data["image_height"]), int(data["image_width"]), patch_size=int(data["patch_size"])
    )
    return BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split=split, action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    cfg: dict[str, Any],
) -> DataLoader:
    data = cfg["data"]
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": bool(data.get("pin_memory", True)),
        "persistent_workers": bool(data.get("persistent_workers", True)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def build_model(cfg: dict[str, Any], device: torch.device, use_mock_dino: bool = False) -> AIEOIAModel:
    primary, backbone, evidence, reason = cfg["primary"], cfg["backbone"], cfg["evidence"], cfg["reason_private"]
    model = AIEOIAModel(
        dim=int(primary["dim"]), action_dim=4, reason_dim=21,
        selected_layers=tuple(backbone["selected_layers"]), pretrained_weights=str(backbone["pretrained_weights"]),
        scene_config=str(primary["scene_predicates"]), grammar_path=str(primary["reason_grammar"]),
        use_mock_dino=use_mock_dino, mock_dim=int(primary["dim"]),
        probes_per_action=int(evidence["probes_per_action"]),
        local_points_per_layer=int(evidence["local_points_per_layer"]), max_offset=float(evidence["max_offset"]),
        predicate_bias_max=float(evidence["predicate_bias_max"]), probe_chunk_size=int(evidence["probe_chunk_size"]), action_kappa=float(evidence["action_kappa"]),
        action_logit_norm_cap=float(cfg["training"]["action_logit_norm_cap"]),
        reason_kappa=float(reason["reason_kappa"]),
    )
    return model.to(device)


def build_optimizer(model: AIEOIAModel, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    training = cfg["training"]
    owned = exact_owner_parameter_groups(model)
    lr_key = {
        "primary": "lr_primary", "action_evidence": "lr_action_evidence",
        "action_contribution": "lr_action_contribution", "reason_private": "lr_reason_private",
    }
    groups = [
        {"params": list(parameters), "lr": float(training[lr_key[owner]]), "base_lr": float(training[lr_key[owner]]), "name": owner}
        for owner, parameters in owned.items()
    ]
    return torch.optim.AdamW(groups, weight_decay=float(training["weight_decay"]))


def normalize_accumulated_gradients(parameters, micro_batches: int) -> None:
    if micro_batches <= 0:
        raise ValueError("micro_batches must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(micro_batches)


def schedule_values(update: int, total_updates: int, cfg: dict[str, Any]) -> dict[str, float]:
    progress = update / max(total_updates, 1)
    warmup = float(cfg["training"]["warmup_ratio"])
    min_ratio = float(cfg["training"]["min_lr_ratio"])
    if progress < warmup:
        lr_multiplier = max(progress / max(warmup, 1e-8), 1e-3)
    else:
        cosine_progress = (progress - warmup) / max(1 - warmup, 1e-8)
        lr_multiplier = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * cosine_progress))
    action_start = float(cfg["evidence"]["action_scale_start"])
    reason_start = float(cfg["reason_private"]["reason_scale_start"])
    action_scale = action_start + (1 - action_start) * min(1.0, progress / float(cfg["evidence"]["action_scale_ramp_ratio"]))
    reason_scale = reason_start + (1 - reason_start) * min(1.0, progress / float(cfg["reason_private"]["reason_scale_ramp_ratio"]))
    cf_start, cf_full = float(cfg["counterfactual"]["start_ratio"]), float(cfg["counterfactual"]["full_ratio"])
    cf_scale = 0.0 if progress <= cf_start else min(1.0, (progress - cf_start) / max(cf_full - cf_start, 1e-8))
    grounding_scale = 0.25 + 0.75 * min(1.0, progress / max(warmup, 1e-8))
    return {"lr": lr_multiplier, "action": action_scale, "reason": reason_scale, "cf": cf_scale, "grounding": grounding_scale}


def compute_counter_confidence(output: dict[str, Any], structured: dict[str, Any], reason_target: torch.Tensor, counter_cfg: dict[str, Any]) -> torch.Tensor:
    contradiction = output["contradiction_score"].detach().float().clamp(0, 1)
    negative_mask = structured["predicate_source_complete"].new_tensor(output["_negative_mask"], dtype=torch.float32)
    counts = negative_mask.sum(-1).clamp_min(1.0)
    source_complete = structured["predicate_source_complete"].detach().float() @ negative_mask.t() / counts
    predicate_observability = output["predicate_attention"].detach().float().max(-1).values
    region_observability = predicate_observability @ negative_mask.t() / counts
    confidence = (
        float(counter_cfg["contradictory_predicate_weight"]) * contradiction
        + float(counter_cfg["source_completeness_weight"]) * source_complete
        + float(counter_cfg["region_observability_weight"]) * region_observability
    ).clamp(0, 1)
    return torch.where(reason_target > 0.5, torch.ones_like(confidence), confidence)


def _zero(output: dict[str, Any]) -> torch.Tensor:
    return output["action_logits_final_train"].sum() * 0


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _quantiles(value: torch.Tensor) -> tuple[float, float, float]:
    result = torch.quantile(value.float().detach(), value.new_tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    return tuple(float(item.cpu()) for item in result)


def _per_label_metrics(metrics: dict[str, Any], prefix: str) -> list[dict[str, float | int]]:
    f1 = metrics[f"{prefix}_per_label_f1"]
    precision = metrics[f"{prefix}_per_label_precision"]
    recall = metrics[f"{prefix}_per_label_recall"]
    average_precision = metrics[f"{prefix}_per_label_ap"]
    return [
        {"label_id": index, "f1": f1[index], "precision": precision[index], "recall": recall[index], "ap": average_precision[index]}
        for index in range(len(f1))
    ]


def enqueue_reason_rank_memory(
    memory: dict[str, torch.Tensor],
    logits: torch.Tensor,
    target: torch.Tensor,
    negative_weight: torch.Tensor,
    capacity: int,
) -> None:
    """Keep a tiny detached FIFO so rare labels see cross-sample rank pairs."""
    for key, value in (
        ("logits", logits),
        ("target", target),
        ("negative_weight", negative_weight),
    ):
        detached = value.detach()
        previous = memory.get(key)
        memory[key] = detached if previous is None else torch.cat((previous, detached), dim=0)[-capacity:]


def compute_losses(
    output: dict[str, Any], batch: dict[str, Any], structured: dict[str, Any], cfg: dict[str, Any],
    grammar: ACPRReasonGrammar, cf: dict[str, Any] | None, grounding_scale: float, cf_scale: float,
    reason_rank_memory: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | str]], dict[str, float], torch.Tensor]:
    registry = AIELossRegistry(cfg["loss_weights"])
    action, reason = batch["action"], batch["reason"]
    counter_confidence = compute_counter_confidence(output, structured, reason, cfg["counter_evidence"])
    reason_weights = reason_negative_weight(reason, counter_confidence, float(cfg["counter_evidence"]["zero_negative_floor"]))
    registry.add("primary_action", "primary", asymmetric_loss_with_logits(output["action_logits_primary"], action))
    registry.add("primary_action_visual", "primary", asymmetric_loss_with_logits(output["action_visual_logits_primary"], action))
    registry.add("primary_action_reason", "primary", asymmetric_loss_with_logits(output["action_reason_logits_primary"], action))
    registry.add("primary_reason_partial", "primary", base_losses.partial_label_reason_loss(
        output["reason_logits_primary"], reason, output["contradiction_score"]
    ))
    registry.add("primary_reason_soft_f1", "primary", soft_f1_loss(output["reason_logits_primary"], reason, reason_weights))
    registry.add("predicate_cls", "primary", float(grounding_scale) * predicate_masked_asl_loss(
        output["predicate_logits"], structured["predicate_target"], structured["predicate_target_mask"],
        structured["predicate_counter_mask"], structured["predicate_reliability"]
    ))
    registry.add("predicate_map", "primary", float(grounding_scale) * predicate_map_loss(
        output["predicate_attention"], structured["predicate_map_target"], structured["predicate_map_mask"]
    ))
    pos_mask = output.get("_grammar_positive_mask")
    neg_mask = output.get("_grammar_contradictory_mask")
    if pos_mask is None:
        pos_mask = output["predicate_probs"].new_tensor(output["_positive_mask"])
        neg_mask = output["predicate_probs"].new_tensor(output["_negative_mask"])
    registry.add("predicate_reason_align", "primary", float(grounding_scale) * predicate_reason_alignment_pu_loss(
        output["predicate_probs"], reason, pos_mask, neg_mask, reason_weights
    ))
    registry.add("predicate_compactness", "primary", float(grounding_scale) * predicate_map_compactness_loss(output["predicate_attention"]))
    registry.add("final_action", "action_contribution", asymmetric_loss_with_logits(output["action_logits_final_train"], action))
    registry.add("final_action_soft_f1", "action_contribution", soft_f1_loss(output["action_logits_final_train"], action))
    registry.add("final_action_cardinality", "action_contribution", action_cardinality_loss(output["action_logits_final_train"], action))
    registry.add("final_reason", "reason_private", evidence_censored_reason_asl_loss(output["reason_logits_final_train"], reason, counter_confidence))
    registry.add(
        "final_reason_rank",
        "reason_private",
        reason_ranking_loss(
            output["reason_logits_final_train"],
            reason,
            negative_weight=reason_weights,
            reference_logits=None if not reason_rank_memory else reason_rank_memory.get("logits"),
            reference_target=None if not reason_rank_memory else reason_rank_memory.get("target"),
            reference_negative_weight=None if not reason_rank_memory else reason_rank_memory.get("negative_weight"),
        ),
    )
    registry.add("final_reason_soft_f1", "reason_private", soft_f1_loss(output["reason_logits_final_train"], reason, reason_weights))
    if cf and cf["cf_valid_count"] > 0:
        valid = cf["valid_mask"]
        registry.add("cf_necessity", "action_evidence", float(cf_scale) * counterfactual_necessity_loss(cf["selected_drop"], cf["control_drop"], valid))
        registry.add("cf_effect", "action_contribution", float(cf_scale) * contribution_effect_loss(cf["supportive_contribution"], cf["selected_minus_control"], valid))
        registry.add("cf_sufficiency", "action_evidence", float(cf_scale) * (cf["sufficiency_loss_raw"] * valid).sum() / valid.sum().clamp_min(1))
        event_valid = torch.zeros_like(output["bounded_contribution"], dtype=torch.bool)
        event_supportive = torch.zeros_like(event_valid)
        event_gap = torch.zeros_like(output["bounded_contribution"])
        sample_by_name = {name: index for index, name in enumerate(batch["file_name"])}
        for index, case in enumerate(cf["cases"]):
            sample = sample_by_name[case["file_name"]]
            action_id, probe_id = case["action_id"], case["probe_id"]
            event_valid[sample, action_id, probe_id] = cf["valid_mask"][index].detach() > 0
            event_supportive[sample, action_id, probe_id] = cf["supportive_contribution"][index].detach() > 0
            event_gap[sample, action_id, probe_id] = cf["selected_minus_control"][index].detach()
        reliable_positive = (structured["predicate_map_mask"] > 0) & (structured["predicate_target"] > 0)
        registry.add(
            "naming",
            "action_evidence",
            float(cf_scale) * naming_alignment_loss(
                output["evidence_map"],
                output["name_quality"],
                structured["predicate_map_target"],
                reliable_positive,
                event_supportive,
                event_gap,
                event_valid,
            ),
        )
    else:
        for name, owner in (("cf_necessity", "action_evidence"), ("cf_effect", "action_contribution"), ("cf_sufficiency", "action_evidence"), ("naming", "action_evidence")):
            registry.add(name, owner, _zero(output))
    registry.add("probe_duplicate", "action_evidence", probe_duplicate_loss(output["evidence_map"], output["bounded_contribution"], action))
    registry.add("delta", "action_contribution", output["action_delta"].square().mean())
    return registry.total(), registry.rows(), {"counter_negative_weight_mean": float(reason_weights.mean().detach().cpu())}, counter_confidence


def attach_grammar_masks(output: dict[str, Any], model: AIEOIAModel) -> None:
    output["_positive_mask"] = model.foundation.predicate_reason.positive_mask.detach().cpu().tolist()
    output["_negative_mask"] = model.foundation.predicate_reason.contradictory_mask.detach().cpu().tolist()


def _append_cpu(store: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor, limit: int) -> None:
    store.setdefault(key, []).append(value[:limit].detach().to("cpu"))


def _slice_batch_output(output: dict[str, Any], count: int) -> dict[str, Any]:
    batch = output["action_logits_primary"].shape[0]
    return {
        key: (value[:count] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch else value)
        for key, value in output.items()
    }


@torch.no_grad()
def collect_logits(
    model: AIEOIAModel,
    loader: DataLoader,
    device: torch.device,
    action_scale: float,
    reason_scale: float,
    audit_limit: int = 0,
    cf_engine: AIECounterfactualEngine | None = None,
):
    model.eval()
    store = {key: [] for key in ("action_primary", "action_final", "reason_primary", "reason_final", "action_target", "reason_target")}
    names: list[str] = []
    audit_tensors: dict[str, list[torch.Tensor]] = {}
    audit_cases: list[dict[str, Any]] = []
    audit_cf_tensors: dict[str, list[torch.Tensor]] = {}
    audit_seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            field = model.encode_images(images)
            output = model.decode_from_field(field, action_scale=action_scale, reason_scale=reason_scale)
        store["action_primary"].append(output["action_logits_primary"].float().cpu())
        store["action_final"].append(output["action_logits_final"].float().cpu())
        store["reason_primary"].append(output["reason_logits_primary"].float().cpu())
        store["reason_final"].append(output["reason_logits_final"].float().cpu())
        store["action_target"].append(batch["action"].float())
        store["reason_target"].append(batch["reason"].float())
        names.extend(batch["file_name"])
        if audit_seen < audit_limit:
            take = min(images.shape[0], audit_limit - audit_seen)
            _append_cpu(audit_tensors, "evidence_map", output["evidence_map"], take)
            _append_cpu(audit_tensors, "contribution", output["bounded_contribution"], take)
            for key in (
                "name_id", "name_confidence", "name_margin", "name_quality", "name_spatial_soft_iou",
                "name_compatibility", "predicate_probs", "reason_action_evidence_attention", "reason_action_prior",
                "reason_predicate_prior", "reason_private_attention", "reason_delta",
            ):
                _append_cpu(audit_tensors, key, output[key], take)
            for variant, kwargs in {
                "primary_only": None,
                "final": {},
                "predicate_bias_off": {"predicate_bias_enabled": False}, "local_reread_off": {"local_reread_enabled": False},
                "global_only": {"local_reread_enabled": False, "group_attention_enabled": False},
                "action_evidence_shuffle": {"action_evidence_shuffle": True},
                "wrong_action_evidence": {"wrong_action_evidence": True},
                "reason_action_prior_off": {"reason_action_prior_enabled": False}, "reason_predicate_prior_off": {"reason_predicate_prior_enabled": False},
                "all_reason_priors_off": {"reason_action_prior_enabled": False, "reason_predicate_prior_enabled": False},
            }.items():
                if kwargs is None:
                    action_logits, reason_logits = output["action_logits_primary"], output["reason_logits_primary"]
                elif not kwargs:
                    action_logits, reason_logits = output["action_logits_final"], output["reason_logits_final"]
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        branch = model.decode_from_field(field, action_scale=action_scale, reason_scale=reason_scale, **kwargs)
                    action_logits, reason_logits = branch["action_logits_final"], branch["reason_logits_final"]
                _append_cpu(audit_tensors, f"{variant}_action_logits", action_logits, take)
                _append_cpu(audit_tensors, f"{variant}_reason_logits", reason_logits, take)
            if cf_engine is not None:
                audit_output = _slice_batch_output(output, take)
                audit_cf = cf_engine.run(
                    model,
                    audit_output,
                    batch["action"][:take].to(device),
                    batch["file_name"][:take],
                    global_update=audit_seen,
                    action_scale=action_scale,
                )
                audit_cases.extend(audit_cf["cases"])
                for key in (
                    "selected_masks", "control_masks", "union_keep_masks", "wrong_probe_masks", "wrong_action_masks",
                    "selected_logits", "control_logits", "union_keep_logits", "wrong_probe_logits", "wrong_action_logits",
                    "selected_drop", "control_drop", "selected_minus_control", "wrong_probe_drop", "wrong_action_drop",
                ):
                    value = audit_cf[key]
                    if value.numel() > 0:
                        audit_cf_tensors.setdefault(key, []).append(value.detach().to("cpu"))
            audit_seen += take
    audit: dict[str, Any] = {key: torch.cat(value) for key, value in audit_tensors.items()}
    audit["counterfactual"] = {
        key: torch.cat(value) for key, value in audit_cf_tensors.items()
    }
    audit["counterfactual"]["cases"] = audit_cases
    return {key: torch.cat(value) for key, value in store.items()}, names, audit


def evaluate_epoch(model, calib_loader, test_loader, device, epoch_dir: Path, action_scale: float, reason_scale: float, cfg):
    # DINO lazily exposes ``vproj = proj`` during its first attention
    # forward. Canonicalization removes that verified alias while retaining
    # every independently owned parameter for the mutation check.
    state_before_dict = canonical_model_state_dict(model.state_dict())
    state_before = state_dict_sha256(state_before_dict)
    state_key_hashes_before = {
        key: state_dict_sha256({key: value}) for key, value in state_before_dict.items()
    }
    calib, _, _ = collect_logits(model, calib_loader, device, action_scale, reason_scale)
    groups = [list(range(4)), list(range(4, 25))]
    thresholds = fit_posthoc_thresholds(
        torch.cat((calib["action_final"], calib["reason_final"]), 1),
        torch.cat((calib["action_target"], calib["reason_target"]), 1), groups,
        shrinkage_support=float(cfg["calibration"]["group_shrinkage_support"]), grid_step=float(cfg["calibration"]["grid_step"]),
    )
    cf_cfg = AIECounterfactualConfig(**{k: cfg["counterfactual"][k] for k in AIECounterfactualConfig.__dataclass_fields__})
    test, names, audit = collect_logits(model, test_loader, device, action_scale, reason_scale, int(cfg["runtime"]["fixed_test_audit_samples"]), AIECounterfactualEngine(cf_cfg))
    state_after_dict = canonical_model_state_dict(model.state_dict())
    if state_dict_sha256(state_after_dict) != state_before:
        changed_keys = [
            key
            for key, value in state_after_dict.items()
            if state_key_hashes_before.get(key) != state_dict_sha256({key: value})
        ]
        raise RuntimeError(f"Post-hoc calibration mutated model state: {changed_keys}")
    primary = aie_branch_metrics(test["action_primary"], test["reason_primary"], test["action_target"], test["reason_target"])
    final = aie_branch_metrics(test["action_final"], test["reason_final"], test["action_target"], test["reason_target"])
    deploy_all = apply_posthoc_threshold(torch.cat((test["action_final"], test["reason_final"]), 1), thresholds["threshold_prob"])
    deploy = aie_branch_metrics(deploy_all[:, :4], deploy_all[:, 4:], test["action_target"], test["reason_target"])
    for name, tensor in {
        "action_logits_primary_test.pt": test["action_primary"], "action_logits_final_test.pt": test["action_final"],
        "reason_logits_primary_test.pt": test["reason_primary"], "reason_logits_final_test.pt": test["reason_final"],
        "labels_action_test.pt": test["action_target"], "labels_reason_test.pt": test["reason_target"],
    }.items(): torch.save(tensor, epoch_dir / name)
    write_json(epoch_dir / "file_names_test.json", names)
    torch.save({key: value for key, value in audit.items() if key.endswith("_logits")}, epoch_dir / "audit_128_ablation_logits.pt")
    torch.save(audit, epoch_dir / "audit_128_full_tensors.pt")
    audit_cf = audit.get("counterfactual", {})
    write_json(epoch_dir / "audit_128_counterfactual_cases.json", audit_cf.get("cases", []))
    write_json(epoch_dir / "calibration_thresholds.json", thresholds)
    return {
        "primary": primary,
        "final": final,
        "deploy": deploy,
        "deploy_joint": deploy["joint"],
        "calibration_thresholds": thresholds,
        "model_state_hash_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--epochs", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int); parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-audit-samples", type=int)
    parser.add_argument("--max-calib-samples", type=int); parser.add_argument("--max-test-samples", type=int); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume"); parser.add_argument("--use-mock-dino", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config); cfg["counter_evidence"] = load_config("configs/aie_reason_counter_evidence.yaml"); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); torch.set_float32_matmul_precision("high")
    training, data_cfg = cfg["training"], cfg["data"]
    seed = int(data_cfg["split_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    epochs = args.epochs or int(training["epochs"]); batch_size = args.batch_size or int(training["batch_size"])
    accumulation = args.gradient_accumulation_steps or int(training["gradient_accumulation_steps"])
    workers = args.num_workers if args.num_workers is not None else int(data_cfg["num_workers"])
    train_ds, test_ds = make_dataset(cfg, "train"), make_dataset(cfg, "test")
    train_ids = [sample.file_name for sample in train_ds.samples]
    splits = stable_split_ids(train_ids, int(data_cfg["split_seed"]), float(data_cfg["train_calib_fraction"]), int(data_cfg["train_audit_count"]))
    id_to_index = {sample.file_name: i for i, sample in enumerate(train_ds.samples)}
    train_count = min(args.max_train_samples or len(train_ds), len(train_ds))
    train_indices = list(range(len(train_ds)))
    random.Random(seed).shuffle(train_indices)
    train_subset = Subset(train_ds, train_indices[:train_count])
    calib_indices = [id_to_index[x] for x in splits["train_calib"][: args.max_calib_samples or None]]
    audit_indices = [id_to_index[x] for x in splits["train_audit"][: args.max_audit_samples or None]]
    test_subset = Subset(test_ds, list(range(min(args.max_test_samples or len(test_ds), len(test_ds)))))
    train_loader = make_loader(train_subset, batch_size, True, workers, cfg)
    calib_loader = make_loader(Subset(train_ds, calib_indices), batch_size, False, workers, cfg)
    audit_loader = make_loader(Subset(train_ds, audit_indices), batch_size, False, workers, cfg)
    test_loader = make_loader(test_subset, batch_size, False, workers, cfg)
    write_split_manifest(output_dir / "split_manifest.json", train_ids, int(data_cfg["split_seed"]), float(data_cfg["train_calib_fraction"]), int(data_cfg["train_audit_count"]))
    write_json(output_dir / "train_calib_ids.json", splits["train_calib"]); write_json(output_dir / "train_audit_ids.json", splits["train_audit"])
    model = build_model(cfg, device, args.use_mock_dino); optimizer = build_optimizer(model, cfg)
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    owner_groups = exact_owner_parameter_groups(model)
    owner_map = {
        owner: sorted(parameter_names[id(parameter)] for parameter in parameters)
        for owner, parameters in owner_groups.items()
    }
    write_json(output_dir / "owner_map.json", owner_map)
    source_contract = {
        "source_branch": "acpr_calalign_v1_2",
        "source_head": "373aa49feac17372574fd7fb056c1d79c7c848fe",
        "target_branch": "acpr_aie_oia_v1_direct_image",
        "foundation_modules": ["dino", "ego", "predicate_head", "trunk", "predicate_reason"],
        "excluded_formal_modules": ["pair_memory", "action_combo", "calibration_head", "threshold_head"],
    }
    write_json(output_dir / "source_contract.json", source_contract)
    start_epoch = 0; optimizer_update = 0; best: dict[str, float] = {}
    reason_rank_memory: dict[str, torch.Tensor] = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        expected_hashes = {
            "config_hash": file_sha256(args.config),
            "source_head": "373aa49feac17372574fd7fb056c1d79c7c848fe",
            "predicate_schema_hash": file_sha256(cfg["primary"]["scene_predicates"]),
            "counter_evidence_schema_hash": file_sha256("configs/aie_reason_counter_evidence.yaml"),
            "source_tree_hash": aie_source_tree_sha256(),
        }
        mismatches = {key: (checkpoint.get(key), value) for key, value in expected_hashes.items() if checkpoint.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Resume contract mismatch: {mismatches}")
        model.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1; optimizer_update = int(checkpoint["optimizer_update"]); best = checkpoint.get("best", {})
        reason_rank_memory = {
            key: value.to(device)
            for key, value in checkpoint.get("reason_rank_memory", {}).items()
        }
        restore_rng_state(checkpoint["rng_state"])
    structured_builder = AIEStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"], data_cfg["bdd100k_root"])
    grammar = ACPRReasonGrammar(cfg["primary"]["reason_grammar"])
    cf_cfg = AIECounterfactualConfig(**{k: cfg["counterfactual"][k] for k in AIECounterfactualConfig.__dataclass_fields__})
    cf_engine = AIECounterfactualEngine(cf_cfg)
    total_updates = math.ceil(len(train_loader) / accumulation) * epochs
    manifest = {
        "run_kind": args.run_kind,
        "command_line": os.sys.argv,
        "git_head": current_git_head(),
        "source_tree_hash": aie_source_tree_sha256(),
        "config_hash": file_sha256(args.config),
        "predicate_schema_hash": file_sha256(cfg["primary"]["scene_predicates"]),
        "counter_evidence_schema_hash": file_sha256("configs/aie_reason_counter_evidence.yaml"),
        "split_manifest_hash": file_sha256(output_dir / "split_manifest.json"),
        "source_head": "373aa49feac17372574fd7fb056c1d79c7c848fe",
        **cfg["experiment"],
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "num_workers": workers,
        "split_seed": seed,
        "train_sample_count": len(train_subset),
        "train_audit_count": len(audit_indices),
        "train_calib_count": len(calib_indices),
        "test_sample_count": len(test_subset),
    }
    write_json(output_dir / "run_manifest.json", manifest); Path(output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for epoch in range(start_epoch, epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); window = 0
        epoch_start = time.perf_counter(); iteration_end = epoch_start
        epoch_cf_cases: list[dict[str, Any]] = []; epoch_cf_invalid = 0
        epoch_coverage: dict[str, float] = defaultdict(float)
        epoch_named_sum = 0.0; epoch_named_count = 0; epoch_name_quality_hits = 0.0; epoch_name_quality_count = 0
        for micro_step, batch in enumerate(train_loader, 1):
            batch_ready = time.perf_counter(); data_time = batch_ready - iteration_end
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            structured = structured_builder.build(batch["file_name"], device=device)
            for key, value in structured["coverage"].items():
                if isinstance(value, (int, float)): epoch_coverage[key] += float(value)
            schedule = schedule_values(optimizer_update, total_updates, cfg)
            for group in optimizer.param_groups: group["lr"] = group["base_lr"] * schedule["lr"]
            use_cf = schedule["cf"] > 0 and (optimizer_update + 1) % int(cfg["counterfactual"]["interval_optimizer_updates"]) == 0 and (micro_step % accumulation == 0 or micro_step == len(train_loader))
            prospective_update = optimizer_update + 1
            profile_step = (window + 1 == accumulation or micro_step == len(train_loader)) and (
                prospective_update % int(cfg["runtime"]["print_every_optimizer_updates"]) == 0 or prospective_update == 1
            )
            if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
            dino_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                field = model.encode_images(batch["image"])
                if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
                dino_time = time.perf_counter() - dino_start
                output = model.decode_from_field(
                    field, action_scale=schedule["action"], reason_scale=schedule["reason"], profile=profile_step
                )
                attach_grammar_masks(output, model)
                if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
                cf_start = time.perf_counter()
                cf = cf_engine.run(model, output, batch["action"], batch["file_name"], global_update=optimizer_update, action_scale=schedule["action"]) if use_cf else None
                if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
                counterfactual_time = time.perf_counter() - cf_start if use_cf else 0.0
                total, rows, diagnostics, counter_confidence = compute_losses(
                    output, batch, structured, cfg, grammar, cf, schedule["grounding"], schedule["cf"],
                    reason_rank_memory,
                )
            enqueue_reason_rank_memory(
                reason_rank_memory,
                output["reason_logits_final_train"],
                batch["reason"],
                reason_negative_weight(
                    batch["reason"],
                    counter_confidence,
                    float(cfg["counter_evidence"]["zero_negative_floor"]),
                ),
                int(training.get("reason_rank_memory_size", 512)),
            )
            reliable_name_batch = (structured["predicate_map_mask"] > 0) & (structured["predicate_target"] > 0)
            quality_batch = output["name_quality"].float()
            reliable_quality_batch = quality_batch.masked_fill(~reliable_name_batch[:, None, None, :], float("-inf")).max(-1).values
            random_quality_batch = quality_batch.mean(-1)
            valid_name_batch = reliable_name_batch.any(-1)[:, None, None].expand_as(reliable_quality_batch)
            epoch_name_quality_hits += float(((reliable_quality_batch > random_quality_batch) & valid_name_batch).float().sum().detach().cpu())
            epoch_name_quality_count += int(valid_name_batch.sum().detach().cpu())
            epoch_named_sum += float((output["name_id"] >= 0).float().sum().detach().cpu())
            epoch_named_count += output["name_id"].numel()
            if cf:
                epoch_cf_cases.extend(cf["cases"])
                epoch_cf_invalid += sum(cf["cf_invalid_reason_counts"].values())
            if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
            backward_start = time.perf_counter()
            total.backward(); window += 1
            if profile_step and device.type == "cuda": torch.cuda.synchronize(device)
            backward_time = time.perf_counter() - backward_start
            should_step = window == accumulation or micro_step == len(train_loader)
            if should_step:
                normalize_accumulated_gradients(model.parameters(), window)
                owners = exact_owner_parameter_groups(model)
                grad_before = {name: gradient_norm(params) for name, params in owners.items()}
                predicate_grad_before = gradient_norm(model.foundation.predicate_head.parameters())
                dino_grad_before = gradient_norm(model.foundation.dino.parameters())
                torch.nn.utils.clip_grad_norm_(owners["primary"], float(training["primary_grad_cap"]))
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["global_grad_clip"]))
                grad_after = {name: gradient_norm(params) for name, params in owners.items()}
                optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_update += 1; window = 0
                if optimizer_update % int(cfg["runtime"]["print_every_optimizer_updates"]) == 0 or optimizer_update == 1:
                    health = probe_health_metrics(output["evidence_map"], output["bounded_contribution"])
                    cf_metrics = counterfactual_metrics(cf)
                    logged_counter_confidence = compute_counter_confidence(output, structured, batch["reason"], cfg["counter_evidence"])
                    p10, p50, p90 = _quantiles(logged_counter_confidence)
                    compatibility = output["predicate_compatibility"].float().clamp_min(1e-9)
                    compatibility_entropy = -(compatibility * compatibility.log()).sum(-1).mean()
                    bounded = output["bounded_contribution"].float()
                    payload = {"event": "aie_batch", "epoch": epoch, "micro_step": micro_step, "optimizer_update": optimizer_update,
                        "loss_total": float(total.detach().cpu()), "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
                        "action_scale": schedule["action"], "reason_scale": schedule["reason"], "grounding_scale": schedule["grounding"], "cf_scale": schedule["cf"],
                        **{f"loss_{row['name']}": row["raw"] for row in rows}, **diagnostics, **health,
                        "raw_contribution_mean": float(output["raw_contribution"].mean().detach().cpu()), "raw_contribution_std": float(output["raw_contribution"].std().detach().cpu()),
                        "bounded_contribution_mean": float(bounded.mean().detach().cpu()), "bounded_contribution_std": float(bounded.std().detach().cpu()),
                        "positive_contribution_rate": float((bounded > 0).float().mean().detach().cpu()), "negative_contribution_rate": float((bounded < 0).float().mean().detach().cpu()),
                        "action_delta_rms": float(output["action_delta"].float().square().mean().sqrt().detach().cpu()), "reason_delta_rms": float(output["reason_delta"].float().square().mean().sqrt().detach().cpu()),
                        "primary_action_logit_rms": _rms(output["action_logits_primary"]), "final_action_logit_rms": _rms(output["action_logits_final"]),
                        "primary_reason_logit_rms": _rms(output["reason_logits_primary"]), "final_reason_logit_rms": _rms(output["reason_logits_final"]),
                        "reason_visual_score_rms": float(output["reason_visual_score_rms"].detach().cpu()),
                        "reason_action_prior_bias_rms": float(output["reason_action_prior_bias_rms"].detach().cpu()),
                        "reason_predicate_prior_bias_rms": float(output["reason_predicate_prior_bias_rms"].detach().cpu()),
                        "predicate_bias_strength_mean": float(output["predicate_bias_strength"].mean().detach().cpu()),
                        "predicate_compatibility_entropy": float(compatibility_entropy.detach().cpu()),
                        "named_coverage": float(output["named_coverage"].detach().cpu()),
                        "unnamed_coverage": 1.0 - float(output["named_coverage"].detach().cpu()),
                        "name_confidence_mean": float(output["name_confidence"].mean().detach().cpu()), "name_margin_mean": float(output["name_margin"].mean().detach().cpu()),
                        "cf_valid_count": cf_metrics["cf_valid_count"], "cf_invalid_count": cf_metrics["cf_invalid_count"],
                        "cf_selected_drop": cf_metrics["selected_drop_mean"], "cf_control_drop": cf_metrics["control_drop_mean"],
                        "cf_selected_minus_control": cf_metrics["selected_minus_control_mean"], "cf_contribution_effect_correlation": cf_metrics["contribution_effect_spearman"],
                        "counter_negative_weight_p10": p10, "counter_negative_weight_p50": p50, "counter_negative_weight_p90": p90,
                        "reliable_negative_rate": float((logged_counter_confidence >= 0.75).float().mean().detach().cpu()),
                        "weak_negative_rate": float((logged_counter_confidence < 0.75).float().mean().detach().cpu()),
                        "primary_grad_raw": grad_before["primary"], "primary_grad_capped": grad_after["primary"],
                        "action_evidence_grad": grad_before["action_evidence"], "action_contribution_grad": grad_before["action_contribution"],
                        "reason_private_grad": grad_before["reason_private"], "predicate_grad": predicate_grad_before,
                        "owner_gradients": grad_before, "dino_grad": dino_grad_before,
                        "data_time": data_time, "dino_time": dino_time,
                        "primary_time": output.get("_profile_primary_time"), "evidence_global_time": output.get("_profile_evidence_global_time"),
                        "evidence_local_time": output.get("_profile_evidence_local_time"), "reason_reread_time": output.get("_profile_reason_reread_time"),
                        "counterfactual_time": counterfactual_time, "backward_time": backward_time,
                        "step_time": dino_time + output.get("_profile_primary_time", 0.0) + output.get("_profile_evidence_global_time", 0.0) + output.get("_profile_evidence_local_time", 0.0) + output.get("_profile_reason_reread_time", 0.0) + counterfactual_time + backward_time,
                        "allocated_gb": torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0, "reserved_gb": torch.cuda.memory_reserved()/2**30 if torch.cuda.is_available() else 0,
                        "max_reserved_gb": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0,
                        "dino_calls_ordinary_batch": 1, "dino_calls_cf_event": 0}
                    payload["loss_name"] = payload.pop("loss_naming")
                    print(json.dumps(json_safe(payload)), flush=True)
                    append_jsonl(output_dir / "loss_components.jsonl", payload); append_jsonl(output_dir / "owner_gradients.jsonl", {"epoch": epoch, "optimizer_update": optimizer_update, **grad_before})
                    append_jsonl(output_dir / "runtime_components.jsonl", {k: payload[k] for k in ("epoch", "optimizer_update", "data_time", "dino_time", "primary_time", "evidence_global_time", "evidence_local_time", "reason_reread_time", "counterfactual_time", "backward_time", "step_time", "allocated_gb", "reserved_gb", "max_reserved_gb", "dino_calls_ordinary_batch", "dino_calls_cf_event")})
                    append_jsonl(output_dir / "evidence_components.jsonl", {"epoch": epoch, "optimizer_update": optimizer_update, **health, "named_coverage": payload["named_coverage"], "cf_valid_count": payload["cf_valid_count"]})
            iteration_end = time.perf_counter()
        epoch_dir = output_dir / f"epoch_{epoch:03d}"; epoch_dir.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_epoch(model, calib_loader, test_loader, device, epoch_dir, schedule["action"], schedule["reason"], cfg)
        train_audit_logits, _, _ = collect_logits(model, audit_loader, device, schedule["action"], schedule["reason"])
        train_audit_metrics = aie_branch_metrics(
            train_audit_logits["action_final"], train_audit_logits["reason_final"],
            train_audit_logits["action_target"], train_audit_logits["reason_target"],
        )
        write_json(epoch_dir / "train_audit_metrics.json", train_audit_metrics)
        naming_summary = {
            "named_coverage": epoch_named_sum / max(epoch_named_count, 1),
            "quality_gt_random_rate": epoch_name_quality_hits / max(epoch_name_quality_count, 1),
            "reliable_grounded_atom_count": epoch_name_quality_count,
        }
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics}); append_jsonl(epoch_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
        append_jsonl(epoch_dir / "branch_metrics.jsonl", metrics); write_json(epoch_dir / "per_label_action_metrics.json", _per_label_metrics(metrics["final"], "Act")); write_json(epoch_dir / "per_label_reason_metrics.json", _per_label_metrics(metrics["final"], "Exp"))
        append_jsonl(epoch_dir / "calibration_diagnostics.jsonl", {"model_state_hash_unchanged": metrics["model_state_hash_unchanged"]})
        append_jsonl(epoch_dir / "predicate_metrics.jsonl", {"positive_count": epoch_coverage.get("positive", 0.0)}); append_jsonl(epoch_dir / "predicate_grounding_metrics.jsonl", dict(epoch_coverage))
        append_jsonl(epoch_dir / "naming_metrics.jsonl", naming_summary); append_jsonl(epoch_dir / "probe_metrics.jsonl", probe_health_metrics(output["evidence_map"], output["bounded_contribution"]))
        append_jsonl(epoch_dir / "counterfactual_metrics.jsonl", counterfactual_case_metrics(epoch_cf_cases, epoch_cf_invalid)); write_json(epoch_dir / "counterfactual_cases.json", epoch_cf_cases); append_jsonl(epoch_dir / "owner_gradient_metrics.jsonl", grad_before)
        append_jsonl(epoch_dir / "runtime_epoch_metrics.jsonl", {"epoch_seconds": time.perf_counter()-epoch_start, "max_reserved_gb": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0})
        write_json(epoch_dir / "metrics_summary.json", {"epoch": epoch, **metrics}); write_json(epoch_dir / "branch_metrics.json", metrics)
        write_json(epoch_dir / "calibration.json", metrics["calibration_thresholds"])
        write_json(epoch_dir / "predicate_metrics.json", {"positive_count": epoch_coverage.get("positive", 0.0), "grounding": dict(epoch_coverage)})
        write_json(epoch_dir / "naming_metrics.json", naming_summary)
        write_json(epoch_dir / "probe_metrics.json", probe_health_metrics(output["evidence_map"], output["bounded_contribution"]))
        write_json(epoch_dir / "counterfactual_metrics.json", counterfactual_case_metrics(epoch_cf_cases, epoch_cf_invalid))
        write_json(epoch_dir / "owner_metrics.json", grad_before)
        write_json(epoch_dir / "runtime_metrics.json", {"epoch_seconds": time.perf_counter()-epoch_start, "max_reserved_gb": torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0})
        criteria = {"deploy_joint": metrics["deploy"]["joint"], "action_mF1": metrics["deploy"]["Act_mF1"], "reason_mF1": metrics["deploy"]["Exp_mF1"], "action_mAP": metrics["final"]["Act_mAP"], "reason_mAP": metrics["final"]["Exp_mAP"]}
        improved = []
        for key, value in criteria.items():
            if value >= best.get(key, float("-inf")):
                best[key] = value; improved.append(key)
        checkpoint = {
            "model": canonical_model_state_dict(model.state_dict()), "optimizer": optimizer.state_dict(), "scheduler_update": optimizer_update,
            "epoch": epoch, "micro_step": len(train_loader), "optimizer_update": optimizer_update, "best": dict(best),
            "rng_state": capture_rng_state(), "metrics": metrics, "calibration": metrics["calibration_thresholds"],
            "reason_rank_memory": {key: value.detach().cpu() for key, value in reason_rank_memory.items()},
            "manifest": manifest, "manifest_hash": object_sha256(manifest), "split_manifest_hash": manifest["split_manifest_hash"],
            "config_hash": manifest["config_hash"], "source_head": manifest["source_head"],
            "predicate_schema_hash": manifest["predicate_schema_hash"], "counter_evidence_schema_hash": manifest["counter_evidence_schema_hash"],
            "source_tree_hash": manifest["source_tree_hash"],
            "train_audit_ids": splits["train_audit"],
        }
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth"); torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
        for key in improved:
            torch.save(checkpoint, output_dir / f"checkpoint_best_test_{key}.pth")
        print(json.dumps(json_safe({"event": "aie_epoch", "epoch": epoch, **metrics})), flush=True)
    if args.run_kind == "full":
        write_json(output_dir / "GOAL_COMPLETED_AIE_OIA_V1.json", {"complete": True, "epochs": epochs, "best": best})


if __name__ == "__main__":
    main()
