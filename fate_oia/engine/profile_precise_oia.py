from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.train_precise_oia import _observed_firewall, _slice_batch_output, build_optimizers, build_train_grounding_targets
from fate_oia.losses.precise_losses import evidence_view_consistency_loss, refinement_loss, total_precise_losses, two_way_consistency_loss
from fate_oia.losses.precise_intervention_losses import packed_target_specific_interventions
from fate_oia.models.precise_oia_model import PRECISEOIAModel
from fate_oia.transforms_precise import PRECISEImageTransform
from fate_oia.utils.precise_runtime import gpu_memory_gb
from fate_oia.utils.precise_gradient_ownership import loss_owner_gradient_matrix, parameter_ownership, projected_target_credit_grads


def choose_runtime_profile(profiles: list[dict[str, Any]], hard_limit_gb: float, target_limit_gb: float | None = None) -> dict[str, Any]:
    limit = hard_limit_gb if target_limit_gb is None else min(hard_limit_gb, target_limit_gb)
    safe = [item for item in profiles if item.get("valid") and float(item.get("peak_reserved_gb", float("inf"))) <= limit]
    if not safe:
        raise RuntimeError("No PRECISE runtime profile completed the complete path safely")
    fastest = max(float(item["samples_per_sec"]) for item in safe)
    near_fastest = [item for item in safe if float(item["samples_per_sec"]) >= fastest * 0.97]
    return min(near_fastest, key=lambda item: float(item["peak_reserved_gb"]))


def _make_dataset(config: dict[str, Any], max_samples: int):
    dataset = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "train", 4, 21, True, PRECISEImageTransform(return_mirror=False))
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def _make_loader(config: dict[str, Any], dataset, batch_size: int) -> DataLoader:
    workers = int(config["training"]["num_workers"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, prefetch_factor=int(config["training"]["prefetch_factor"]) if workers > 0 else None)


def _profile_one(config: dict[str, Any], args: argparse.Namespace, dataset, adapter, targets, active_fields, batch_size: int, accum: int, device: torch.device, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loader = _make_loader(config, dataset, batch_size)
    model = PRECISEOIAModel(Path(args.config).parent, config["pretrained_weights"], evidence_schema=active_fields, model_config=config).to(device)
    optimizers = build_optimizers(model, config)
    rows: list[dict[str, Any]] = []
    measured = 0
    total_images = 0
    start = None
    valid = True
    reason = ""
    firewall_report: dict[str, float] = {}
    owner_gradient_matrix: dict[str, dict[str, float]] = {}
    forward_shapes: dict[str, list[int]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    autocast_enabled = device.type == "cuda" and config["training"]["precision"] == "bf16"
    for step, batch in enumerate(loader):
        try:
            if step == args.warmup_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                total_images = 0
            images = batch["image"].to(device, non_blocking=True)
            mirror_count = min(images.shape[0], max(0, int(round(images.shape[0] * float(config["augmentation"]["mirror_pair_fraction"])))))
            model_input = torch.cat([images, images[:mirror_count].flip(-1)], dim=0) if mirror_count else images
            calls_before = model.dino.dino_call_count
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                full_output = model(model_input)
            calls_this_batch = model.dino.dino_call_count - calls_before
            output = _slice_batch_output(full_output, images.shape[0], model_input.shape[0])
            target = adapter.stack_batch([targets[name] for name in batch["file_name"]], device)
            action_target = batch["action"].to(device)
            reason_target = batch["reason"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                losses = total_precise_losses(output, action_target, reason_target, target)
                refine = 0.05 * (refinement_loss(output["action_logits_direct"], output["action_logits_final_raw"], action_target) + refinement_loss(output["reason_logits_direct"], output["reason_logits_semantic"], reason_target))
                intervention = packed_target_specific_interventions(model, output, action_target, reason_target, int(config["intervention"]["max_pairs_per_batch"]))
                mirror_loss = losses["loss_total"] * 0.0
                if mirror_count:
                    mirrored = _slice_batch_output(full_output, model_input.shape[0], model_input.shape[0], start=images.shape[0])
                    action_map = torch.tensor([0, 1, 3, 2], device=device)
                    reason_map = torch.tensor([int(row["mirror_partner"]) for row in model.reason_schema], device=device)
                    field_map = model.evidence_fields.mirror_field_indices
                    mirror_loss = 0.05 * two_way_consistency_loss(output["action_logits_final_raw"][:mirror_count], mirrored["action_logits_final_raw"], action_map) + 0.05 * two_way_consistency_loss(output["reason_logits_semantic"][:mirror_count], mirrored["reason_logits_semantic"], reason_map) + 0.0075 * evidence_view_consistency_loss(output, mirrored, field_map)
                full_loss = losses["loss_total"] + refine + intervention["loss_intervention"] + mirror_loss
            if not firewall_report:
                firewall_report = _observed_firewall(model, losses["loss_reason_observed"])
                threshold_probe = model.threshold_head(output["action_logits_final_raw"].detach(), output["reason_logits_observed"].detach())
                threshold_probe_loss = F.binary_cross_entropy_with_logits(threshold_probe["action_logits_deploy"], action_target.float()) + F.binary_cross_entropy_with_logits(threshold_probe["reason_logits_deploy"], reason_target.float())
                owner_gradient_matrix = loss_owner_gradient_matrix(model, {
                    "action": losses["loss_action_final"] + 0.5 * losses["loss_action_direct"],
                    "reason_semantic": losses["loss_reason_semantic"] + 0.5 * losses["loss_reason_direct"],
                    "reason_observed": losses["loss_reason_observed"],
                    "evidence": losses["loss_evidence"],
                    "intervention": intervention["loss_intervention"],
                    "threshold": threshold_probe_loss,
                })
                forward_shapes = {
                    "action_logits_final_raw": list(output["action_logits_final_raw"].shape),
                    "reason_logits_final_raw": list(output["reason_logits_final_raw"].shape),
                    "evidence_masks": list(output["evidence_masks"].shape),
                }
            evidence_parameters = parameter_ownership(model)["evidence_core"]
            _, raw_credit, projected_credit = projected_target_credit_grads(0.15 * losses["loss_evidence"], intervention["loss_intervention"], evidence_parameters, float(config["intervention"]["target_credit_grad_ratio"]))
            (full_loss / accum).backward()
            for parameter, raw_grad, projected_grad in zip(evidence_parameters, raw_credit, projected_credit):
                if parameter.grad is not None and raw_grad is not None and projected_grad is not None:
                    parameter.grad.add_((projected_grad - raw_grad) / accum)
            if (step + 1) % accum == 0:
                for optimizer in optimizers.values():
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step >= args.warmup_steps:
                measured += 1; total_images += int(batch["image"].shape[0])
            row = {"batch_size": batch_size, "grad_accum": accum, "step": step, "loss_total": float(full_loss.detach()), "loss_intervention": float(intervention["loss_intervention"].detach()), "mirror_pair_count": mirror_count, "dino_call_count": int(calls_this_batch), **gpu_memory_gb(device)}
            rows.append(row)
            if measured >= args.measure_steps:
                break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            valid = False; reason = "oom"; break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6) if start is not None else float("inf")
    peak = max((float(item.get("gpu_reserved_gb", 0.0)) for item in rows), default=0.0)
    core = bool(rows) and all(item["dino_call_count"] == 1 for item in rows)
    non_annotation = {key: value for key, value in firewall_report.items() if key != "observed_to_annotation_adapter_grad_norm"}
    evidence_rows = ("action", "reason_semantic", "reason_observed", "threshold")
    evidence_isolated = bool(owner_gradient_matrix) and all(owner_gradient_matrix[row]["evidence_core"] == 0.0 for row in evidence_rows)
    observed_isolated = bool(owner_gradient_matrix) and all(value == 0.0 for owner, value in owner_gradient_matrix["reason_observed"].items() if owner != "annotation_adapter")
    threshold_isolated = bool(owner_gradient_matrix) and all(value == 0.0 for owner, value in owner_gradient_matrix["threshold"].items() if owner != "threshold_head")
    expected_active = bool(owner_gradient_matrix) and all((
        owner_gradient_matrix["action"]["action_foundation"] > 0.0,
        owner_gradient_matrix["action"]["action_decoder"] > 0.0,
        owner_gradient_matrix["reason_semantic"]["reason_semantic"] > 0.0,
        owner_gradient_matrix["evidence"]["evidence_core"] > 0.0,
        owner_gradient_matrix["reason_observed"]["annotation_adapter"] > 0.0,
        owner_gradient_matrix["threshold"]["threshold_head"] > 0.0,
        owner_gradient_matrix["intervention"]["evidence_core"] > 0.0,
    ))
    profile = {"batch_size": batch_size, "grad_accum": accum, "workers": int(config["training"]["num_workers"]), "warmup_steps": args.warmup_steps, "measure_steps": measured, "samples_per_sec": total_images / elapsed if measured else 0.0, "peak_reserved_gb": peak, "dino_call_count": 1 if core else 0, "core_mechanisms_enabled": core, "valid": bool(valid and measured == args.measure_steps and core), "failure_reason": reason, "forward_shapes": forward_shapes, "gradient_firewall": firewall_report, "gradient_firewall_passed": bool(non_annotation) and all(float(value) == 0.0 for value in non_annotation.values()), "owner_gradient_matrix": owner_gradient_matrix, "owner_gradient_matrix_passed": evidence_isolated and observed_isolated and threshold_isolated and expected_active, "curve_distance_valid_count": float(losses.get("curve_distance_valid_count", torch.tensor(0.0)).detach().item()) if 'losses' in locals() else 0.0}
    del model, optimizers, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=320)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = _make_dataset(config, args.max_train_samples)
    adapter, targets, active_fields = build_train_grounding_targets(dataset, config, root / "targets_shared", enforce_coverage=False)
    profiles, steps = [], []
    for batch, accum in ((10, 3), (8, 4), (6, 5)):
        profile, rows = _profile_one(config, args, dataset, adapter, targets, active_fields, batch, accum, device, root)
        profiles.append(profile); steps.extend(rows)
    (root / "runtime_profile.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    (root / "runtime_steps.jsonl").write_text("".join(json.dumps(item) + "\n" for item in steps), encoding="utf-8")
    selected = choose_runtime_profile(profiles, float(config["runtime"]["hard_max_reserved_gb"]), float(config["runtime"]["target_peak_reserved_gb"]))
    selected["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    selected["config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    (root / "selected_runtime_profile.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    real_forward = {
        "passed": bool(selected["valid"] and selected["dino_call_count"] == 1 and selected["owner_gradient_matrix_passed"] and selected["curve_distance_valid_count"] > 0 and selected["forward_shapes"].get("action_logits_final_raw", [0, 0])[-1] == 4 and selected["forward_shapes"].get("reason_logits_final_raw", [0, 0])[-1] == 21),
        "gradient_firewall_passed": bool(selected["gradient_firewall_passed"]),
        "gradient_firewall": selected["gradient_firewall"],
        "owner_gradient_matrix": selected["owner_gradient_matrix"],
        "owner_gradient_matrix_passed": bool(selected["owner_gradient_matrix_passed"]),
        "curve_distance_valid_count": float(selected["curve_distance_valid_count"]),
        "forward_shapes": selected["forward_shapes"],
        "git_head": selected["git_head"],
        "config_sha256": selected["config_sha256"],
    }
    (root.parent / "real_forward.json").write_text(json.dumps(real_forward, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
