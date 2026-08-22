from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.tida_contracts import _best_label_threshold
from fate_oia.utils.tida_artifacts import atomic_write_json
from fate_oia.utils.tida_temporal_metrics import paired_temporal_contribution, robust_motion_score


def gt_margin_advantage(
    real_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    sign = 2.0 * target.to(real_logits.dtype) - 1.0
    return sign * (real_logits - counterfactual_logits)


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def select_predicate_intervention_indices(
    route: torch.Tensor,
    contribution: torch.Tensor,
    *,
    count: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route.shape != contribution.shape or route.ndim != 3:
        raise ValueError("route and contribution must both be [B,A,F]")
    if int(count) * 2 > route.shape[-1]:
        raise ValueError("factor bank is too small for disjoint selected/control sets")
    route_score = route.mean((0, 1))
    contribution_score = contribution.abs().mean((0, 1))
    selected = contribution_score.topk(int(count)).indices
    available = torch.ones_like(route_score, dtype=torch.bool)
    available[selected] = False
    controls = []
    for selected_index in selected:
        distance = (route_score - route_score[selected_index]).abs().masked_fill(~available, float("inf"))
        control = distance.argmin()
        controls.append(control)
        available[control] = False
    return selected, torch.stack(controls)


@torch.no_grad()
def collect_tida_outputs(
    model, loader, device: torch.device, *, temporal_scale: float = 1.0,
    collect_mechanism: bool = False, mechanism_samples: int = 128,
    collect_audit_tensors: bool = False,
) -> dict[str, Any]:
    store = {key: [] for key in ("image_action", "video_action", "image_reason", "video_reason", "action_target", "reason_target")}
    diagnostics = {key: [] for key in (
        "rho", "action_delta", "reason_delta", "null_mass", "route_entropy",
        "action_evidence_confidence", "action_effective_trust",
        "reason_evidence_confidence", "reason_effective_trust",
        "action_flow_route_mass", "reason_flow_route_mass", "transition_reliability",
        "action_temporal_budget", "reason_temporal_budget",
        "action_temporal_need", "reason_temporal_need",
        "action_temporal_target_motion", "reason_temporal_target_motion",
        "reason_pu_weight",
        "velocity_norm", "acceleration_norm",
    )}
    audit_keys = (
        "terminal_prediction_history", "terminal_prediction_no_history", "terminal_target_evidence",
        "terminal_error_history", "terminal_error_no_history", "innovation_token",
        "predicate_differential_state", "predicate_velocity_norm", "predicate_acceleration_norm",
        "predicate_persistence", "predicate_region_mass", "predicate_region_mass_velocity", "common_motion_norm",
        "transition_tokens", "transition_tokens_by_scale", "motion_salience", "transition_consistency",
        "velocity", "acceleration", "region_velocity", "transition_reliability",
        "action_flow_route_mass", "reason_flow_route_mass",
        "action_temporal_route", "action_factor_contribution", "reason_temporal_route", "frame_valid_mask", "timestamps",
    )
    audit_store: dict[str, list[torch.Tensor]] = {key: [] for key in audit_keys}
    dynamic_concepts: list[dict[str, Any]] = []
    file_names: list[str] = []
    mechanism_rows: dict[str, list[float]] = {
        name: [] for name in (
            "history_off", "repeated_last", "time_shuffle", "time_reverse",
            "selected_predicate_flatten", "matched_predicate_flatten", "wrong_action_route",
            "static_only", "dynamic_only",
        )
    }
    mechanism_outputs: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {key: [] for key in ("action", "reason", "velocity")} for name in mechanism_rows
    }
    mechanism_base = {key: [] for key in ("action", "reason", "action_target", "reason_target", "velocity")}
    mechanism_count = 0
    model.eval()
    for batch in loader:
        batch = _device_batch(batch, device)
        output = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=temporal_scale, temporal_reason_scale=temporal_scale,
        )
        values = {
            "image_action": output["image_action_logits"], "video_action": output["video_action_logits"],
            "image_reason": output["image_reason_logits"], "video_reason": output["video_reason_logits"],
            "action_target": batch["action"], "reason_target": batch["reason"],
        }
        for key, value in values.items():
            store[key].append(value.detach().float().cpu())
        image_branch = output.get("image_branch", {})
        contradiction = image_branch.get("contradiction_score") if isinstance(image_branch, dict) else None
        reason_negative_weight = (
            torch.full_like(batch["reason"], 0.2)
            if contradiction is None
            else 0.2 + 0.8 * contradiction.detach().clamp(0.0, 1.0)
        )
        reason_pu = torch.where(batch["reason"] > 0.5, torch.ones_like(reason_negative_weight), reason_negative_weight)
        for key, value in {
            "rho": output["innovation_reliability"], "action_delta": output["action_temporal_delta"],
            "reason_delta": output["reason_temporal_delta"], "null_mass": output["action_null_mass"],
            "route_entropy": output["action_route_entropy"],
            "action_evidence_confidence": output["action_evidence_confidence"],
            "action_effective_trust": output["action_effective_trust"],
            "reason_evidence_confidence": output["reason_evidence_confidence"],
            "reason_effective_trust": output["reason_effective_trust"],
            "action_flow_route_mass": output["action_flow_route_mass"],
            "reason_flow_route_mass": output["reason_flow_route_mass"],
            "transition_reliability": output["transition_reliability"],
            "action_temporal_budget": output["action_temporal_budget"],
            "reason_temporal_budget": output["reason_temporal_budget"],
            "action_temporal_need": output["action_temporal_need"],
            "reason_temporal_need": output["reason_temporal_need"],
            "action_temporal_target_motion": output["action_temporal_target_motion"],
            "reason_temporal_target_motion": output["reason_temporal_target_motion"],
            "reason_pu_weight": reason_pu,
            "velocity_norm": output["velocity"].norm(dim=-1),
            "acceleration_norm": output["acceleration"].norm(dim=-1),
        }.items():
            diagnostics[key].append(value.detach().float().cpu())
        if collect_audit_tensors:
            for key in audit_keys:
                value = output[key].detach().cpu()
                audit_store[key].append(value if key == "frame_valid_mask" else value.float())
            dynamic_concepts.extend(output["dynamic_concepts"])
        file_names.extend(batch["file_name"])
        if collect_mechanism and mechanism_count < mechanism_samples:
            selected, matched = select_predicate_intervention_indices(
                output["action_route"][..., :32],
                output["action_factor_contribution"][..., :32],
            )
            mechanism_base["action"].append(output["video_action_logits"].detach().float().cpu())
            mechanism_base["reason"].append(output["video_reason_logits"].detach().float().cpu())
            mechanism_base["action_target"].append(batch["action"].detach().float().cpu())
            mechanism_base["reason_target"].append(batch["reason"].detach().float().cpu())
            mechanism_base["velocity"].append(output["velocity"].detach().float().cpu())
            for name in mechanism_rows:
                if "predicate_flatten" in name:
                    indices = selected if name.startswith("selected") else matched
                    output["intervention_predicate_indices"] = tuple(int(value) for value in indices)
                changed = model.rerun_temporal_from_output(
                    output, name, temporal_action_scale=temporal_scale, temporal_reason_scale=temporal_scale
                )
                mechanism_rows[name].append(float(
                    (changed["terminal_error_history"].mean() - output["terminal_error_history"].mean()).cpu()
                ))
                mechanism_outputs[name]["action"].append(changed["video_action_logits"].detach().float().cpu())
                mechanism_outputs[name]["reason"].append(changed["video_reason_logits"].detach().float().cpu())
                mechanism_outputs[name]["velocity"].append(changed["velocity"].detach().float().cpu())
            mechanism_count += batch["target_image"].shape[0]
    result = {key: torch.cat(value) for key, value in store.items()} | {
        key: torch.cat(value) for key, value in diagnostics.items()
    } | {"file_names": file_names}
    if collect_audit_tensors:
        result.update({key: torch.cat(value) for key, value in audit_store.items()})
        result["dynamic_concepts"] = dynamic_concepts
    if collect_mechanism:
        base_rows = {
            "image_action": torch.cat(mechanism_base["action"]), "video_action": torch.cat(mechanism_base["action"]),
            "image_reason": torch.cat(mechanism_base["reason"]), "video_reason": torch.cat(mechanism_base["reason"]),
            "action_target": torch.cat(mechanism_base["action_target"]), "reason_target": torch.cat(mechanism_base["reason_target"]),
        }
        base_metric = branch_metrics(base_rows)["video"]
        intervention_metrics = {}
        for name, values in mechanism_outputs.items():
            changed_rows = dict(base_rows)
            changed_rows["video_action"] = torch.cat(values["action"])
            changed_rows["video_reason"] = torch.cat(values["reason"])
            metric = branch_metrics(changed_rows)["video"]
            changed_action = torch.cat(values["action"])
            changed_reason = torch.cat(values["reason"])
            action_advantage = gt_margin_advantage(
                base_rows["video_action"], changed_action, base_rows["action_target"]
            )
            reason_advantage = gt_margin_advantage(
                base_rows["video_reason"], changed_reason, base_rows["reason_target"]
            )
            changed_velocity = torch.cat(values["velocity"])
            real_velocity = torch.cat(mechanism_base["velocity"])
            velocity_cosine = torch.nn.functional.cosine_similarity(
                real_velocity.flatten(1), changed_velocity.flatten(1), dim=-1
            )
            intervention_metrics[name] = {
                **metric,
                "joint_drop_from_real": base_metric["joint"] - metric["joint"],
                "action_mf1_drop_from_real": base_metric["Act_mF1"] - metric["Act_mF1"],
                "reason_mf1_drop_from_real": base_metric["Exp_mF1"] - metric["Exp_mF1"],
                "action_gt_margin_advantage_mean": float(action_advantage.mean()),
                "reason_gt_margin_advantage_mean": float(reason_advantage.mean()),
                "action_gt_margin_advantage_by_label": action_advantage.mean(0).tolist(),
                "reason_gt_margin_advantage_by_label": reason_advantage.mean(0).tolist(),
                "velocity_cosine_with_reference": float(velocity_cosine.mean()),
            }
        result["_mechanism"] = {
            "available": mechanism_count > 0,
            "sample_count": min(mechanism_count, mechanism_samples),
            "target_dino_reruns": 0,
            "mean_error_increase": {
                name: sum(values) / max(len(values), 1) for name, values in mechanism_rows.items()
            },
            "real_history_metrics": base_metric,
            "intervention_metrics": intervention_metrics,
        }
    return result


def branch_metrics(rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5) -> dict[str, Any]:
    return {
        "image": aie_branch_metrics(rows["image_action"], rows["image_reason"], rows["action_target"], rows["reason_target"], threshold=thresholds),
        "video": aie_branch_metrics(rows["video_action"], rows["video_reason"], rows["action_target"], rows["reason_target"], threshold=thresholds),
    }


def dynamic_slice_metrics(rows: dict[str, Any], thresholds: torch.Tensor | float = 0.5) -> dict[str, Any]:
    score = rows["rho"].mean(-1)
    masks = {"low_dynamic": score <= 0.10, "high_dynamic": score >= 0.25}
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            result[name] = {"available": False, "count": 0}
            continue
        sliced = {key: value[mask] for key, value in rows.items() if torch.is_tensor(value) and value.shape[0] == mask.shape[0]}
        metrics = branch_metrics(sliced, thresholds)
        result[name] = {
            "available": True, "count": int(mask.sum()), "rho_mean": float(score[mask].mean()),
            "image": metrics["image"], "video": metrics["video"],
            "action_mf1_delta": metrics["video"]["Act_mF1"] - metrics["image"]["Act_mF1"],
            "reason_mf1_delta": metrics["video"]["Exp_mF1"] - metrics["image"]["Exp_mF1"],
        }
    return result


def temporal_contribution_metrics(rows: dict[str, Any]) -> dict[str, Any]:
    motion = robust_motion_score(rows["velocity_norm"], rows["acceleration_norm"])
    return {
        "action": paired_temporal_contribution(
            rows["image_action"], rows["video_action"], rows["action_target"], motion_score=motion,
        ),
        "reason": paired_temporal_contribution(
            rows["image_reason"], rows["video_reason"], rows["reason_target"], motion_score=motion,
            pu_negative_weight=rows.get("reason_pu_weight"),
        ),
    }


def fit_train_calib_thresholds(rows: dict[str, Any]) -> dict[str, torch.Tensor]:
    video_logits = torch.cat([rows["video_action"], rows["video_reason"]], dim=-1)
    image_logits = torch.cat([rows["image_action"], rows["image_reason"]], dim=-1)
    targets = torch.cat([rows["action_target"], rows["reason_target"]], dim=-1)
    return {
        "video": _best_label_threshold(video_logits, targets),
        "image": _best_label_threshold(image_logits, targets),
    }


@torch.no_grad()
def collect_intervention_audit(model, loader, device: torch.device, max_samples: int = 128) -> dict[str, Any]:
    interventions = (
        "history_off", "repeated_last", "time_shuffle", "time_reverse",
        "wrong_action_route", "static_only", "dynamic_only",
    )
    rows: dict[str, list[float]] = {name: [] for name in interventions}
    count = 0
    target_encode_calls = 0
    for batch in loader:
        batch = _device_batch(batch, device)
        output = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
        )
        target_encode_calls += 1
        base = output["terminal_error_history"].mean()
        for name in interventions:
            changed = model.rerun_temporal_from_output(
                output, name, temporal_action_scale=1.0, temporal_reason_scale=1.0
            )
            rows[name].append(float((changed["terminal_error_history"].mean() - base).cpu()))
        count += batch["target_image"].shape[0]
        if count >= max_samples:
            break
    return {
        "available": count > 0,
        "sample_count": min(count, max_samples),
        "target_encode_calls": target_encode_calls,
        "mean_error_increase": {name: sum(values) / max(len(values), 1) for name, values in rows.items()},
    }


def save_epoch_outputs(output_dir: Path, epoch: int, rows: dict[str, Any], metrics: dict[str, Any], thresholds: dict[str, torch.Tensor], mechanism: dict[str, Any]) -> None:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(epoch_dir / "metrics_summary.json", metrics)
    atomic_write_json(epoch_dir / "calibration.json", {key: value.tolist() for key, value in thresholds.items()})
    atomic_write_json(epoch_dir / "temporal_mechanism_audit.json", mechanism)
    contribution = metrics.get("online", {}).get("temporal_contribution")
    if contribution is not None:
        atomic_write_json(epoch_dir / "temporal_contribution_metrics.json", contribution)
    atomic_write_json(epoch_dir / "file_names_test.json", rows["file_names"])
    if "dynamic_concepts" in rows:
        with (epoch_dir / "dynamic_concepts_test.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for file_name, concepts in zip(rows["file_names"], rows["dynamic_concepts"]):
                handle.write(json.dumps({"file_name": file_name, "dynamic_concepts": concepts}, ensure_ascii=False) + "\n")
    tensor_keys = (
        "image_action", "video_action", "image_reason", "video_reason", "action_target", "reason_target",
        "rho", "action_delta", "reason_delta", "null_mass", "route_entropy",
        "action_evidence_confidence", "action_effective_trust",
        "reason_evidence_confidence", "reason_effective_trust",
        "action_flow_route_mass", "reason_flow_route_mass", "transition_reliability",
        "action_temporal_budget", "reason_temporal_budget",
        "action_temporal_need", "reason_temporal_need",
        "action_temporal_target_motion", "reason_temporal_target_motion",
        "reason_pu_weight",
        "velocity_norm", "acceleration_norm",
        "terminal_prediction_history", "terminal_prediction_no_history", "terminal_target_evidence",
        "terminal_error_history", "terminal_error_no_history", "innovation_token",
        "predicate_differential_state", "predicate_velocity_norm", "predicate_acceleration_norm",
        "predicate_persistence", "predicate_region_mass", "predicate_region_mass_velocity", "common_motion_norm",
        "transition_tokens", "transition_tokens_by_scale", "motion_salience", "transition_consistency",
        "velocity", "acceleration", "region_velocity",
        "action_temporal_route", "action_factor_contribution", "reason_temporal_route", "frame_valid_mask", "timestamps",
    )
    for key in tensor_keys:
        if key not in rows:
            continue
        torch.save(rows[key], epoch_dir / f"{key}_test.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", choices=("online", "ema"), default="online")
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from fate_oia.engine.train_tida_oia import build_runtime

    runtime = build_runtime(args, evaluation_only=True)
    rows = collect_tida_outputs(runtime.model, runtime.loaders["test"], runtime.device)
    calib = collect_tida_outputs(runtime.model, runtime.loaders["train_calib"], runtime.device)
    thresholds = fit_train_calib_thresholds(calib)
    metrics = {
        "raw_fixed": branch_metrics(rows),
        "deploy": {
            "image": branch_metrics(rows, thresholds["image"])["image"],
            "video": branch_metrics(rows, thresholds["video"])["video"],
        },
        "temporal_contribution": temporal_contribution_metrics(rows),
    }
    atomic_write_json(Path(args.output_dir) / "evaluation.json", metrics)
    print(json.dumps(metrics, default=str), flush=True)


if __name__ == "__main__":
    main()
