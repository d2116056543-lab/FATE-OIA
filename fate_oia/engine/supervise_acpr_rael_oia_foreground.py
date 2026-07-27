"""Real foreground RAEL launch construction; no detached-process path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml


def _weighted_value(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    """Average an actual scalar/list diagnostic by the decoded sample count."""

    if not rows:
        raise ValueError("P18 adapter received no evaluator diagnostic rows")
    weights = [int(row["sample_count"]) for row in rows]
    if any(weight <= 0 for weight in weights):
        raise ValueError("P18 evaluator diagnostic sample_count must be positive")

    def combine(values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                raise TypeError(f"P18 diagnostic {field} has inconsistent scalar values")
            return sum(float(weight) * float(value) for weight, value in zip(weights, values)) / float(sum(weights))
        if isinstance(first, list):
            if not all(isinstance(value, list) and len(value) == len(first) for value in values):
                raise ValueError(f"P18 diagnostic {field} has inconsistent list shape")
            return [combine([value[index] for value in values]) for index in range(len(first))]
        if isinstance(first, Mapping):
            if not all(isinstance(value, Mapping) and set(value) == set(first) for value in values):
                raise ValueError(f"P18 diagnostic {field} has inconsistent mapping keys")
            return {key: combine([value[key] for value in values]) for key in first}
        raise TypeError(f"P18 diagnostic {field} has unsupported type {type(first).__name__}")

    if any(field not in row for row in rows):
        raise ValueError(f"P18 evaluator omitted required diagnostic {field}")
    return combine([row[field] for row in rows])


def _require_epoch_pu_count(trainer: Any, label_id: int) -> float:
    """Read the real P17 epoch accumulation; missing state is never zero-filled."""

    value = getattr(trainer, "last_epoch_pu_soft_positive", None)
    if value is None or not hasattr(value, "shape") or tuple(value.shape) != (21,):
        raise ValueError("P18 adapter requires real epoch PU soft-positive totals [21]")
    scalar = float(value[label_id].item())
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError("P18 adapter received an invalid real PU soft-positive total")
    return scalar


def build_p18_epoch_artifacts(
    *,
    runtime: "RAELRuntime",
    trainer: Any,
    epoch: int,
    last_step_result: Any,
    step_count: int,
    evaluation: Mapping[str, Any],
    action_calibration: Mapping[str, Any],
    reason_calibration: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    pu_audit: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Map real P17/P19 values to P18; absent evidence is a hard failure.

    This adapter has no synthetic defaults.  It only serializes values emitted
    by the current test decode, P17 public step results, the train-calib fit,
    and a scheduled real counterfactual/case exporter.
    """

    rows = evaluation.get("diagnostic_rows")
    tensors = evaluation.get("tensors")
    cases = evaluation.get("case_exports")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not isinstance(tensors, Mapping):
        raise ValueError("P18 adapter requires evaluator diagnostic_rows and tensors")
    sample_count = sum(int(row.get("sample_count", 0)) for row in rows if isinstance(row, Mapping))
    if sample_count <= 0 or not isinstance(cases, Mapping):
        raise ValueError("P18 adapter requires real nonempty test diagnostics and case exports")
    required_cases = {"failure_cases.jsonl", "evidence_cases.jsonl"}
    if set(cases) != required_cases or not all(cases[name] for name in required_cases):
        raise ValueError("P18 adapter requires actual failure and evidence case rows")
    cf = counterfactual
    if not isinstance(cf, Mapping) or cf.get("available") is not True:
        raise ValueError("P18 adapter requires the real formal counterfactual audit")
    sample_ids = cf.get("sample_ids")
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)) or len(sample_ids) != 128:
        raise ValueError(
            "P18 counterfactual artifact requires the formal 128-case audit; "
            "P17's scheduled single-case training counterfactual is not an artifact substitute"
        )
    fingerprints = trainer.state_dict().get("resume_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ValueError("P18 adapter requires the public P17 source fingerprint")
    config_sha = _resolved_config_sha(runtime.config)
    common = {
        "schema_version": "rael-artifact-v1",
        "producer": "fate_oia.engine.supervise_acpr_rael_oia_foreground",
        "source_fingerprint_sha256": str(fingerprints["required_files_hash"]),
        "config_sha256": config_sha,
        "epoch": int(epoch),
        "sample_count": int(sample_count),
    }
    named = _weighted_value(rows, "named_ratio_by_target")
    latent = _weighted_value(rows, "latent_ratio_by_target")
    if not isinstance(named, Mapping) or not isinstance(latent, Mapping):
        raise TypeError("P18 named/latent target ratios must be mappings")
    per_target = [
        {"target": index, "named": float(named["action"][index]), "latent": float(latent["action"][index]), "global": float(1.0 - named["action"][index] - latent["action"][index])}
        for index in range(4)
    ] + [
        {"target": 4 + index, "named": float(named["reason"][index]), "latent": float(latent["reason"][index]), "global": float(1.0 - named["reason"][index] - latent["reason"][index])}
        for index in range(21)
    ]
    diagnostics = {
        "slot_mass": _weighted_value(rows, "slot_mass"),
        "slot_area_mean": _weighted_value(rows, "slot_area_mean"),
        "slot_area_std": _weighted_value(rows, "slot_area_std"),
        "slot_reliability_mean": _weighted_value(rows, "slot_reliability_mean"),
        "action_layer_weights": _weighted_value(rows, "action_layer_weights"),
        "reason_layer_weights": _weighted_value(rows, "reason_layer_weights"),
        "slot_layer_weights": _weighted_value(rows, "slot_layer_weights"),
        "unary_rms": _weighted_value(rows, "unary_rms"),
        "pairwise_rms": _weighted_value(rows, "pairwise_rms"),
        "global_rms": _weighted_value(rows, "global_rms"),
        "positive": _weighted_value(rows, "positive"),
        "negative": _weighted_value(rows, "negative"),
        "null_mass": _weighted_value(rows, "null_mass"),
        "reconstruction_error": _weighted_value(rows, "reconstruction_error"),
        "active_pair_count": int(round(_weighted_value(rows, "active_pair_count"))),
        "total_pair_count": int(round(_weighted_value(rows, "total_pair_count"))),
        "pu_scores": _weighted_value(rows, "pu_scores"),
        "pu_active_labels": rows[-1]["pu_active_labels"],
    }
    if int(step_count) <= 0 or last_step_result is None:
        raise ValueError("P18 adapter requires at least one real P17 optimizer step")
    last = last_step_result
    admission_summary = getattr(trainer, "last_admission_summary", None)
    if not isinstance(admission_summary, Mapping):
        raise ValueError("P18 adapter requires a real P13 gradient admission summary")
    required_admission = {"raw_norms", "projected_norms", "cosines", "caps", "ema_norms"}
    if set(admission_summary) != required_admission or not all(isinstance(admission_summary[name], Mapping) and admission_summary[name] for name in required_admission):
        raise ValueError("P18 adapter requires nonempty real P13 admission maps")
    gradient = {
        "cosine": {str(name): float(value) for name, value in admission_summary["cosines"].items()},
        "projection": {"registered": float(last.admission_registered_count), "triggered": float(last.admission_triggered_count)},
        "admission": {str(name): float(value) for name, value in admission_summary["raw_norms"].items()},
        "caps": {str(name): float(value) for name, value in admission_summary["caps"].items()},
        "ema": {str(name): float(value) for name, value in admission_summary["ema_norms"].items()},
    }
    action_candidates = action_calibration.get("candidates")
    reason_candidates = reason_calibration.get("candidates")
    if not isinstance(action_candidates, Sequence) or not isinstance(reason_candidates, Sequence) or len(action_candidates) != 4 or len(reason_candidates) != 4:
        raise ValueError("P18 calibration artifact requires four real train-calib candidates per family")
    calibration_candidates = []
    for action_candidate, reason_candidate in zip(action_candidates, reason_candidates):
        if not isinstance(action_candidate, Mapping) or not isinstance(reason_candidate, Mapping):
            raise TypeError("P18 calibration candidates must be mappings")
        action_metrics = action_candidate.get("metrics")
        reason_metrics = reason_candidate.get("metrics")
        if not isinstance(action_metrics, Mapping) or not isinstance(reason_metrics, Mapping):
            raise ValueError("P18 calibration candidate metrics are missing")
        calibration_candidates.append({"name": str(action_candidate.get("kind")), "joint": 0.5 * (float(action_metrics["mf1"]) + float(reason_metrics["mf1"]))})
    def chosen(value: Mapping[str, Any], key: str) -> list[float]:
        payload = value.get("chosen")
        if not isinstance(payload, Mapping) or not isinstance(payload.get(key), list):
            raise ValueError(f"P18 calibration chosen {key} is missing")
        return [float(item) for item in payload[key]]
    action_threshold = chosen(action_calibration, "threshold")
    reason_threshold = chosen(reason_calibration, "threshold")
    action_temperature = chosen(action_calibration, "temperature")
    reason_temperature = chosen(reason_calibration, "temperature")
    if len(pu_audit) != 21 or not all(isinstance(row, Mapping) for row in pu_audit):
        raise ValueError("P18 adapter requires the 21-row fixed train-audit PU result")
    return {
        "raw_metrics.json": {**common, **evaluation["raw_metrics"]},
        "deploy_metrics.json": {**common, **evaluation["deploy_metrics"]},
        "branch_metrics.json": {**common, **evaluation["branch_metrics"]},
        "per_action.json": {**common, **evaluation["per_action"]},
        "per_reason.json": {**common, **evaluation["per_reason"]},
        "slot_stats.json": {**common, "slot_count": 20, "mass": diagnostics["slot_mass"], "area": {"mean": diagnostics["slot_area_mean"], "std": diagnostics["slot_area_std"]}, "entropy": _weighted_value(rows, "slot_entropy"), "iou": _weighted_value(rows, "slot_iou"), "attributes": {"entity_type_entropy": _weighted_value(rows, "entity_type_entropy")}, "reliability": {"mean": diagnostics["slot_reliability_mean"]}},
        "layer_stats.json": {**common, "action_layer_weights": diagnostics["action_layer_weights"], "reason_layer_weights": diagnostics["reason_layer_weights"], "slot_layer_weights": diagnostics["slot_layer_weights"], "entropy": _weighted_value(rows, "layer_entropy"), "collapse": bool(any(bool(row["collapse"]) for row in rows))},
        "relation_stats.json": {**common, "unary": diagnostics["unary_rms"], "pairwise": diagnostics["pairwise_rms"], "null": diagnostics["null_mass"], "alpha": diagnostics["positive"], "active_pair_count": diagnostics["active_pair_count"], "total_pair_count": diagnostics["total_pair_count"]},
        "contribution_stats.json": {**common, "global": diagnostics["global_rms"], "unary": diagnostics["unary_rms"], "pairwise": diagnostics["pairwise_rms"], "positive": diagnostics["positive"], "negative": diagnostics["negative"], "reconstruction_error": diagnostics["reconstruction_error"]},
        "named_latent_global.json": {**common, "named_ratio": float(_weighted_value(rows, "named_ratio")["action"]), "latent_ratio": float(_weighted_value(rows, "latent_ratio")["action"]), "global_ratio": float(1.0 - _weighted_value(rows, "named_ratio")["action"] - _weighted_value(rows, "latent_ratio")["action"]), "per_target": per_target, "overall": {"named": float(_weighted_value(rows, "named_ratio")["action"]), "latent": float(_weighted_value(rows, "latent_ratio")["action"]), "global": float(1.0 - _weighted_value(rows, "named_ratio")["action"] - _weighted_value(rows, "latent_ratio")["action"])}},
        "gradient_admission.json": {**common, **gradient},
        "pu_stats.json": {**common, "labels": [{**dict(pu_audit[index]), "gate": bool(trainer.pu_active_labels[index].item()), "lambda": float(trainer.pu_lambda[index].item()), "soft_positive_count": float(_require_epoch_pu_count(trainer, index))} for index in range(21)]},
        "counterfactual.json": {**common, "sample_ids": list(sample_ids), "selected": {"effect": float(cf["selected_effect"])}, "control": {"effect": float(cf["control_effect"])}, "wrong": {"effect": float(cf["wrong_effect"])}, "valid_action_target_count": int(cf["valid_action_target_count"]), "valid_reason_target_count": int(cf["valid_reason_target_count"])},
        "calibration.json": {**common, "candidates": calibration_candidates, "chosen_thresholds": {"action": action_threshold, "reason": reason_threshold}, "temperature": {"action": sum(action_temperature) / len(action_temperature), "reason": sum(reason_temperature) / len(reason_temperature)}, "threshold_rms": {"action": (sum(value * value for value in action_threshold) / len(action_threshold)) ** 0.5, "reason": (sum(value * value for value in reason_threshold) / len(reason_threshold)) ** 0.5}, "raw_map": {"action": evaluation["raw_metrics"]["metrics"]["action"]["mAP"], "reason": evaluation["raw_metrics"]["metrics"]["reason"]["mAP"]}, "deploy_map": {"action": evaluation["deploy_metrics"]["metrics"]["action"]["mAP"], "reason": evaluation["deploy_metrics"]["metrics"]["reason"]["mAP"]}, "fallback": {"used": False, "reason": "train_calib_chosen"}},
        "failure_cases.jsonl": cases["failure_cases.jsonl"],
        "evidence_cases.jsonl": cases["evidence_cases.jsonl"],
        "logits_raw.pt": {"_meta": common, **evaluation["tensors"]["logits_raw"]},
        "logits_deploy.pt": {"_meta": common, **evaluation["tensors"]["logits_deploy"]},
        "labels.pt": {"_meta": common, **evaluation["tensors"]["labels"], "file_names": evaluation["file_names"]},
    }


def fit_train_calib_calibration(runtime: "RAELRuntime") -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """Fit post-hoc calibration only on the real disjoint train-calib loader."""

    import torch
    from fate_oia.engine.eval_acpr_rael_oia import evaluate_rael_test_only
    from fate_oia.utils.rael_artifacts import RAELArtifactWriter
    from fate_oia.utils.rael_posthoc_calibration import fit_posthoc_calibration

    # Import checks keep the full publisher bound to the P19/P18 production
    # owners rather than allowing a local substitute to shadow either API.
    if not callable(evaluate_rael_test_only) or not isinstance(RAELArtifactWriter, type):
        raise RuntimeError("P21 requires the production P19 evaluator and P18 artifact writer")

    action_rows: list[Any] = []
    reason_rows: list[Any] = []
    action_labels: list[Any] = []
    reason_labels: list[Any] = []
    names: list[str] = []
    prior_training = runtime.model.training
    runtime.model.eval()
    try:
        with torch.no_grad():
            for batch in runtime.train_calib_loader:
                device_batch = _device_batch(batch, runtime.device)
                field = runtime.model.encode_images(device_batch["images"])
                outputs = runtime.model.decode_from_field(field)
                action_rows.append(outputs["action_logits_final"].detach().float().cpu())
                reason_rows.append(outputs["reason_logits_final"].detach().float().cpu())
                action_labels.append(device_batch["action_targets"].detach().float().cpu())
                reason_labels.append(device_batch["reason_targets"].detach().float().cpu())
                names.extend(str(name) for name in batch["file_names"])
    finally:
        runtime.model.train(prior_training)
    if not names or len(names) != len(set(names)):
        raise ValueError("train-calib calibration requires unique real file names")
    split_hash = _split_hash(names)
    action = fit_posthoc_calibration(
        raw_logits=torch.cat(action_rows), labels=torch.cat(action_labels), split="train_calib",
        group_ids=tuple(range(4)), stable_ids=tuple(names), split_hash=split_hash,
    )
    reason = fit_posthoc_calibration(
        raw_logits=torch.cat(reason_rows), labels=torch.cat(reason_labels), split="train_calib",
        group_ids=tuple(range(21)), stable_ids=tuple(names), split_hash=split_hash,
    )
    return action, reason, split_hash


def labelwise_pu_audit(
    *,
    known_targets: Any,
    visual_baseline_logits: Any,
    recovered_scores: Any,
    hidden_positive_fraction: float,
    minimum_positive_count: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Measure hidden-known-positive recovery against the visual-only route.

    For each label, a fixed deterministic subset of known positives is hidden
    from the audit target.  Only those hidden positives versus observed-zero
    rows are scored.  This avoids the invalid shortcut of activating PU from
    AP on the labels already observed during normal supervision.
    """

    import torch
    from fate_oia.engine.eval_acpr_rael_oia import binary_average_precision_tie_stable

    if not 0.0 < float(hidden_positive_fraction) < 1.0:
        raise ValueError("hidden_positive_fraction must be strictly between zero and one")
    if tuple(known_targets.shape) != tuple(visual_baseline_logits.shape) or tuple(known_targets.shape) != tuple(recovered_scores.shape):
        raise ValueError("PU audit targets, visual baseline, and recovered scores must share [N,21]")
    if known_targets.ndim != 2 or known_targets.shape[1] != 21 or known_targets.shape[0] == 0:
        raise ValueError("PU audit requires a nonempty [N,21] fixed train-audit tensor")
    if not bool(torch.isfinite(visual_baseline_logits).all()) or not bool(torch.isfinite(recovered_scores).all()):
        raise ValueError("PU audit scores must be finite")

    known_targets = known_targets.detach().float().cpu()
    visual_baseline_logits = visual_baseline_logits.detach().float().cpu()
    recovered_scores = recovered_scores.detach().float().cpu()
    rows: list[dict[str, Any]] = []
    for label_id in range(21):
        known_positive = torch.nonzero(known_targets[:, label_id] > 0.5, as_tuple=False).flatten()
        observed_zero = torch.nonzero(known_targets[:, label_id] <= 0.5, as_tuple=False).flatten()
        positive_count = int(known_positive.numel())
        hidden_count = max(1, int(round(positive_count * float(hidden_positive_fraction)))) if positive_count else 0
        if positive_count < int(minimum_positive_count) or hidden_count == 0 or observed_zero.numel() == 0:
            rows.append({
                "label_id": label_id,
                "positive_count": positive_count,
                "hidden_positive_count": hidden_count,
                "observed_zero_count": int(observed_zero.numel()),
                "visual_hidden_auprc": None,
                "recovered_hidden_auprc": None,
                "recovery_delta": None,
                "recovery_lcb95": None,
                "eligible": False,
                "reason": "insufficient_known_positive_or_observed_zero",
            })
            continue
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1009 * label_id)
        chosen = known_positive.index_select(0, torch.randperm(positive_count, generator=generator)[:hidden_count])
        audit_index = torch.cat((chosen, observed_zero), dim=0)
        audit_target = torch.cat((torch.ones(hidden_count), torch.zeros(observed_zero.numel())), dim=0)
        visual = visual_baseline_logits[:, label_id].index_select(0, audit_index)
        recovered = recovered_scores[:, label_id].index_select(0, audit_index)
        visual_ap = binary_average_precision_tie_stable(visual, audit_target)
        recovered_ap = binary_average_precision_tie_stable(recovered, audit_target)
        if not math.isfinite(visual_ap) or not math.isfinite(recovered_ap):
            raise ValueError("hidden-known-positive PU audit must have both classes")
        bootstrap: list[float] = []
        bootstrap_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7919 * label_id)
        for _ in range(32):
            sample = torch.randint(audit_index.numel(), (audit_index.numel(),), generator=bootstrap_generator)
            sampled_target = audit_target.index_select(0, sample)
            candidate = binary_average_precision_tie_stable(recovered.index_select(0, sample), sampled_target)
            reference = binary_average_precision_tie_stable(visual.index_select(0, sample), sampled_target)
            if math.isfinite(candidate) and math.isfinite(reference):
                bootstrap.append(float(candidate - reference))
        if not bootstrap:
            raise ValueError("hidden-known-positive PU bootstrap produced no identifiable recovery delta")
        lcb95 = float(torch.quantile(torch.tensor(bootstrap), 0.05).item())
        rows.append({
            "label_id": label_id,
            "positive_count": positive_count,
            "hidden_positive_count": hidden_count,
            "observed_zero_count": int(observed_zero.numel()),
            "visual_hidden_auprc": float(visual_ap),
            "recovered_hidden_auprc": float(recovered_ap),
            "recovery_delta": float(recovered_ap - visual_ap),
            "recovery_lcb95": lcb95,
            "eligible": bool(lcb95 > 0.0),
            "reason": "hidden_positive_recovery",
        })
    return tuple(rows)


def run_fixed_train_audit_and_update_pu(runtime: "RAELRuntime", *, epoch: int) -> tuple[dict[str, Any], ...]:
    """Update PU gates only from the fixed, disjoint train-audit split.

    Epoch zero deliberately keeps every label off.  Later epochs use an
    empirical, deterministic bootstrap lower bound on the real per-label
    recovery delta; test examples and test metrics never enter this decision.
    """

    import torch

    targets: list[Any] = []
    visual_baseline_scores: list[Any] = []
    pu_scores: list[Any] = []
    prior_training = runtime.model.training
    runtime.model.eval()
    try:
        with torch.no_grad():
            for batch in runtime.train_audit_loader:
                device_batch = _device_batch(batch, runtime.device)
                field = runtime.model.encode_images(device_batch["images"])
                outputs = runtime.model.decode_from_field(field)
                # The formal model's visual-semantic head is the only
                # pre-private, pre-PU reference suitable for this audit.
                # ``reason_logits_global`` already contains the private
                # route, so using it would contaminate the recovery baseline.
                baseline = outputs.get("reason_logits_semantic_global")
                recovered = outputs.get("pu_scores")
                if not isinstance(baseline, torch.Tensor) or not isinstance(recovered, torch.Tensor):
                    raise ValueError("train-audit PU update requires visual-only reason logits and PU recovery scores")
                if baseline.shape != device_batch["reason_targets"].shape or recovered.shape != baseline.shape:
                    raise ValueError("train-audit PU outputs must be [B,21]")
                targets.append(device_batch["reason_targets"].detach().float().cpu())
                visual_baseline_scores.append(baseline.detach().float().cpu())
                pu_scores.append(recovered.detach().float().cpu())
    finally:
        runtime.model.train(prior_training)
    target = torch.cat(targets, dim=0)
    visual_baseline = torch.cat(visual_baseline_scores, dim=0)
    recovered = torch.cat(pu_scores, dim=0)
    if target.shape[1] != 21 or target.shape[0] == 0:
        raise ValueError("fixed train-audit split is empty or has invalid reason width")
    maximum = float(runtime.config["pu"]["max_lambda"])
    minimum_positive = int(runtime.config["pu"]["min_positive_count"])
    audit_rows = labelwise_pu_audit(
        known_targets=target,
        visual_baseline_logits=visual_baseline,
        recovered_scores=recovered,
        hidden_positive_fraction=float(runtime.config["pu"]["hidden_positive_fraction"]),
        minimum_positive_count=minimum_positive,
        # Fixed partition across epochs.  Epoch may affect model scores, never
        # which known positives are hidden for the admission decision.
        seed=int(runtime.config["training"]["seed"]),
    )
    active = torch.zeros(21, dtype=torch.bool)
    lambdas = torch.zeros(21, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for audit in audit_rows:
        label_id = int(audit["label_id"])
        lcb95 = audit["recovery_lcb95"]
        enabled = bool(epoch > 0 and audit["eligible"] is True)
        active[label_id] = enabled
        lambdas[label_id] = maximum * min(1.0, max(0.0, float(lcb95))) if enabled and isinstance(lcb95, float) else 0.0
        rows.append({
            **audit,
            "lambda": float(lambdas[label_id].item()),
            "decision": "active" if enabled else ("epoch0_off" if epoch == 0 else "rejected"),
        })
    runtime.trainer.set_pu_label_gate(active, lambdas)
    if epoch == 0 and (bool(active.any()) or bool(lambdas.any())):
        raise RuntimeError("epoch zero PU policy must keep every label disabled")
    return tuple(rows)


def load_rael_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RAEL config must be a mapping")
    required = ("data", "backbone", "model", "training", "runtime", "constraints")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"RAEL config missing {missing}")
    if value["constraints"].get("feature_cache_enabled") is not False or value["constraints"].get("token_compression") != "none":
        raise ValueError("RAEL launch rejects feature cache or token compression")
    return dict(value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_config_sha(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, check=True, capture_output=True).stdout.strip()


def _split_hash(names: Sequence[str]) -> str:
    canonical = [str(name).replace("\\", "/") for name in names]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _audited_bdd100k_layout(sources: Mapping[str, Any]) -> Mapping[str, Path]:
    """Validate the host-declared per-frame BDD100K train/val layout.

    The actual host stores label JSON per frame under ``labels/100k/<split>``.
    We accept only those explicit directories from YAML and verify direct JSON
    children before a DINO model or GPU is constructed.  No aggregate-path
    fallback, root walk, or dense drivable PNG discovery is permitted.
    """

    label_directories = sources.get("label_directories")
    if not isinstance(label_directories, Mapping) or set(label_directories) != {"train", "val"}:
        raise ValueError("grounding_sources.label_directories must explicitly contain train and val")
    layout: dict[str, Path] = {}
    for split in ("train", "val"):
        raw_path = label_directories[split]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"grounding_sources.label_directories.{split} must be a nonempty path")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"configured BDD100K {split} label directory does not exist: {path}")
        if not any(path.glob("*.json")):
            raise ValueError(f"configured BDD100K {split} label directory has no direct JSON files: {path}")
        layout[split] = path
    return layout


def _grounding_index(
    config: Mapping[str, Any],
    *,
    include_file_names: Sequence[str] | None = None,
) -> Any:
    from fate_oia.datasets.bdd100k_task_aware_index import RAELTaskAwareBDD100KIndex

    sources = config.get("grounding_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("grounding_sources must explicitly provide the audited train/val per-frame label directories")
    return RAELTaskAwareBDD100KIndex(
        label_directories=_audited_bdd100k_layout(sources),
        include_file_names=include_file_names,
    )


class _RealRAELDataset:
    """Wrap official BDD-OIA samples with one image transform and metadata-only grounding."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        split: str,
        index: Any,
        indices: Sequence[int] | None = None,
        base: Any | None = None,
    ) -> None:
        from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
        from fate_oia.transforms_rael import RAELGroundingTransform

        data = config["data"]
        self.base = base if base is not None else BDDOIAMultiTaskDataset(data_root=data["data_root"], raw_root=data["raw_root"], split=split, action_dim=4, reason_dim=21, load_image=False)
        self.index = index
        self.transform = RAELGroundingTransform(image_height=int(data["image_height"]), image_width=int(data["image_width"]), patch_size=8)
        self.indices = tuple(range(len(self.base))) if indices is None else tuple(int(value) for value in indices)
        if not self.indices:
            raise ValueError(f"RAEL {split} dataset selection is empty")
        self.split = split
        # Metadata only: no image decode.  This is the stable split identity
        # consumed by P15/P19, not a cached feature or cached batch.
        self.file_names = tuple(str(self.base[item]["file_name"]) for item in self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        sample = self.base[self.indices[index]]
        record = self.index.lookup(str(sample["file_name"]))
        with Image.open(sample["image_path"]) as image:
            transformed = self.transform(image.convert("RGB"), record)
        return {
            "split": self.split,
            "file_name": str(sample["file_name"]),
            "images": transformed.image,
            "action_targets": sample["action"],
            "reason_targets": sample["reason"],
            "grounding_record": transformed.record,
            "transform_meta": transformed.meta,
        }


def _collate_rael(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    if not rows or len({row["split"] for row in rows}) != 1:
        raise ValueError("RAEL batches must be nonempty and one split")
    return {
        "split": rows[0]["split"],
        "images": torch.stack([row["images"] for row in rows]),
        "action_targets": torch.stack([row["action_targets"] for row in rows]),
        "reason_targets": torch.stack([row["reason_targets"] for row in rows]),
        "file_names": tuple(str(row["file_name"]) for row in rows),
        "grounding_records": tuple(row["grounding_record"] for row in rows),
        "transform_meta": tuple(row["transform_meta"] for row in rows),
        "grounding_mode": "dynamic",
    }


def _device_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {name: value.to(device, non_blocking=True) if name in {"images", "action_targets", "reason_targets"} else value for name, value in batch.items()}


def _timed_device_batches(loader: Iterable[Mapping[str, Any]], device: Any) -> Iterator[Mapping[str, Any]]:
    """Yield one epoch exactly once while measuring real loader plus H2D time."""

    import time

    iterator = iter(loader)
    while True:
        began = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        device_batch = _device_batch(batch, device)
        device_batch["_data_time"] = time.perf_counter() - began
        yield device_batch


def _deterministic_indices(total: int, *, seed: int, fraction: float | None = None, limit: int | None = None, exclude: set[int] | None = None) -> tuple[int, ...]:
    import random

    values = [value for value in range(total) if value not in (exclude or set())]
    random.Random(seed).shuffle(values)
    wanted = len(values) if fraction is None else max(1, int(math.floor(len(values) * fraction)))
    if limit is not None:
        wanted = min(wanted, int(limit))
    return tuple(sorted(values[:wanted]))


@dataclass
class RAELRuntime:
    config: dict[str, Any]
    device: Any
    model: Any
    trainer: Any
    train_loader: Any
    train_audit_loader: Any
    train_calib_loader: Any
    test_loader: Any
    test_split_hash: str
    repository_root: Path


def build_rael_runtime(*, config_path: str | Path, device: str, batch_size: int, gradient_accumulation_steps: int, num_workers: int, mode: str, max_train_samples: int | None = None, max_test_samples: int | None = None) -> RAELRuntime:
    import torch
    from torch.utils.data import DataLoader
    from fate_oia.models.rael_dino_field import RAELDinoFieldExtractor
    from fate_oia.models.rael_oia_model import RAELOIAModel
    from fate_oia.engine.train_acpr_rael_oia import RAELTrainer

    config = load_rael_config(config_path)
    root = Path(config_path).resolve().parents[1]
    seed = int(config["training"]["seed"])
    from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset

    data = config["data"]
    train_base = BDDOIAMultiTaskDataset(data_root=data["data_root"], raw_root=data["raw_root"], split="train", action_dim=4, reason_dim=21, load_image=False)
    test_base = BDDOIAMultiTaskDataset(data_root=data["data_root"], raw_root=data["raw_root"], split="test", action_dim=4, reason_dim=21, load_image=False)
    audit_count = min(1024, len(train_base))
    audit_indices = _deterministic_indices(len(train_base), seed=seed, limit=audit_count)
    # Train representation, fixed train-audit PU admission, and post-hoc
    # train-calib are three disjoint views.  Test data never participates.
    calib_indices = _deterministic_indices(
        len(train_base),
        seed=seed + 2,
        fraction=float(config["calibration"]["train_calib_fraction"]),
        limit=max_train_samples,
        exclude=set(audit_indices),
    )
    main_indices = _deterministic_indices(
        len(train_base),
        seed=seed + 1,
        limit=max_train_samples,
        exclude=set(audit_indices).union(calib_indices),
    )
    if set(audit_indices).intersection(calib_indices) or set(audit_indices).intersection(main_indices) or set(calib_indices).intersection(main_indices):
        raise RuntimeError("train representation, fixed audit, and train-calib splits must be disjoint")
    test_indices = _deterministic_indices(len(test_base), seed=seed + 3, limit=max_test_samples)
    required_train_indices = sorted(set(main_indices).union(audit_indices).union(calib_indices))
    required_file_names = [
        *(str(train_base[index]["file_name"]) for index in required_train_indices),
        *(str(test_base[index]["file_name"]) for index in test_indices),
    ]
    index = _grounding_index(config, include_file_names=required_file_names)
    train_data = _RealRAELDataset(config=config, split="train", index=index, indices=main_indices, base=train_base)
    train_audit_data = _RealRAELDataset(config=config, split="train", index=index, indices=audit_indices, base=train_base)
    calib_data = _RealRAELDataset(config=config, split="train", index=index, indices=calib_indices, base=train_base)
    test_data = _RealRAELDataset(config=config, split="test", index=index, indices=test_indices, base=test_base)
    # Windows spawn copies each dataset-owned metadata index into every worker.
    # Bound the train pool and keep the three sequential auxiliary loaders in
    # the parent process instead of creating four independent persistent pools.
    train_worker_count = min(int(num_workers), 4)
    train_loader_kwargs = {"num_workers": train_worker_count, "pin_memory": True, "collate_fn": _collate_rael}
    if train_worker_count > 0:
        train_loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    aux_loader_kwargs = {"num_workers": 0, "pin_memory": True, "collate_fn": _collate_rael}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **train_loader_kwargs)
    train_audit_loader = DataLoader(train_audit_data, batch_size=batch_size, shuffle=False, **aux_loader_kwargs)
    calib_loader = DataLoader(calib_data, batch_size=batch_size, shuffle=False, **aux_loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **aux_loader_kwargs)
    backbone = config["backbone"]
    dino = RAELDinoFieldExtractor(arch=backbone["arch"], patch_size=int(backbone["patch_size"]), selected_layers=tuple(backbone["selected_layers"]), checkpoint_key=backbone["checkpoint_key"], pretrained_weights=backbone["pretrained_weights"])
    resolved_device = torch.device(device)
    model = RAELOIAModel(dino_extractor=dino, reason_schema_path=root / "configs/rael_reason_semantics.yaml", dim=int(config["model"]["dim"]), num_heads=int(config["model"]["attention_heads"])).to(resolved_device)
    total_updates = max(3, int(config["training"]["epochs"]) * max(1, math.ceil(len(train_loader) / gradient_accumulation_steps)))
    trainer = RAELTrainer(model, total_optimizer_updates=total_updates, gradient_accumulation_steps=gradient_accumulation_steps, precision=config["training"]["precision"])
    return RAELRuntime(config, resolved_device, model, trainer, train_loader, train_audit_loader, calib_loader, test_loader, _split_hash(test_data.file_names), root)


class _RealRuntimeRunner:
    """P20 adapter which executes real DataLoader batches through RAELTrainer."""

    def __init__(self, runtime: RAELRuntime) -> None:
        self.runtime = runtime
        self._iterator: Iterator[Mapping[str, Any]] = iter(runtime.train_loader)
        self._context: dict[str, Any] = {}

    def _next_train_batch(self) -> Mapping[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            # DataLoader itself is re-iterable; do not cache one epoch of
            # image tensors in a cycling iterator.
            self._iterator = iter(self.runtime.train_loader)
            return next(self._iterator)

    def artifact_provenance(self) -> Mapping[str, Any]:
        config_path = self.runtime.repository_root / "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml"
        return {"schema_version": "rael-artifact-v1", "producer": "P21RealRuntimeRunner", "source_fingerprint_sha256": hashlib.sha256(_git_head(self.runtime.repository_root).encode()).hexdigest(), "config_sha256": _sha256_file(config_path)}

    def configure_runtime_profile(self, context: Mapping[str, Any]) -> None:
        self._context = dict(context)

    def optimizer_update(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        import time
        import torch

        data_time = 0.0
        h2d_time = 0.0
        results = []
        for _ in range(int(context["gradient_accumulation_steps"])):
            data_started = time.perf_counter()
            cpu_batch = self._next_train_batch()
            data_time += time.perf_counter() - data_started
            h2d_started = time.perf_counter()
            batch = _device_batch(cpu_batch, self.runtime.device)
            if torch.cuda.is_available() and self.runtime.device.type == "cuda":
                torch.cuda.synchronize(self.runtime.device)
            h2d_time += time.perf_counter() - h2d_started
            batch["_data_time"] = data_time
            results.append(self.runtime.trainer.train_microbatch(batch, epoch=0))
        final = results[-1]
        observed_counts = [int(result.mechanism_observation["dino_call_count"]) for result in results]
        return {
            "samples": int(context["batch_size"]) * int(context["gradient_accumulation_steps"]),
            "microbatches": int(context["gradient_accumulation_steps"]),
            "finite": all(all(bool(value.isfinite().all()) for value in result.components.values()) for result in results),
            "dino_call_count_per_microbatch": observed_counts[0] if len(set(observed_counts)) == 1 else -1,
            "dino_call_count_total": sum(observed_counts),
            "data_time": data_time,
            "h2d_time": h2d_time,
            "dino_time": sum(float(result.mechanism_observation["field_time"]) for result in results),
            "backward_time": sum(float(result.mechanism_observation["backward_time"]) for result in results),
            "optimizer_time": sum(float(result.mechanism_observation["optimizer_time"]) for result in results),
            "mechanism_flags": dict(context["mechanism_flags"]),
            "owner_parameter_delta": final.owner_parameter_delta,
        }

    def measure_counterfactual_overhead(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        import time

        trainer = self.runtime.trainer
        prepare = getattr(trainer, "prepare_counterfactual_handoff", None)
        replay = getattr(trainer, "replay_counterfactual_from_encoded_field", None)
        if not callable(prepare) or not callable(replay):
            raise RuntimeError("P20 requires the P21 public encoded-field replay handoff")
        batch = _device_batch(self._next_train_batch(), self.runtime.device)
        handoff = prepare(batch)
        began = time.perf_counter()
        result = replay(
            handoff,
            target_family="action",
            optimizer_update=max(1, int(self.runtime.trainer.optimizer_step) + 1),
        )
        elapsed = time.perf_counter() - began
        if int(result.get("replay_dino_call_count", -1)) != 0:
            raise RuntimeError("counterfactual replay must not issue an additional DINO call")
        return {
            "counterfactual_replay_executed": True,
            "valid_target_count": int(result.get("valid_target_count", 0)),
            "finite": bool(result.get("available", False)),
            "extra_dino_call_count": 0,
            "counterfactual_time": elapsed,
        }


def build_runtime_runner(*, candidate: Mapping[str, Any], device: str) -> _RealRuntimeRunner:
    config_path = os.environ.get("RAEL_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("RAEL_CONFIG_PATH is required for the real P20 runner factory")
    return _RealRuntimeRunner(build_rael_runtime(config_path=config_path, device=device, batch_size=int(candidate["batch_size"]), gradient_accumulation_steps=int(candidate["gradient_accumulation_steps"]), num_workers=int(candidate["num_workers"]), mode="profile", max_train_samples=int(os.environ.get("RAEL_PROFILE_MAX_TRAIN", "128")), max_test_samples=1))


def _mechanism_row(result: Any) -> dict[str, Any]:
    observed = result.mechanism_observation.get("dino_call_count")
    if isinstance(observed, bool) or not isinstance(observed, int):
        raise ValueError("smoke requires a real integer dino_call_count observation")
    mechanism_keys = (
        "action_unary_rms_over_global", "action_pairwise_rms_over_global",
        "reason_unary_rms_over_global", "reason_pairwise_rms_over_global",
        "gamma_AS", "gamma_RA", "gamma_unary", "gamma_pairwise",
        "q_view_source_counts", "q_view_bootstrap_count", "rho_nonzero_rate",
        "slot_feature_dropout_consistency_mean", "active_entity_count",
        "named_contribution_ratio", "latent_contribution_ratio",
        "pu_active_label_count", "pu_soft_positive_count",
    )
    mechanism = {
        key: result.mechanism_observation[key]
        for key in mechanism_keys
        if key in result.mechanism_observation
    }
    return {"dino_call_count": observed, "optimizer_step": int(result.optimizer_step), "finite": all(bool(value.isfinite().all()) for value in result.components.values()), "mechanism": mechanism, "owner_parameter_delta": {name: float(value) for name, value in result.owner_parameter_delta.items()}, "owner_gradient_norms": {name: float(value) for name, value in result.owner_gradient_norms_pre_clip.items()}}


def require_mode_gate(mode: str, gate_path: str | Path, *, expected_git_head: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    if mode == "smoke":
        return
    path = Path(gate_path)
    if not path.is_file():
        raise FileNotFoundError("FULL_TRAIN_READY gate is required for full mode")
    payload = json.loads(path.read_text(encoding="utf-8"))
    override = payload.get("pilot_override")
    smoke = payload.get("smoke_result")
    if (
        payload.get("pass") is not True
        or payload.get("git_head") != expected_git_head
        or payload.get("unresolved") != []
        or not isinstance(override, Mapping)
        or override.get("pilot_protocol_override") is not True
        or override.get("pilot_completed") is not False
        or override.get("replacement") != "minimal_real_smoke_only"
        or not isinstance(smoke, Mapping)
        or smoke.get("passed") is not True
    ):
        raise RuntimeError("FULL_TRAIN_READY is stale or failed")


def _assert_full_contract_available(runtime: RAELRuntime) -> None:
    """Reject a 14-epoch request until its public P17/P18 hand-off is complete.

    P21 is not allowed to manufacture epoch metrics/artifacts itself.  The
    trainer must expose its encoded field for counterfactual replay and the
    public epoch publisher must receive a real P18 artifact builder.
    """
    trainer = runtime.trainer
    if not callable(getattr(trainer, "train_epoch_and_publish", None)):
        raise RuntimeError(
            "full launch blocked: P17 lacks public train_epoch_and_publish; P21 will not label a training-only loop as a 14-epoch test-only run"
        )
    if not callable(getattr(trainer, "replay_counterfactual_from_encoded_field", None)):
        raise RuntimeError(
            "full launch blocked: P17 lacks public read-only field replay required for the P20 zero-extra-DINO counterfactual contract"
        )


def _artifact_provenance(runtime: RAELRuntime) -> dict[str, Any]:
    state = runtime.trainer.state_dict()
    fingerprints = state.get("resume_fingerprints")
    if not isinstance(fingerprints, Mapping) or not isinstance(fingerprints.get("required_files_hash"), str):
        raise RuntimeError("P18 publication requires the live P17 resume fingerprint")
    return {
        "schema_version": "rael-artifact-v1",
        "producer": "fate_oia.engine.supervise_acpr_rael_oia_foreground",
        "source_fingerprint_sha256": fingerprints["required_files_hash"],
        "config_sha256": _resolved_config_sha(runtime.config),
    }


def _load_selected_runtime_profile(path: str | Path, *, provenance: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Consume an actual P20 profile; never invent throughput or memory rows."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError("P20 runtime profile argument must be the measured profile directory")
    payload = json.loads((root / "runtime_profile.json").read_text(encoding="utf-8"))
    selected_payload = json.loads((root / "selected_runtime_profile.json").read_text(encoding="utf-8"))
    steps = [
        json.loads(line)
        for line in (root / "runtime_steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not isinstance(payload, Mapping) or not isinstance(selected_payload, Mapping):
        raise ValueError("P20 runtime profile artifacts must be JSON mappings")
    candidates = payload.get("candidates")
    selected = selected_payload.get("selected")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ValueError("P20 runtime profile must contain real nonempty candidates")
    if not isinstance(selected, Mapping) or not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ValueError("P20 runtime profile must contain selected profile and real runtime_steps")
    runtime_profile = {**provenance, "candidates": [dict(row) for row in candidates]}
    selected_profile = {**provenance, "selected": dict(selected), "reason": str(selected_payload.get("reason") or "P20 selected fastest stable measured candidate")}
    return runtime_profile, selected_profile, tuple(dict(row) for row in steps)


def _initialize_full_artifact_root(
    *,
    runtime: RAELRuntime,
    writer: Any,
    command_line: Sequence[str],
    runtime_profile_path: str | Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Bind immutable P18 run identity before the first train update."""

    from fate_oia.utils.rael_artifacts import trainer_run_artifact_contract

    provenance = _artifact_provenance(runtime)
    source_contract = trainer_run_artifact_contract(
        runtime.trainer,
        artifact_context=provenance,
    )
    profile, selected_profile, profile_steps = _load_selected_runtime_profile(
        runtime_profile_path, provenance=provenance
    )
    git_head = _git_head(runtime.repository_root)
    manifest = {
        **provenance,
        "git_head": git_head,
        "remote_head": git_head,
        "base_head": git_head,
        "command": list(command_line),
        "data_split": {"name": "test", "ids_sha256": runtime.test_split_hash, "sample_count": len(runtime.test_loader.dataset)},
        "dino": {"source_sha256": hashlib.sha256(str(runtime.config["backbone"]["pretrained_weights"]).encode("utf-8")).hexdigest(), "weight_sha256": _sha256_file(Path(runtime.config["backbone"]["pretrained_weights"]))},
        "formal_flags": {"direct_image": True, "feature_cache_enabled": False, "token_compression": "none", "test_only": True},
        "selected_runtime_profile": str(selected_profile["selected"]["name"]),
        "seed": int(runtime.config["training"]["seed"]),
        "test_selected": True,
        "publication_eligible": False,
    }
    config_payload = {
        **provenance,
        "resolved_config": runtime.config,
        "resolved_config_sha256": provenance["config_sha256"],
    }
    writer.write_run_file("run_manifest.json", manifest)
    writer.write_run_file("config_resolved.yaml", config_payload)
    writer.write_run_file("source_fingerprint.json", source_contract["source_fingerprint"])
    writer.write_run_file("runtime_profile.json", profile)
    writer.write_run_file("selected_runtime_profile.json", selected_profile)
    writer.write_run_file("optimizer_owners.json", source_contract["optimizer_owners"])
    return provenance, profile_steps


def _append_epoch_run_rows(
    *,
    writer: Any,
    runtime: RAELRuntime,
    provenance: Mapping[str, Any],
    epoch: int,
    publication: Mapping[str, Any],
    best_flags: Mapping[str, bool],
    is_best: bool,
) -> None:
    """Append only rows derived from the just-completed real epoch."""

    evaluation = publication.get("evaluation")
    last_step = publication.get("last_step_result")
    step_count = publication.get("step_count")
    if last_step is None or not isinstance(step_count, int) or step_count <= 0 or not isinstance(evaluation, Mapping):
        raise ValueError("P18 run rows require a completed real epoch publication")
    admission = getattr(runtime.trainer, "last_admission_summary", None)
    counterfactual = getattr(runtime.trainer, "last_counterfactual_result", None)
    if not isinstance(admission, Mapping) or not isinstance(counterfactual, Mapping):
        raise ValueError("P18 run rows require public P13/P14 epoch state")
    context = {
        **provenance,
        "epoch": int(epoch),
        "total_optimizer_updates": int(runtime.trainer.schedule.total_optimizer_updates),
        "valid_counts": {
            "grounding": int(step_count),
            "counterfactual": int(counterfactual["valid_action_target_count"]) + int(counterfactual["valid_reason_target_count"]),
        },
        "admission": admission,
    }
    mechanism = getattr(runtime.trainer, "mechanism_stats_from_step", None)
    if not callable(mechanism):
        raise RuntimeError("P17 must expose real mechanism_stats_from_step for P18 publication")
    # Counterfactual audit completes after the train stream.  Record one
    # truthful epoch-end mechanism row from the actual final update, rather
    # than copying that audit outcome onto every earlier update.
    writer.append_run_jsonl(
        "mechanism_stats.jsonl",
        mechanism(last_step, artifact_context=context, counterfactual=counterfactual),
    )
    raw_metrics = evaluation["raw_metrics"]["metrics"]
    deploy_metrics = evaluation["deploy_metrics"]["metrics"]
    writer.append_run_jsonl(
        "metrics_summary.jsonl",
        {
            **provenance,
            "epoch": int(epoch),
            "raw_action": raw_metrics["action"],
            "raw_reason": raw_metrics["reason"],
            "raw_joint": raw_metrics["joint"],
            "deploy_action": deploy_metrics["action"],
            "deploy_reason": deploy_metrics["reason"],
            "deploy_joint": deploy_metrics["joint"],
            "best_flags": dict(best_flags),
            "is_best": bool(is_best),
        },
    )
    pu_audit = publication.get("pu_audit")
    if not isinstance(pu_audit, Sequence) or len(pu_audit) != 21:
        raise ValueError("P18 publication requires per-label real PU audit outputs")
    for row in pu_audit:
        writer.append_run_jsonl("pu_audit.jsonl", {**provenance, "epoch": int(epoch), **dict(row)})


def _append_streamed_train_step_rows(
    *,
    writer: Any,
    runtime: RAELRuntime,
    provenance: Mapping[str, Any],
    epoch: int,
    step: Any,
) -> None:
    """Persist only this completed update; never retain an epoch in memory."""

    from fate_oia.utils.rael_artifacts import step_result_artifact_rows

    admission = getattr(runtime.trainer, "last_admission_summary", None)
    if not isinstance(admission, Mapping):
        raise ValueError("streamed P18 rows require the real P13 admission summary")
    rows = step_result_artifact_rows(
        step,
        artifact_context={
            **provenance,
            "epoch": int(epoch),
            "total_optimizer_updates": int(runtime.trainer.schedule.total_optimizer_updates),
            "valid_counts": {"grounding": 1, "counterfactual": 0},
            "admission": admission,
        },
    )
    writer.append_run_jsonl("loss_components.jsonl", rows["loss_components"])
    writer.append_run_jsonl("gradient_admission.jsonl", rows["gradient_admission"])


def _save_full_checkpoints(
    *,
    output_dir: Path,
    runtime: RAELRuntime,
    epoch: int,
    evaluation: Mapping[str, Any],
    best: dict[str, float],
) -> dict[str, bool]:
    """Checkpoint exactly the state that produced this test-only epoch result."""

    import torch

    deploy = evaluation["deploy_metrics"]["metrics"]
    branches = evaluation.get("branch_metrics", {}).get("branches")
    if not isinstance(branches, Sequence):
        raise ValueError("checkpoint selection requires real diagnostic branch metrics")
    global_branch = next((row for row in branches if isinstance(row, Mapping) and row.get("name") == "global"), None)
    if not isinstance(global_branch, Mapping):
        raise ValueError("checkpoint selection requires the real global diagnostic action branch")
    global_metrics = global_branch.get("metrics")
    if not isinstance(global_metrics, Mapping) or not isinstance(global_metrics.get("action"), Mapping):
        raise ValueError("global diagnostic branch lacks action metrics")
    flags = {
        "deploy_joint": float(deploy["joint"]) > best["deploy_joint"],
        "action_mf1": float(deploy["action"]["mF1"]) > best["action_mf1"],
        "exp_mf1": float(deploy["reason"]["mF1"]) > best["exp_mf1"],
        "exp_map": float(deploy["reason"]["mAP"]) > best["exp_map"],
        "global_action": float(global_metrics["action"]["mF1"]) > best["global_action"],
    }
    payload = {
        "epoch": int(epoch),
        "trainer": runtime.trainer.state_dict(),
        "evaluation_selection": evaluation["selection"],
        "deploy_metrics": deploy,
    }
    torch.save(payload, output_dir / "checkpoint_latest.pth")
    if flags["deploy_joint"]:
        torch.save(payload, output_dir / "checkpoint_best_test_deploy_joint.pth")
        best["deploy_joint"] = float(deploy["joint"])
    if flags["action_mf1"]:
        torch.save(payload, output_dir / "checkpoint_best_test_action_mf1.pth")
        best["action_mf1"] = float(deploy["action"]["mF1"])
    if flags["exp_mf1"]:
        torch.save(payload, output_dir / "checkpoint_best_test_exp_mf1.pth")
        best["exp_mf1"] = float(deploy["reason"]["mF1"])
    if flags["exp_map"]:
        torch.save(payload, output_dir / "checkpoint_best_test_exp_map.pth")
        best["exp_map"] = float(deploy["reason"]["mAP"])
    if flags["global_action"]:
        torch.save(payload, output_dir / "checkpoint_best_test_global_action.pth")
        best["global_action"] = float(global_metrics["action"]["mF1"])
    return flags


def run_rael_mode(*, mode: str, config_path: str | Path, output_dir: str | Path, device: str, batch_size: int, gradient_accumulation_steps: int, num_workers: int, max_train_samples: int | None = None, max_test_samples: int | None = None, max_optimizer_updates: int | None = None, full_gate: str | Path | None = None, runtime_profile: str | Path | None = None) -> dict[str, Any]:
    repository_root = Path(config_path).resolve().parents[1]
    if mode == "full":
        if full_gate is None:
            raise ValueError("full mode requires --full_gate bound to the current HEAD")
        require_mode_gate(mode, full_gate, expected_git_head=_git_head(repository_root))
    runtime = build_rael_runtime(config_path=config_path, device=device, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps, num_workers=num_workers, mode=mode, max_train_samples=max_train_samples, max_test_samples=max_test_samples)
    if mode == "full":
        _assert_full_contract_available(runtime)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        # Smoke is deliberately only a few real optimizer updates.  It does
        # not claim an epoch metric, P18 publication, or full-train checkpoint.
        updates = int(max_optimizer_updates or 3)
        iterator = iter(runtime.train_loader)
        rows: list[dict[str, Any]] = []
        for update in range(updates):
            result = None
            for _ in range(gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(runtime.train_loader)
                    batch = next(iterator)
                result = runtime.trainer.train_microbatch(_device_batch(batch, runtime.device), epoch=0)
            if result is None:
                raise RuntimeError("minimal smoke did not execute a real update")
            row = _mechanism_row(result)
            rows.append(row)
            print(json.dumps({"rael_update": update + 1, **row}, sort_keys=True), flush=True)
        smoke = {
            "mode": "smoke",
            "synthetic": False,
            "git_head": _git_head(runtime.repository_root),
            "config_sha256": _resolved_config_sha(runtime.config),
            "updates": len(rows),
            "pilot_protocol_override": True,
            "full_epoch_claimed": False,
        }
        (root / "smoke_manifest.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "smoke_updates.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        # Keep the minimal smoke directly consumable by the fail-closed audit.
        (root / "run_manifest.json").write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (root / "config_resolved.yaml").write_text(yaml.safe_dump(runtime.config, sort_keys=True), encoding="utf-8")
        fingerprints = runtime.trainer.state_dict().get("resume_fingerprints", {})
        (root / "source_fingerprint.json").write_text(
            json.dumps(fingerprints, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "mechanism_stats.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return {"mode": mode, "updates": len(rows), "output_dir": str(root), "rows": rows}

    if runtime_profile is None:
        raise ValueError("full mode requires a measured P20 --runtime_profile; throughput/memory may not be fabricated")
    from fate_oia.engine.export_rael_cases import RAELCaseExportCollector
    from fate_oia.utils.rael_artifacts import RAELArtifactWriter

    writer = RAELArtifactWriter(root)
    command = ["python", "-m", "fate_oia.engine.supervise_acpr_rael_oia_foreground", "--mode", "full"]
    provenance, profile_steps = _initialize_full_artifact_root(
        runtime=runtime, writer=writer, command_line=command, runtime_profile_path=runtime_profile
    )
    for row in profile_steps:
        writer.append_run_jsonl("runtime_steps.jsonl", {**provenance, **row})
    epochs = int(runtime.config["training"]["epochs"])
    best = {
        "deploy_joint": float("-inf"),
        "action_mf1": float("-inf"),
        "exp_mf1": float("-inf"),
        "exp_map": float("-inf"),
        "global_action": float("-inf"),
    }
    published: list[dict[str, Any]] = []
    action_schema = runtime.repository_root / "configs" / "rael_action_semantics.yaml"
    reason_schema = runtime.repository_root / "configs" / "rael_reason_semantics.yaml"
    if not action_schema.is_file() or not reason_schema.is_file():
        raise FileNotFoundError("full P19 evaluation requires real action/reason semantic schema files")
    for epoch in range(epochs):
        case_provenance = {**provenance, "epoch": epoch, "sample_count": len(runtime.test_loader.dataset)}
        collector = RAELCaseExportCollector(max_failure_cases=32, max_evidence_cases=32, top_slots=5)

        def epoch_transition() -> Mapping[str, Any]:
            # Fixed train-audit is disjoint from both train representation
            # updates and train-calib.  It is the sole owner of per-label PU
            # admission for the next epoch and of the persisted 21-row audit.
            pu_audit = run_fixed_train_audit_and_update_pu(runtime, epoch=epoch)
            action_calibration, reason_calibration, train_calib_hash = fit_train_calib_calibration(runtime)
            return {
                "pu_audit": pu_audit,
                "action_calibration": action_calibration,
                "reason_calibration": reason_calibration,
                "train_calib_split_hash": train_calib_hash,
            }

        publication = runtime.trainer.train_epoch_and_publish(
            _timed_device_batches(runtime.train_loader, runtime.device),
            epoch=epoch,
            test_batches=runtime.test_loader,
            epoch_transition=epoch_transition,
            expected_test_split_hash=runtime.test_split_hash,
            action_schema_path=action_schema,
            reason_schema_path=reason_schema,
            device=runtime.device,
            writer=writer,
            epoch_artifact_builder=lambda **values: build_p18_epoch_artifacts(runtime=runtime, **values),
            on_step_result=lambda step: _append_streamed_train_step_rows(
                writer=writer,
                runtime=runtime,
                provenance=provenance,
                epoch=epoch,
                step=step,
            ),
            case_collector=collector,
            case_export_provenance=case_provenance,
        )
        flags = _save_full_checkpoints(output_dir=root, runtime=runtime, epoch=epoch, evaluation=publication["evaluation"], best=best)
        _append_epoch_run_rows(
            writer=writer,
            runtime=runtime,
            provenance=provenance,
            epoch=epoch,
            publication=publication,
            best_flags=flags,
            is_best=flags["deploy_joint"],
        )
        metric = publication["evaluation"]["deploy_metrics"]["metrics"]
        summary = {"rael_epoch": epoch, "Act_mF1": metric["action"]["mF1"], "Exp_mF1": metric["reason"]["mF1"], "joint": metric["joint"], "best": flags}
        published.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return {"mode": "full", "epochs": epochs, "output_dir": str(root), "published": published}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAEL foreground supervisor")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_optimizer_updates", type=int)
    parser.add_argument("--full_gate")
    parser.add_argument("--runtime_profile")
    args = parser.parse_args(argv)
    run_rael_mode(mode=args.mode, config_path=args.config, output_dir=args.output_dir, device=args.device, batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, num_workers=args.num_workers, max_train_samples=args.max_train_samples, max_test_samples=args.max_test_samples, max_optimizer_updates=args.max_optimizer_updates, full_gate=args.full_gate, runtime_profile=args.runtime_profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
