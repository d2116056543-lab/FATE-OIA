from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.threshold_tuning import tune_per_label_thresholds


def _metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str) -> dict[str, Any]:
    values = multilabel_metrics_from_logits(logits, labels, threshold=0.5)
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    positive = scores[targets > 0.5]
    negative = scores[targets <= 0.5]
    if not positive.numel() or not negative.numel():
        return float("nan")
    return float(((positive[:, None] > negative[None, :]).float() + 0.5 * (positive[:, None] == negative[None, :]).float()).mean())


def _per_label_ranking(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, float]]:
    probability = torch.sigmoid(logits)
    rows = []
    for label in range(logits.shape[1]):
        metric = multilabel_metrics_from_logits(logits[:, label:label+1], labels[:, label:label+1], threshold=0.5)
        rows.append({"label_id": label, "F1": float(metric["mF1"]), "AP": float(metric["mAP"]), "AUC": _binary_auc(probability[:, label], labels[:, label])})
    return rows


def _flip_counts(base: torch.Tensor, final: torch.Tensor, labels: torch.Tensor) -> list[dict[str, int]]:
    base_pred, final_pred, truth = torch.sigmoid(base) >= 0.5, torch.sigmoid(final) >= 0.5, labels > 0.5
    rows = []
    for label in range(labels.shape[1]):
        b, f, y = base_pred[:, label], final_pred[:, label], truth[:, label]
        rows.append({
            "label_id": label,
            "FP_to_TN": int((b & ~y & ~f).sum()),
            "TP_to_FN": int((b & y & ~f & y).sum()),
            "TN_to_FP": int((~b & ~y & f & ~y).sum()),
            "FN_to_TP": int((~b & y & f).sum()),
        })
    return rows


def _cat(collection: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat(values, dim=0) for name, values in collection.items()}


@torch.no_grad()
def evaluate_icdor(
    model: torch.nn.Module,
    loader: DataLoader | list[dict[str, Any]],
    device: torch.device,
    *,
    epoch: int,
    route_mode: str,
    latent_enabled: bool,
) -> dict[str, Any]:
    """Test-only IC-DOR evaluator with explicit branch isolations."""
    model.eval()
    names: list[str] = []
    collection: dict[str, list[torch.Tensor]] = {
        "action_visual_logits.pt": [], "action_shadow_logits.pt": [], "action_final_logits.pt": [],
        "action_deploy_logits.pt": [], "reason_visual_observed_logits.pt": [], "reason_latent_logits.pt": [],
        "reason_observation_model_prob.pt": [], "reason_observed_logits.pt": [], "reason_deploy_logits.pt": [],
        "action_labels.pt": [], "reason_labels.pt": [], "action_support_logits": [], "action_veto_logits": [],
        "action_factor_off": [], "action_factor_shuffled": [], "action_wrong_target": [],
        "action_equal_mass_random": [],
        "reason_factor_route_off": [], "reason_factor_route_shuffled": [],
        "reason_propensity": [], "reason_observed_mix_gate": [], "reason_escape_weight": [],
        "reason_factor_router_weights": [], "reason_factor_masks": [],
        "factor_negative_evidence": [],
    }
    factor_sums: dict[str, torch.Tensor] | None = None
    route_sums: dict[str, torch.Tensor] | None = None
    prototype_sums: dict[str, torch.Tensor] | None = None
    credibility_values: list[torch.Tensor] = []
    credibility_ema_values: list[torch.Tensor] = []
    route_credibility_values: list[torch.Tensor] = []
    semantic_compatibility_values: list[torch.Tensor] = []
    credibility_measurements: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "cV_prior", "cV_query_shuffle_score", "cV_image_shuffle_score",
            "cV_grounding", "cV_stability", "cV_n_eff",
        )
    }
    fine_transport_rows: list[dict[str, Any]] = []
    route_ownership_rows: list[dict[str, Any]] = []
    fine_transport_diagnostics = getattr(model, "fine_transport_diagnostics", {})
    run_fine_off = bool(getattr(fine_transport_diagnostics, "get", lambda *_: False)("fine_off", False))
    run_coarse_off = bool(getattr(fine_transport_diagnostics, "get", lambda *_: False)("coarse_off", False))
    fine_transport_logits: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"action": [], "reason": []}
        for name, enabled in (("fine_off", run_fine_off), ("coarse_off", run_coarse_off))
        if enabled
    }
    sample_count = 0
    for batch in loader:
        splits = batch["split"] if isinstance(batch["split"], list) else [batch["split"]]
        if any(value != "test" for value in splits):
            raise ValueError("IC-DOR evaluator accepts test batches only")
        images = batch["image"].to(device, non_blocking=True)
        action_labels = batch["action"].to(device, non_blocking=True)
        reason_labels = batch["reason"].to(device, non_blocking=True)
        output = model(images, route_mode=route_mode, latent_enabled=latent_enabled, return_masks=True, reason_route_mode="full", return_diagnostics=True)
        fine_transport_outputs: dict[str, dict[str, Any]] = {}
        if run_fine_off:
            fine_transport_outputs["fine_off"] = model(
                images,
                route_mode=route_mode,
                latent_enabled=latent_enabled,
                return_masks=False,
                reason_route_mode="full",
                factor_mask_mode="coarse",
            )
        if run_coarse_off:
            fine_transport_outputs["coarse_off"] = model(
                images,
                route_mode=route_mode,
                latent_enabled=latent_enabled,
                return_masks=False,
                reason_route_mode="full",
                factor_mask_mode="fine",
            )
        for name, branch_output in fine_transport_outputs.items():
            fine_transport_logits[name]["action"].append(branch_output["action_final_logits"].detach().float().cpu())
            fine_transport_logits[name]["reason"].append(branch_output["reason_observed_logits"].detach().float().cpu())
        output_keys = {
            "action_visual_logits.pt": "action_visual_logits",
            "action_shadow_logits.pt": "action_shadow_logits",
            "action_final_logits.pt": "action_final_logits",
            "action_deploy_logits.pt": "action_logits_deploy",
            "reason_visual_observed_logits.pt": "reason_visual_observed_logits",
            "reason_latent_logits.pt": "reason_logits_latent",
            "reason_observation_model_prob.pt": "reason_observation_prob",
            "reason_observed_logits.pt": "reason_observed_logits",
            "reason_deploy_logits.pt": "reason_logits_deploy",
        }
        for key, output_key in output_keys.items():
            collection[key].append(output[output_key].detach().float().cpu())
        collection["action_labels.pt"].append(action_labels.detach().float().cpu())
        collection["reason_labels.pt"].append(reason_labels.detach().float().cpu())
        propensity = output.get("reason_propensity")
        if not isinstance(propensity, torch.Tensor):
            propensity = torch.full_like(reason_labels, 0.5)
        collection["reason_propensity"].append(propensity.detach().float().cpu())
        factor_count = int(output["factor_presence_prob"].shape[1])
        mix_gate = output.get("reason_observed_mix_gate")
        if not isinstance(mix_gate, torch.Tensor) or mix_gate.shape != reason_labels.shape:
            mix_gate = torch.zeros_like(reason_labels)
        collection["reason_observed_mix_gate"].append(mix_gate.detach().float().cpu())
        escape_weight = output.get("reason_escape_weight")
        if not isinstance(escape_weight, torch.Tensor) or escape_weight.shape != reason_labels.shape:
            escape_weight = torch.zeros_like(reason_labels)
        collection["reason_escape_weight"].append(escape_weight.detach().float().cpu())
        router_weights = output.get("reason_factor_router_weights")
        if not isinstance(router_weights, torch.Tensor) or router_weights.shape != (images.shape[0], 21, factor_count):
            router_weights = torch.zeros(images.shape[0], 21, factor_count, device=images.device)
        collection["reason_factor_router_weights"].append(router_weights.detach().float().cpu())
        reason_factor_masks = output.get("reason_factor_masks")
        if not isinstance(reason_factor_masks, torch.Tensor) or reason_factor_masks.ndim != 4 or reason_factor_masks.shape[:2] != (images.shape[0], 21):
            reason_factor_masks = torch.zeros(images.shape[0], 21, 1, 1, device=images.device)
        collection["reason_factor_masks"].append(reason_factor_masks.detach().float().cpu())
        collection["factor_negative_evidence"].append(output["factor_negative_evidence"].detach().float().cpu())
        semantic_compatibility = output.get("reason_semantic_compatibility_effective")
        if not isinstance(semantic_compatibility, torch.Tensor) or semantic_compatibility.shape != (21, factor_count):
            semantic_compatibility = torch.zeros(21, factor_count, device=images.device)
        semantic_compatibility_values.append(semantic_compatibility.detach().float().cpu())
        collection["action_support_logits"].append(output["action_support_logits"].detach().float().cpu())
        collection["action_veto_logits"].append(output["action_veto_logits"].detach().float().cpu())
        for collection_key, output_key in (
            ("action_factor_off", "action_factor_off_logits"),
            ("action_factor_shuffled", "action_factor_shuffled_logits"),
            ("action_wrong_target", "action_wrong_target_logits"),
            ("action_equal_mass_random", "action_equal_mass_random_logits"),
        ):
            fallback = output["action_visual_logits"] if collection_key == "action_factor_off" else output["action_shadow_logits"]
            collection[collection_key].append(output.get(output_key, fallback).detach().float().cpu())
        if "reason_observed_logits_route_off" in output and "reason_observed_logits_route_shuffled" in output:
            off_logits = output["reason_observed_logits_route_off"]
            shuffled_logits = output["reason_observed_logits_route_shuffled"]
        else:
            # Compatibility path for protocol test doubles; the formal model
            # exports both ablations from one DINO pass.
            off_logits = model(
                images,
                route_mode=route_mode,
                latent_enabled=latent_enabled,
                reason_route_mode="off",
                return_masks=False,
            )["reason_observed_logits"]
            shuffled_logits = model(
                images,
                route_mode=route_mode,
                latent_enabled=latent_enabled,
                reason_route_mode="shuffled",
                return_masks=False,
            )["reason_observed_logits"]
        collection["reason_factor_route_off"].append(off_logits.detach().float().cpu())
        collection["reason_factor_route_shuffled"].append(shuffled_logits.detach().float().cpu())
        cV = output.get("cV")
        cV_ema = output.get("cV_ema")
        cV_route_effective = output.get("cV_route_effective")
        if isinstance(cV, torch.Tensor) and isinstance(cV_ema, torch.Tensor):
            credibility_values.append(cV.detach().float().cpu())
            credibility_ema_values.append(cV_ema.detach().float().cpu())
            if isinstance(cV_route_effective, torch.Tensor) and cV_route_effective.shape == cV.shape:
                route_credibility_values.append(cV_route_effective.detach().float().cpu())
            for key in credibility_measurements:
                value = output.get(key)
                if isinstance(value, torch.Tensor) and value.shape == cV.shape:
                    credibility_measurements[key].append(value.detach().float().cpu())
        measurement = output.get("measurement_stats", {})
        fine_transport_rows.append({
            "epoch": epoch,
            "split": "test",
            "fine_mask_delta_mean": float(measurement.get("fine_mask_delta_mean", 0.0)),
            "fine_mask_delta_max": float(measurement.get("fine_mask_delta_max", 0.0)),
            "anchor_separation_mean": float(measurement.get("anchor_separation_mean", 0.0)),
            "sample_attention_support_mean": float(measurement.get("sample_attention_support_mean", 0.0)),
            "typed_coordinates_present": bool(isinstance(output.get("sampling_coordinates"), torch.Tensor)),
            "fine_off_available": run_fine_off,
            "coarse_off_available": run_coarse_off,
            "fine_off_action_delta_abs_mean": (
                float((fine_transport_outputs["fine_off"]["action_final_logits"] - output["action_final_logits"]).abs().mean())
                if "fine_off" in fine_transport_outputs else None
            ),
            # FOUNDATION intentionally keeps final action exactly visual.
            # Fine-coordinate efficacy must therefore be measured on the
            # shadow route, not on the final deployment output.
            "fine_off_action_shadow_delta_abs_mean": (
                float((fine_transport_outputs["fine_off"]["action_shadow_logits"] - output["action_shadow_logits"]).abs().mean())
                if "fine_off" in fine_transport_outputs else None
            ),
            "fine_off_reason_delta_abs_mean": (
                float((fine_transport_outputs["fine_off"]["reason_observed_logits"] - output["reason_observed_logits"]).abs().mean())
                if "fine_off" in fine_transport_outputs else None
            ),
            "fine_off_reason_latent_delta_abs_mean": (
                float((fine_transport_outputs["fine_off"]["reason_logits_latent"] - output["reason_logits_latent"]).abs().mean())
                if "fine_off" in fine_transport_outputs else None
            ),
            "coarse_off_action_delta_abs_mean": (
                float((fine_transport_outputs["coarse_off"]["action_final_logits"] - output["action_final_logits"]).abs().mean())
                if "coarse_off" in fine_transport_outputs else None
            ),
            "coarse_off_action_shadow_delta_abs_mean": (
                float((fine_transport_outputs["coarse_off"]["action_shadow_logits"] - output["action_shadow_logits"]).abs().mean())
                if "coarse_off" in fine_transport_outputs else None
            ),
            "coarse_off_reason_delta_abs_mean": (
                float((fine_transport_outputs["coarse_off"]["reason_observed_logits"] - output["reason_observed_logits"]).abs().mean())
                if "coarse_off" in fine_transport_outputs else None
            ),
            "coarse_off_reason_latent_delta_abs_mean": (
                float((fine_transport_outputs["coarse_off"]["reason_logits_latent"] - output["reason_logits_latent"]).abs().mean())
                if "coarse_off" in fine_transport_outputs else None
            ),
        })
        mix_gate = output.get("reason_observed_mix_gate", 0.0)
        mix_gate_mean = float(mix_gate.mean()) if isinstance(mix_gate, torch.Tensor) else float(mix_gate)
        route_ownership_rows.append({
            "epoch": epoch,
            "split": "test",
            "route_mode": route_mode,
            "latent_enabled": bool(latent_enabled),
            "action_route_gate_cap": float(output.get("action_route_gate_cap", 0.0)),
            "action_shadow_delta_rms": float((output["action_shadow_logits"] - output["action_visual_logits"]).square().mean().sqrt()),
            "action_final_visual_equal": bool(torch.allclose(output["action_final_logits"], output["action_visual_logits"], atol=1e-7, rtol=0.0)),
            "reason_observed_mix_gate_mean": mix_gate_mean,
        })
        count = images.shape[0]
        sample_count += count
        names.extend(str(name) for name in batch["file_name"])
        current_factors = {
            "presence": output["factor_presence_prob"].detach().float().sum(0).cpu(),
            "visibility": output["factor_visibility_prob"].detach().float().sum(0).cpu(),
            "positive": output["factor_positive_evidence"].detach().float().sum(0).cpu(),
            "negative": output["factor_negative_evidence"].detach().float().sum(0).cpu(),
            "uncertainty": output["factor_uncertainty"].detach().float().sum(0).cpu(),
        }
        current_routes = {
            "support": output["support_weights"].detach().float().sum(0).cpu(),
            "veto": output["veto_weights"].detach().float().sum(0).cpu(),
            "support_dustbin": output["support_dustbin"].detach().float().sum(0).cpu(),
            "veto_dustbin": output["veto_dustbin"].detach().float().sum(0).cpu(),
        }
        stats = output.get("measurement_stats", output.get("prototype_stats", {}))
        current_prototypes: dict[str, torch.Tensor] = {}
        if isinstance(stats, dict):
            for key in ("prototype_effective_count", "dominant_prototype_rate", "dead_prototype_count"):
                value = stats.get(key)
                if isinstance(value, torch.Tensor) and value.ndim == 1:
                    current_prototypes[key] = value.detach().float().cpu() * count
        factor_sums = current_factors if factor_sums is None else {key: factor_sums[key] + value for key, value in current_factors.items()}
        route_sums = current_routes if route_sums is None else {key: route_sums[key] + value for key, value in current_routes.items()}
        if current_prototypes:
            prototype_sums = current_prototypes if prototype_sums is None else {
                key: prototype_sums[key] + value for key, value in current_prototypes.items()
            }
    if not sample_count:
        raise ValueError("IC-DOR evaluator received no test samples")
    tensors = _cat(collection)
    action_labels = tensors["action_labels.pt"]
    reason_labels = tensors["reason_labels.pt"]
    action_visual = tensors["action_visual_logits.pt"]
    # Keep the component deltas for the per-action route-effect artifact as
    # well as the isolated branch metrics below.
    action_support_logits = tensors.pop("action_support_logits")
    action_veto_logits = tensors.pop("action_veto_logits")
    action_support_only = action_visual + action_support_logits
    action_veto_only = action_visual - action_veto_logits
    action_factor_off = tensors.pop("action_factor_off")
    action_factor_shuffled = tensors.pop("action_factor_shuffled")
    action_wrong_target = tensors.pop("action_wrong_target")
    action_equal_mass_random = tensors.pop("action_equal_mass_random")
    tensors["action_factor_off_logits.pt"] = action_factor_off
    tensors["action_factor_shuffled_logits.pt"] = action_factor_shuffled
    tensors["action_wrong_target_logits.pt"] = action_wrong_target
    tensors["action_equal_mass_random_logits.pt"] = action_equal_mass_random
    action_branches = {
        "visual": _metrics(action_visual, action_labels, "Act"),
        "shadow": _metrics(tensors["action_shadow_logits.pt"], action_labels, "Act"),
        "final": _metrics(tensors["action_final_logits.pt"], action_labels, "Act"),
        "threshold_off": _metrics(tensors["action_final_logits.pt"], action_labels, "Act"),
        "deploy": _metrics(tensors["action_deploy_logits.pt"], action_labels, "Act"),
        "support_only": _metrics(action_support_only, action_labels, "Act"),
        "veto_only": _metrics(action_veto_only, action_labels, "Act"),
        "factor_off": _metrics(action_factor_off, action_labels, "Act"),
        "factor_shuffled": _metrics(action_factor_shuffled, action_labels, "Act"),
        "wrong_target": _metrics(action_wrong_target, action_labels, "Act"),
        "equal_mass_random": _metrics(action_equal_mass_random, action_labels, "Act"),
    }
    reason_route_off_logits = tensors.pop("reason_factor_route_off")
    reason_route_shuffled_logits = tensors.pop("reason_factor_route_shuffled")
    tensors["reason_factor_route_off_logits.pt"] = reason_route_off_logits
    tensors["reason_factor_route_shuffled_logits.pt"] = reason_route_shuffled_logits
    reason_branches = {
        "visual_observed": _metrics(tensors["reason_visual_observed_logits.pt"], reason_labels, "Exp"),
        "latent_semantic": _metrics(tensors["reason_latent_logits.pt"], reason_labels, "Exp"),
        "observation_model": _metrics(torch.logit(tensors["reason_observation_model_prob.pt"].clamp(1e-6, 1 - 1e-6)), reason_labels, "Exp"),
        "final_observed": _metrics(tensors["reason_observed_logits.pt"], reason_labels, "Exp"),
        "factor_route_off": _metrics(reason_route_off_logits, reason_labels, "Exp"),
        "factor_route_shuffled": _metrics(reason_route_shuffled_logits, reason_labels, "Exp"),
        "threshold_off": _metrics(tensors["reason_observed_logits.pt"], reason_labels, "Exp"),
        "deploy": _metrics(tensors["reason_deploy_logits.pt"], reason_labels, "Exp"),
    }
    fine_transport_branches = {
        name: {
            "action": _metrics(torch.cat(values["action"], dim=0), action_labels, "Act"),
            "reason": _metrics(torch.cat(values["reason"], dim=0), reason_labels, "Exp"),
        }
        for name, values in fine_transport_logits.items()
    }
    action_oracle_thresholds, action_oracle = tune_per_label_thresholds(tensors["action_final_logits.pt"], action_labels)
    reason_oracle_thresholds, reason_oracle = tune_per_label_thresholds(tensors["reason_observed_logits.pt"], reason_labels)
    metrics_summary = {
        "epoch": epoch, "split": "test", "sample_count": sample_count,
        "raw": {**action_branches["final"], **reason_branches["final_observed"], "joint": 0.5 * (action_branches["final"]["Act_mF1"] + reason_branches["final_observed"]["Exp_mF1"])},
        "deploy_fixed": {**action_branches["deploy"], **reason_branches["deploy"], "joint": 0.5 * (action_branches["deploy"]["Act_mF1"] + reason_branches["deploy"]["Exp_mF1"])},
        "test_oracle_diagnostic": {"writeback_allowed": False, "Act_mF1": action_oracle["mF1"], "Exp_mF1": reason_oracle["mF1"], "action_thresholds": action_oracle_thresholds.tolist(), "reason_thresholds": reason_oracle_thresholds.tolist()},
    }
    factor_rows = [{"epoch": epoch, "split": "test", "factor_id": factor_id, **{key: float(value[factor_id] / sample_count) for key, value in (factor_sums or {}).items()}} for factor_id in range((factor_sums or {"presence": torch.empty(0)})["presence"].numel())]
    route_rows = [
        {
            "epoch": epoch, "split": "test", "factor_id": factor_id, "action_id": action_id,
            "support_mass": float((route_sums or {})["support"][factor_id, action_id] / sample_count),
            "veto_mass": float((route_sums or {})["veto"][factor_id, action_id] / sample_count),
            "support_dustbin": float((route_sums or {})["support_dustbin"][action_id] / sample_count),
            "veto_dustbin": float((route_sums or {})["veto_dustbin"][action_id] / sample_count),
        }
        for factor_id in range((route_sums or {"support": torch.empty(0, 0)})["support"].shape[0])
        for action_id in range((route_sums or {"support": torch.empty(0, 0)})["support"].shape[1])
    ]
    route_delta = tensors["action_shadow_logits.pt"] - action_visual
    route_credibility_mean = (
        float(torch.cat(route_credibility_values, dim=0).mean())
        if route_credibility_values else None
    )
    for action_id in range(4):
        visual_rms = action_visual[:, action_id].square().mean().sqrt().clamp_min(1e-8)
        delta = route_delta[:, action_id]
        desired_sign = action_labels[:, action_id] * 2.0 - 1.0
        route_rows.append({
            "epoch": epoch, "split": "test", "factor_id": -1, "action_id": action_id,
            "summary": "per_action_route_effect",
            "route_delta_rms": float(delta.square().mean().sqrt()),
            "visual_logit_rms": float(visual_rms),
            "route_to_visual_rms_ratio": float(delta.square().mean().sqrt() / visual_rms),
            "delta_gt_direction_agreement": float(((delta * desired_sign) > 0).float().mean()),
            "support_delta_rms": float(action_support_logits[:, action_id].square().mean().sqrt()),
            "veto_delta_rms": float(action_veto_logits[:, action_id].square().mean().sqrt()),
            "route_credibility_effective_mean": route_credibility_mean,
            "factor_shuffle_delta_abs_mean": float((tensors["action_shadow_logits.pt"][:, action_id] - action_factor_shuffled[:, action_id]).abs().mean()),
            "wrong_target_delta_abs_mean": float((tensors["action_shadow_logits.pt"][:, action_id] - action_wrong_target[:, action_id]).abs().mean()),
            "selected_minus_equal_random_abs_mean": float((tensors["action_shadow_logits.pt"][:, action_id] - action_equal_mass_random[:, action_id]).abs().mean()),
        })
    prototype_rows = [
        {
            "epoch": epoch, "split": "test", "factor_id": factor_id,
            **{key: float(value[factor_id] / sample_count) for key, value in (prototype_sums or {}).items()},
        }
        for factor_id in range(next(iter((prototype_sums or {"empty": torch.empty(0)}).values())).numel())
    ]
    if credibility_values:
        credibility = torch.cat(credibility_values, dim=0)
        credibility_ema = torch.cat(credibility_ema_values, dim=0)
        credibility_route = torch.cat(route_credibility_values, dim=0) if route_credibility_values else credibility
        credibility_rows = [
            {
                "epoch": epoch,
                "split": "test",
                "factor_id": factor_id,
                "cV_mean": float(credibility[:, factor_id].mean()),
                "cV_p50": float(credibility[:, factor_id].median()),
                "cV_p95": float(torch.quantile(credibility[:, factor_id], 0.95)),
                "cV_ema_mean": float(credibility_ema[:, factor_id].mean()),
                "cV_nonzero_rate": float((credibility[:, factor_id] > 1e-6).float().mean()),
                "cV_route_effective_mean": float(credibility_route[:, factor_id].mean()),
                **{
                    f"{key}_mean": float(torch.cat(values, dim=0)[:, factor_id].mean())
                    for key, values in credibility_measurements.items()
                    if values
                },
            }
            for factor_id in range(credibility.shape[1])
        ]
    else:
        credibility_rows = [{"epoch": epoch, "split": "test", "available": False, "reason": "model_did_not_export_cV"}]
    reason_mix_gate = tensors.pop("reason_observed_mix_gate")
    reason_escape_weight = tensors.pop("reason_escape_weight")
    reason_router_weights = tensors.pop("reason_factor_router_weights")
    reason_factor_masks = tensors.pop("reason_factor_masks")
    factor_negative_evidence = tensors.pop("factor_negative_evidence")
    semantic_compatibility = (
        torch.stack(semantic_compatibility_values).mean(dim=0)
        if semantic_compatibility_values else torch.zeros(21, reason_router_weights.shape[-1])
    )
    latent_decoder = getattr(model, "reason_latent_decoder", None)
    allowed_mask = getattr(latent_decoder, "reason_factor_allow_mask", None)
    if not isinstance(allowed_mask, torch.Tensor) or allowed_mask.shape != semantic_compatibility.shape:
        allowed_mask = torch.ones_like(semantic_compatibility, dtype=torch.bool)
    else:
        allowed_mask = allowed_mask.detach().cpu().to(torch.bool)
    absence_mask = getattr(latent_decoder, "reason_factor_absence_mask", None)
    if not isinstance(absence_mask, torch.Tensor) or absence_mask.shape != semantic_compatibility.shape:
        absence_mask = torch.zeros_like(semantic_compatibility, dtype=torch.bool)
    else:
        absence_mask = absence_mask.detach().cpu().to(torch.bool)
    mask_flat = reason_factor_masks.flatten(2).clamp_min(0.0)
    mask_probability = mask_flat / mask_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mask_entropy = -(mask_probability * mask_probability.clamp_min(1e-8).log()).sum(dim=-1)
    allowed = allowed_mask.unsqueeze(0).to(reason_router_weights)
    reason_rows = [
        {
            "epoch": epoch, "split": "test", "reason_id": reason_id,
            "observed_positive_rate": float(reason_labels[:, reason_id].mean()),
            "latent_probability_mean": float(torch.sigmoid(tensors["reason_latent_logits.pt"][:, reason_id]).mean()),
            "observation_probability_mean": float(tensors["reason_observation_model_prob.pt"][:, reason_id].mean()),
            "final_probability_mean": float(torch.sigmoid(tensors["reason_observed_logits.pt"][:, reason_id]).mean()),
            "factor_route_effect_abs_mean": float((tensors["reason_observed_logits.pt"][:, reason_id] - reason_route_off_logits[:, reason_id]).abs().mean()),
            "factor_shuffle_effect_abs_mean": float((tensors["reason_observed_logits.pt"][:, reason_id] - reason_route_shuffled_logits[:, reason_id]).abs().mean()),
            "residual_alpha_mean": float(reason_mix_gate[:, reason_id].mean()),
            "escape_weight_mean": float(reason_escape_weight[:, reason_id].mean()),
            "allowed_factor_mass_mean": float((reason_router_weights[:, reason_id] * allowed[:, reason_id]).sum(dim=-1).mean()),
            "disallowed_factor_mass_mean": float((reason_router_weights[:, reason_id] * (1.0 - allowed[:, reason_id])).sum(dim=-1).mean()),
            "reason_factor_mask_area_mean": float((reason_factor_masks[:, reason_id] > 1e-6).float().mean()),
            "reason_factor_mask_entropy": float(mask_entropy[:, reason_id].mean()),
            "semantic_compatibility_mean": float((reason_router_weights[:, reason_id] * semantic_compatibility[reason_id].unsqueeze(0)).sum(dim=-1).mean()),
            "absence_factor_mass_mean": float((reason_router_weights[:, reason_id] * absence_mask[reason_id].unsqueeze(0)).sum(dim=-1).mean()),
            "absence_negative_evidence_mean": float(
                (factor_negative_evidence * absence_mask[reason_id].unsqueeze(0)).sum(dim=-1).mean()
                / max(int(absence_mask[reason_id].sum()), 1)
            ),
            "propensity_mean": float(tensors["reason_propensity"][:, reason_id].mean()),
            "propensity_min": float(tensors["reason_propensity"][:, reason_id].min()),
            "propensity_max": float(tensors["reason_propensity"][:, reason_id].max()),
            "propensity_bound_saturation_rate": float(((tensors["reason_propensity"][:, reason_id] <= 0.201) | (tensors["reason_propensity"][:, reason_id] >= 0.949)).float().mean()),
            # Test labels are only used for reporting metrics.  Synthetic
            # hidden-positive recovery needs observed targets and therefore is
            # isolated to audit_target, never constructed inside test eval.
            "posterior_q_observed_zero_available": False,
            "synthetic_hidden_positive_auprc_available": False,
            "top_q_observed_zero_manual_precision_available": False,
            "top_q_observed_zero_cases": [],
        }
        for reason_id in range(reason_labels.shape[1])
    ]
    action_error = (torch.sigmoid(tensors["action_deploy_logits.pt"]) - action_labels).abs().mean(1)
    reason_error = (torch.sigmoid(tensors["reason_deploy_logits.pt"]) - reason_labels).abs().mean(1)
    hardest = torch.argsort(action_error + reason_error, descending=True)[: min(100, sample_count)]
    failure_rows = [
        {
            "epoch": epoch, "split": "test", "file_name": names[int(index)],
            "action_absolute_error_mean": float(action_error[index]),
            "reason_absolute_error_mean": float(reason_error[index]),
        }
        for index in hardest
    ]
    per_label = {
        "action": _per_label_ranking(tensors["action_deploy_logits.pt"], action_labels),
        "reason": _per_label_ranking(tensors["reason_deploy_logits.pt"], reason_labels),
        "action_visual_to_final_flips": _flip_counts(action_visual, tensors["action_final_logits.pt"], action_labels),
    }
    return {
        "metrics_summary": metrics_summary,
        "branch_metrics": {
            "available": True,
            "action": action_branches,
            "reason": reason_branches,
            "fine_transport": fine_transport_branches,
        },
        "per_label_metrics": per_label,
        "factor_rows": factor_rows,
        "prototype_rows": prototype_rows,
        "credibility_rows": credibility_rows,
        "fine_transport_rows": fine_transport_rows,
        "route_ownership_rows": route_ownership_rows,
        "route_rows": route_rows,
        "reason_rows": reason_rows,
        "failure_rows": failure_rows,
        "logits": {key: value for key, value in tensors.items() if key.endswith(".pt")},
        "file_names": names,
    }
