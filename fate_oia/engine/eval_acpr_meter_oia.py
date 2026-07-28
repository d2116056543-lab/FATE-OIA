from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.utils.meter_posthoc_calibration import METERCalibrationResult, apply_meter_deploy


ACTION_BRANCHES = {
    "visual": (),
    "semantic": (),
    "peer": (),
    "final": (),
    "factor_off": ("factor_off",),
    "factor_shuffle": ("factor_shuffle",),
    "support_only": ("support_only",),
    "counter_only": ("counter_only",),
    "meta_off": ("meta_off",),
}
REASON_BRANCHES = {
    "calalign": (),
    "global": (),
    "local": (),
    "mix": (),
    "final": (),
    "annotation_off": ("annotation_off",),
    "factor_context_off": ("factor_context_off",),
    "map_shuffle": ("map_shuffle",),
    "decision_context_off": ("decision_context_off",),
    "meta_off": ("meta_off",),
}


def _cat(values: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    return torch.cat(values, dim=dim) if values else torch.empty(0)


def _entropy(probability: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(dim=dim)


def _mechanism_stats(mechanism: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Reduce real forward tensors into auditable, non-placeholder statistics."""
    support = mechanism["factor_support_map"]
    counter = mechanism["factor_counter_map"]
    support_null = mechanism["factor_support_null"]
    counter_null = mechanism["factor_counter_null"]
    support_full = torch.cat([support, support_null.unsqueeze(-1)], dim=-1)
    counter_full = torch.cat([counter, counter_null.unsqueeze(-1)], dim=-1)
    layer = mechanism["factor_layer_weights"]
    selector = mechanism["action_selector"]
    visual = mechanism["action_logits_visual"]
    semantic = mechanism["action_logits_semantic"]
    contributions = mechanism["action_factor_contributions"]
    factor_reliability = mechanism["factor_reliability"]
    mix_gate = mechanism["reason_mix_gate"]
    annotation = mechanism["reason_annotation_delta"]
    return {
        "factor_count": int(support.shape[1]),
        "patch_count": int(support.shape[-1]),
        "support_map_sum_mean": float(support.sum(-1).mean().item()),
        "counter_map_sum_mean": float(counter.sum(-1).mean().item()),
        "support_null_mean": float(support_null.mean().item()),
        "counter_null_mean": float(counter_null.mean().item()),
        "support_entropy_mean": float(_entropy(support_full).mean().item()),
        "counter_entropy_mean": float(_entropy(counter_full).mean().item()),
        "support_counter_cosine_mean": float(torch.nn.functional.cosine_similarity(support, counter, dim=-1).mean().item()),
        "reliability_mean": float(factor_reliability.mean().item()),
        "reliability_min": float(factor_reliability.min().item()),
        "reliability_max": float(factor_reliability.max().item()),
        "per_factor_support_null": support_null.mean(0).tolist(),
        "per_factor_counter_null": counter_null.mean(0).tolist(),
        "per_factor_support_entropy": _entropy(support_full).mean(0).tolist(),
        "per_factor_counter_entropy": _entropy(counter_full).mean(0).tolist(),
        "layer_weights": layer.detach().cpu().tolist(),
        "layer_entropy": _entropy(layer).mean().item(),
        "selector_mean": float(selector.mean().item()),
        "selector_std": float(selector.std(unbiased=False).item()),
        "selector_min": float(selector.min().item()),
        "selector_max": float(selector.max().item()),
        "semantic_visual_rms_ratio": float((semantic.float().square().mean().sqrt() / (visual.float().square().mean().sqrt() + 1e-6)).item()),
        "semantic_contribution_rms": float(contributions.float().square().mean().sqrt().item()),
        "visual_logit_rms": float(visual.float().square().mean().sqrt().item()),
        "factor_contribution_sum_error": float((semantic - (mechanism["semantic_bias"] + contributions.sum(-1))).abs().max().item()) if "semantic_bias" in mechanism else None,
        "reason_mix_gate_mean": float(mix_gate.mean().item()),
        "reason_mix_gate_std": float(mix_gate.std(unbiased=False).item()),
        "reason_annotation_rms": float(annotation.float().square().mean().sqrt().item()),
        "reason_global_local_rms": float((mechanism["reason_logits_global"] - mechanism["reason_logits_local"]).float().square().mean().sqrt().item()),
    }


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device, *, progress: float, max_batches: int | None = None) -> dict[str, Any]:
    model.eval()
    all_action: dict[str, list[torch.Tensor]] = {name: [] for name in ACTION_BRANCHES}
    all_reason: dict[str, list[torch.Tensor]] = {name: [] for name in REASON_BRANCHES}
    labels_action: list[torch.Tensor] = []
    labels_reason: list[torch.Tensor] = []
    file_names: list[str] = []
    mechanism_values: dict[str, list[torch.Tensor]] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        field = model.encode_images(images)
        for name, modes in ACTION_BRANCHES.items():
            output = model.decode_from_field(field, progress=progress, diagnostic_modes=modes)
            key = {
                "visual": "action_logits_visual",
                "semantic": "action_logits_semantic",
                "peer": "action_logits_peer",
                "final": "action_logits_final",
            }.get(name, "action_logits_final")
            all_action[name].append(output[key].detach().cpu())
            if name == "final":
                for key in ("factor_support_map", "factor_counter_map", "factor_support_null", "factor_counter_null", "factor_layer_weights", "factor_reliability", "action_selector", "action_factor_contributions", "semantic_bias", "action_logits_visual", "action_logits_semantic"):
                    mechanism_values.setdefault(key, []).append(output[key].detach().cpu())
        for name, modes in REASON_BRANCHES.items():
            output = model.decode_from_field(field, progress=progress, diagnostic_modes=modes)
            key = {
                "calalign": "reason_logits_calalign",
                "global": "reason_logits_global",
                "local": "reason_logits_local",
                "mix": "reason_logits_mix",
                "final": "reason_logits_final",
            }.get(name, "reason_logits_final")
            all_reason[name].append(output[key].detach().cpu())
            if name == "final":
                for key in ("reason_mix_gate", "reason_annotation_delta", "reason_logits_global", "reason_logits_local"):
                    mechanism_values.setdefault(key, []).append(output[key].detach().cpu())
        labels_action.append(batch["action"].detach().cpu())
        labels_reason.append(batch["reason"].detach().cpu())
        file_names.extend([str(x) for x in batch["file_name"]])
    return {
        "action": {name: _cat(values) for name, values in all_action.items()},
        "reason": {name: _cat(values) for name, values in all_reason.items()},
        "labels_action": _cat(labels_action),
        "labels_reason": _cat(labels_reason),
        "file_names": file_names,
        "mechanism": {key: _cat(values) for key, values in mechanism_values.items()},
    }


def branch_metrics(collected: dict[str, Any]) -> dict[str, Any]:
    action_labels = collected["labels_action"]
    reason_labels = collected["labels_reason"]
    result: dict[str, Any] = {}
    for name, logits in collected["action"].items():
        result[f"action_{name}"] = multilabel_metrics_from_logits(logits, action_labels, prefix="Act_")
    for name, logits in collected["reason"].items():
        result[f"reason_{name}"] = multilabel_metrics_from_logits(logits, reason_labels, prefix="Exp_")
    return result


def metrics_summary(collected: dict[str, Any], calibration: METERCalibrationResult | None = None) -> dict[str, Any]:
    action = collected["action"]["final"]
    reason = collected["reason"]["final"]
    raw = {
        **multilabel_metrics_from_logits(action, collected["labels_action"], prefix="Act_"),
        **multilabel_metrics_from_logits(reason, collected["labels_reason"], prefix="Exp_"),
    }
    deploy = dict(raw)
    if calibration is not None:
        action_calibration = METERCalibrationResult(
            theta=calibration.theta[: action.shape[1]],
            model_state_hash_before=calibration.model_state_hash_before,
            model_state_hash_after=calibration.model_state_hash_after,
            fit_split=calibration.fit_split,
            representation_updated=calibration.representation_updated,
        )
        reason_calibration = METERCalibrationResult(
            theta=calibration.theta[action.shape[1] : action.shape[1] + reason.shape[1]],
            model_state_hash_before=calibration.model_state_hash_before,
            model_state_hash_after=calibration.model_state_hash_after,
            fit_split=calibration.fit_split,
            representation_updated=calibration.representation_updated,
        )
        deploy_action = apply_meter_deploy(action, action_calibration)
        deploy_reason = apply_meter_deploy(reason, reason_calibration)
        deploy = {
            **multilabel_metrics_from_logits(deploy_action, collected["labels_action"], prefix="Act_"),
            **multilabel_metrics_from_logits(deploy_reason, collected["labels_reason"], prefix="Exp_"),
        }
    raw["raw_joint"] = 0.5 * (raw.get("Act_mF1", 0.0) + raw.get("Exp_mF1", 0.0))
    deploy["deploy_joint"] = 0.5 * (deploy.get("Act_mF1", 0.0) + deploy.get("Exp_mF1", 0.0))
    return {"metrics_raw": raw, "metrics_deploy": deploy}


def evaluate_checkpoint(model: torch.nn.Module, loader: Iterable[dict[str, Any]], device: torch.device, *, progress: float) -> dict[str, Any]:
    collected = collect_outputs(model, loader, device, progress=progress)
    return {"collected": collected, "branches": branch_metrics(collected), "summary": metrics_summary(collected)}


def mechanism_stats_from_collected(collected: dict[str, Any]) -> dict[str, Any]:
    mechanism = collected.get("mechanism", {})
    if not mechanism:
        return {"factor_count": 0, "patch_count": 0, "available": False}
    return _mechanism_stats(mechanism) | {"available": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.parse_args()
    raise SystemExit("Use train_acpr_meter_oia.py for data-bound evaluation; this entrypoint is intentionally not a hidden training launcher.")


if __name__ == "__main__":
    main()
