from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.meter_dataset import METERDataset
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.meter_artifacts import load_checkpoint, write_json
from fate_oia.utils.meter_config import load_meter_config
from fate_oia.utils.meter_posthoc_calibration import METERCalibrationResult, apply_meter_deploy


ACTION_BRANCHES = {
    "calalign_visual": (),
    "visual": (),
    "semantic": (),
    "peer_candidate": (),
    "peer": (),
    "final": (),
    "factor_off": ("factor_off",),
    "factor_shuffle": ("factor_shuffle",),
    "factor_shuffled": ("factor_shuffle",),
    "support_only": ("support_only",),
    "counter_only": ("counter_only",),
    "meta_off": ("meta_off",),
    "selector_visual_only": ("selector_visual_only",),
    "selector_semantic_only": ("selector_semantic_only",),
}
REASON_BRANCHES = {
    "calalign_reason": (),
    "calalign": (),
    "global_private": (),
    "global": (),
    "local_private": (),
    "local": (),
    "mix_private": (),
    "mix": (),
    "final": (),
    "annotation_off": ("annotation_off",),
    "annotation_residual_off": ("annotation_off",),
    "factor_context_off": ("factor_context_off",),
    "map_shuffle": ("map_shuffle",),
    "factor_map_shuffled": ("map_shuffle",),
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
                "calalign_visual": "action_logits_visual",
                "visual": "action_logits_visual",
                "semantic": "action_logits_semantic",
                "peer_candidate": "action_logits_peer",
                "peer": "action_logits_peer",
                "final": "action_logits_final",
            }.get(name, "action_logits_final")
            all_action[name].append(output[key].detach().cpu())
            if name == "final":
                for key in ("factor_support_map", "factor_counter_map", "factor_support_null", "factor_counter_null", "factor_support_score", "factor_counter_score", "factor_layer_weights", "factor_reliability", "action_selector", "action_factor_contributions", "semantic_bias", "action_logits_visual", "action_logits_semantic"):
                    mechanism_values.setdefault(key, []).append(output[key].detach().cpu())
        for name, modes in REASON_BRANCHES.items():
            output = model.decode_from_field(field, progress=progress, diagnostic_modes=modes)
            key = {
                "calalign_reason": "reason_logits_calalign",
                "calalign": "reason_logits_calalign",
                "global_private": "reason_logits_global",
                "global": "reason_logits_global",
                "local_private": "reason_logits_local",
                "local": "reason_logits_local",
                "mix_private": "reason_logits_mix",
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
            temperature=None if calibration.temperature is None else calibration.temperature[: action.shape[1]],
            strategy=calibration.strategy,
            model_state_hash_before=calibration.model_state_hash_before,
            model_state_hash_after=calibration.model_state_hash_after,
            fit_split=calibration.fit_split,
            representation_updated=calibration.representation_updated,
        )
        reason_calibration = METERCalibrationResult(
            theta=calibration.theta[action.shape[1] : action.shape[1] + reason.shape[1]],
            temperature=None if calibration.temperature is None else calibration.temperature[action.shape[1] : action.shape[1] + reason.shape[1]],
            strategy=calibration.strategy,
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
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--use_mock_dino", action="store_true")
    args = parser.parse_args()
    config = load_meter_config(args.config)
    device = torch.device(args.device)
    model = METEROIAModel(
        dim=config["model"]["dim"],
        action_dim=config["model"]["action_dim"],
        reason_dim=config["model"]["reason_dim"],
        selected_layers=tuple(config["backbone"]["selected_layers"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=args.use_mock_dino,
        factor_rank=config["model"].get("factor_rank", 16),
    ).to(device)
    payload = load_checkpoint(args.checkpoint, model=model)
    dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="test",
        transform=meter_image_transform(),
        grounding_index=None,
        include_grounding=False,
    )
    indices = list(range(len(dataset)))
    if args.max_test_samples:
        indices = indices[: args.max_test_samples]
    workers = int(config["data"].get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(config["training"].get("batch_size", 6)),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(config["data"].get("pin_memory", True)),
        "persistent_workers": workers > 0 and bool(config["data"].get("persistent_workers", True)),
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)
    calibration_payload = payload.get("calibration", {})
    calibration = None
    if calibration_payload.get("theta") is not None:
        calibration = METERCalibrationResult(
            theta=torch.as_tensor(calibration_payload["theta"]),
            temperature=(
                None
                if calibration_payload.get("temperature") is None
                else torch.as_tensor(calibration_payload["temperature"])
            ),
            strategy=str(calibration_payload.get("strategy", "per_label")),
            model_state_hash_before="checkpoint",
            model_state_hash_after="checkpoint",
            fit_split="train_calib",
            representation_updated=False,
        )
    collected = collect_outputs(model, loader, device, progress=1.0)
    result = {
        "summary": metrics_summary(collected, calibration),
        "branches": branch_metrics(collected),
        "mechanism": mechanism_stats_from_collected(collected),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_samples": len(indices),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
