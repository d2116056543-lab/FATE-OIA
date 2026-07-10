from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel
from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead
from fate_oia.threshold_tuning import tune_per_label_thresholds
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.mosaic_artifacts import write_json


def _prefixed_metrics(logits: torch.Tensor, targets: torch.Tensor, prefix: str) -> dict[str, Any]:
    metrics = multilabel_metrics_from_logits(logits, targets, threshold=0.5)
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _collate_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "action": torch.stack([row["action"] for row in batch]),
        "reason": torch.stack([row["reason"] for row in batch]),
        "file_name": [row["file_name"] for row in batch],
        "split": [row["split"] for row in batch],
    }


def _factor_rows(
    sums: dict[str, torch.Tensor],
    sample_count: int,
    factor_names: list[str],
    *,
    epoch: int,
) -> list[dict[str, Any]]:
    denominator = max(sample_count, 1)
    rows: list[dict[str, Any]] = []
    for factor_id, factor_name in enumerate(factor_names):
        rows.append(
            {
                "epoch": epoch,
                "split": "test",
                "factor_id": factor_id,
                "factor_name": factor_name,
                "presence_mean": float(sums["presence"][factor_id] / denominator),
                "visibility_mean": float(sums["visibility"][factor_id] / denominator),
                "positive_evidence_mean": float(sums["positive"][factor_id] / denominator),
                "negative_evidence_mean": float(sums["negative"][factor_id] / denominator),
                "uncertainty_mean": float(sums["uncertainty"][factor_id] / denominator),
                "mask_support_mean": float(sums["mask_support"][factor_id] / denominator),
            }
        )
    return rows


@torch.no_grad()
def evaluate_mosaic(
    model: MOSAICADModel,
    threshold_head: MOSAICGroupThresholdHead,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    threshold_head.eval()
    calibrator_before = {name: value.detach().clone() for name, value in threshold_head.state_dict().items()}
    collection: dict[str, list[torch.Tensor]] = {
        "action_visual": [],
        "action_state": [],
        "action_raw": [],
        "action_deploy": [],
        "reason_latent": [],
        "reason_deploy": [],
        "labels_action": [],
        "labels_reason": [],
    }
    sample_ids: list[str] = []
    factor_sums: dict[str, torch.Tensor] | None = None
    state_probability_sum: torch.Tensor | None = None
    state_uncertainty_sum: torch.Tensor | None = None
    sample_count = 0
    prototype_rows: list[dict[str, Any]] = []
    for batch in loader:
        if "split" in batch:
            split_values = batch["split"]
            if isinstance(split_values, str):
                split_values = [split_values]
            if any(value != "test" for value in split_values):
                raise ValueError("formal MOSAIC evaluator accepts test batches only")
        images = batch["image"].to(device, non_blocking=True)
        action_targets = batch["action"].to(device, non_blocking=True)
        reason_targets = batch["reason"].to(device, non_blocking=True)
        output = model(images, return_masks=True)
        threshold = threshold_head(output["action_logits_raw"], output["reason_logits_latent"])
        values = {
            "action_visual": output["action_logits_visual"],
            "action_state": output["action_logits_state"],
            "action_raw": output["action_logits_raw"],
            "action_deploy": threshold["action_logits_deploy"],
            "reason_latent": output["reason_logits_latent"],
            "reason_deploy": threshold["reason_logits_deploy"],
            "labels_action": action_targets,
            "labels_reason": reason_targets,
        }
        for name, value in values.items():
            collection[name].append(value.detach().float().cpu())
        sample_ids.extend(str(value) for value in batch["file_name"])

        count = images.shape[0]
        sample_count += count
        batch_factor = {
            "presence": output["factor_presence_prob"].detach().float().sum(0).cpu(),
            "visibility": output["factor_visibility_prob"].detach().float().sum(0).cpu(),
            "positive": output["factor_positive_evidence"].detach().float().sum(0).cpu(),
            "negative": output["factor_negative_evidence"].detach().float().sum(0).cpu(),
            "uncertainty": output["factor_uncertainty"].detach().float().sum(0).cpu(),
            "mask_support": (output["factor_soft_masks"].detach().float() > 1e-4).float().mean((-2, -1)).sum(0).cpu(),
        }
        if factor_sums is None:
            factor_sums = batch_factor
        else:
            for name in factor_sums:
                factor_sums[name] += batch_factor[name]
        state_batch = output["decision_state_prob"].detach().float().sum(0).cpu()
        uncertainty_batch = output["decision_state_uncertainty"].detach().float().sum(0).cpu()
        state_probability_sum = state_batch if state_probability_sum is None else state_probability_sum + state_batch
        state_uncertainty_sum = (
            uncertainty_batch if state_uncertainty_sum is None else state_uncertainty_sum + uncertainty_batch
        )
        stats = output.get("measurement_stats", {})
        if stats:
            prototype_rows.append(
                {
                    "epoch": epoch,
                    "split": "test",
                    "batch_size": count,
                    **{name: value for name, value in stats.items()},
                }
            )

    if sample_count == 0:
        raise ValueError("formal MOSAIC evaluation received no test samples")
    tensors = {name: torch.cat(values, dim=0) for name, values in collection.items()}
    action_raw_metrics = _prefixed_metrics(tensors["action_raw"], tensors["labels_action"], "Act")
    action_deploy_metrics = _prefixed_metrics(tensors["action_deploy"], tensors["labels_action"], "Act")
    reason_raw_metrics = _prefixed_metrics(tensors["reason_latent"], tensors["labels_reason"], "Exp")
    reason_deploy_metrics = _prefixed_metrics(tensors["reason_deploy"], tensors["labels_reason"], "Exp")
    action_oracle_thresholds, action_oracle = tune_per_label_thresholds(tensors["action_raw"], tensors["labels_action"])
    reason_oracle_thresholds, reason_oracle = tune_per_label_thresholds(tensors["reason_latent"], tensors["labels_reason"])
    deploy_joint = 0.5 * (action_deploy_metrics["Act_mF1"] + reason_deploy_metrics["Exp_mF1"])
    raw_joint = 0.5 * (action_raw_metrics["Act_mF1"] + reason_raw_metrics["Exp_mF1"])
    summary = {
        "epoch": epoch,
        "split": "test",
        "sample_count": sample_count,
        "raw": {**action_raw_metrics, **reason_raw_metrics, "joint": raw_joint},
        "deploy_fixed": {**action_deploy_metrics, **reason_deploy_metrics, "joint": deploy_joint},
        "test_oracle_diagnostic": {
            "writeback_allowed": False,
            "Act_mF1": action_oracle["mF1"],
            "Act_oF1": action_oracle["oF1"],
            "Exp_mF1": reason_oracle["mF1"],
            "Exp_oF1": reason_oracle["oF1"],
            "action_thresholds": action_oracle_thresholds.tolist(),
            "reason_thresholds": reason_oracle_thresholds.tolist(),
        },
    }
    for name, before in calibrator_before.items():
        if not torch.equal(before, threshold_head.state_dict()[name]):
            raise RuntimeError("test evaluation mutated MOSAIC calibration state")

    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    factor_rows = _factor_rows(factor_sums or {}, sample_count, factor_names, epoch=epoch)
    state_rows = [
        {
            "epoch": epoch,
            "split": "test",
            "state_name": name,
            "probability_mean": float(state_probability_sum[index] / sample_count),
            "uncertainty_mean": float(state_uncertainty_sum[index] / sample_count),
        }
        for index, name in enumerate(model.schema_bundle["states"])
    ]
    action_branches = {
        "visual": _prefixed_metrics(tensors["action_visual"], tensors["labels_action"], "Act"),
        "state": _prefixed_metrics(tensors["action_state"], tensors["labels_action"], "Act"),
        "raw": action_raw_metrics,
        "deploy_fixed": action_deploy_metrics,
    }
    reason_branches = {"latent": reason_raw_metrics, "deploy_fixed": reason_deploy_metrics}
    return {
        "metrics_summary": summary,
        "per_label_metrics": {
            "action": {key: value for key, value in action_deploy_metrics.items() if "per_label" in key},
            "reason": {key: value for key, value in reason_deploy_metrics.items() if "per_label" in key},
        },
        "action_branch_metrics": action_branches,
        "reason_branch_metrics": reason_branches,
        "factor_rows": factor_rows,
        "prototype_rows": prototype_rows or [{"epoch": epoch, "split": "test", "available": False}],
        "state_rows": state_rows,
        "threshold_rows": [
            {
                "epoch": epoch,
                "split": "test",
                "writeback_allowed": False,
                "threshold_prob": torch.sigmoid(threshold_head.compose_theta()),
            }
        ],
        "logits": tensors,
        "sample_ids": sample_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = MOSAICADModel(
        config_root=Path(args.config).parent,
        backbone_arch=str(config["backbone"]["arch"]),
        backbone_patch_size=int(config["backbone"]["patch_size"]),
        selected_layers=tuple(int(value) for value in config["backbone"]["selected_layers"]),
        checkpoint_key=str(config["backbone"]["checkpoint_key"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        decoder_layers=int(config["model"]["decoder_layers"]),
        self_attention_heads=int(config["model"]["self_attention_heads"]),
        highres_topk=int(config["model"]["highres_topk"]),
        midres_topk=int(config["model"]["midres_topk"]),
        anchors_per_factor=int(config["model"]["anchors_per_factor"]),
        typed_attention_heads=int(config["model"]["typed_attention_heads"]),
        point_samples=int(config["model"]["point_samples"]),
        curve_samples=int(config["model"]["curve_samples"]),
        region_samples=int(config["model"]["region_samples"]),
        spatial_prior_scale_init=float(config["model"]["spatial_prior_scale_init"]),
        spatial_prior_scale_max=float(config["model"]["spatial_prior_scale_max"]),
        spatial_prior_dropout=float(config["model"]["spatial_prior_dropout"]),
        content_temperature_init=float(config["model"]["content_temperature_init"]),
        state_residual_cap=float(config["model"]["state_residual_cap"]),
    ).to(args.device)
    threshold = MOSAICGroupThresholdHead(
        tail_reason_indices=config["calibration"]["tail_reason_indices"]
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    threshold.load_state_dict(checkpoint["calibrator"])
    data = config["data"]
    transform = AspectRatioLetterboxTransform(
        data["image_height"], data["image_width"], patch_size=data["patch_size"]
    )
    dataset = BDDOIAMultiTaskDataset(
        data["data_root"], data["raw_root"], split="test", action_dim=4, reason_dim=21,
        load_image=True, transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers_eval", 2)),
        pin_memory=bool(data.get("pin_memory", True)),
        collate_fn=_collate_eval,
    )
    result = evaluate_mosaic(model, threshold, loader, torch.device(args.device), epoch=int(checkpoint["epoch"]))
    write_json(args.output, result["metrics_summary"])


if __name__ == "__main__":
    main()
