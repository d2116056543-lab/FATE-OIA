from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.datasets.mosaic_multiview import MOSAICWeakMultiView
from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.mosaic_checkpoint import load_mosaic_model_state_strict
from fate_oia.utils.mosaic_artifacts import write_json


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in batch]),
        "reason": torch.stack([row["reason"] for row in batch]),
        "file_name": [row["file_name"] for row in batch],
        "image_path": [row["image_path"] for row in batch],
    }


def _records(index: BDD100KGroundingIndex, file_names: list[str]) -> list[dict[str, Any] | None]:
    records = []
    for file_name in file_names:
        paths = index.lookup(file_name)
        records.append(
            {"label_json": paths.label_json, "drivable_map": paths.drivable_map, "image_size": (720, 1280)}
            if paths.label_json or paths.drivable_map
            else None
        )
    return records


def _weak_factor_score(output: dict[str, torch.Tensor], observations: dict[str, torch.Tensor]) -> dict[str, float]:
    reliability = observations["source_reliability"].clamp(0.0, 1.0)
    presence_weight = observations["presence_mask"] * reliability
    visibility_weight = observations["visibility_mask"] * reliability
    presence = (
        output["factor_presence_prob"] * observations["presence_target"] * presence_weight
    ).sum() / presence_weight.sum().clamp_min(1e-6)
    visibility = (
        output["factor_visibility_prob"] * observations["visibility_target"] * visibility_weight
    ).sum() / visibility_weight.sum().clamp_min(1e-6)
    valid = observations["geometry_mask_valid"] > 0
    if valid.any():
        predicted = output["factor_soft_masks"][valid].clamp(0.0, 1.0)
        target = observations["geometry_mask"][valid]
        intersection = (predicted * target).sum((-2, -1))
        union = (predicted + target - predicted * target).sum((-2, -1)).clamp_min(1e-6)
        geometry_iou = (intersection / union).mean()
        score = 0.375 * presence + 0.375 * visibility + 0.25 * geometry_iou
    else:
        geometry_iou = presence.new_zeros(())
        score = 0.5 * presence + 0.5 * visibility
    return {
        "score": float(score.detach().cpu()),
        "presence": float(presence.detach().cpu()),
        "visibility": float(visibility.detach().cpu()),
        "geometry_iou": float(geometry_iou.detach().cpu()),
        "geometry_valid_count": int(valid.sum().detach().cpu()),
    }


def _permute_factor_output(output: dict[str, torch.Tensor], permutation: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "factor_presence_prob": output["factor_presence_prob"].index_select(1, permutation),
        "factor_visibility_prob": output["factor_visibility_prob"].index_select(1, permutation),
        "factor_soft_masks": output["factor_soft_masks"].index_select(1, permutation),
    }


def _save_overlay(image_path: str, mask: torch.Tensor, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB").resize((640, 360))
    mask_image = Image.fromarray((mask.clamp(0, 1).cpu().numpy() * 255).astype("uint8")).resize((640, 360))
    red = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.blend(Image.composite(red, image, mask_image), image, 0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


@torch.no_grad()
def run_visual_audit(
    model: MOSAICADModel,
    loader: DataLoader,
    grounding_builder: MOSAICGroundingObservationBuilder,
    grounding_index: BDD100KGroundingIndex,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    model.eval()
    output_dir = Path(output_dir)
    for name in (
        "factor_attention_overlays", "factor_content_only", "factor_prior_only",
        "factor_query_shuffle", "factor_image_shuffle", "left_right_flip", "geometry_alignment",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    score_accumulator = {name: [] for name in ("full", "content_only", "prior_only", "query_shuffle", "image_shuffle")}
    factor_count = len(model.schema_bundle["factors"])
    query_permutation = torch.roll(torch.arange(factor_count, device=device), shifts=1)
    prior_scales = []
    prototype_dominance = []
    prototype_effective_count = []
    dead_prototype_count = []
    flip_consistency = []
    visual_cases = 0
    geometry_cases = 0
    factor_names = [factor["name"] for factor in model.schema_bundle["factors"]]
    flip_helper = MOSAICWeakMultiView(
        factor_names, flip_probability=1.0, brightness_jitter=0.0, contrast_jitter=0.0, seed=1
    )
    flip_metadata = flip_helper(torch.zeros(3, 2, 2))["metadata"][0]
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        reasons = batch["reason"].to(device, non_blocking=True)
        observations = grounding_builder(reasons, _records(grounding_index, batch["file_name"]), split="train")
        mode_outputs = {
            mode: model(images, prior_mode=mode, return_masks=True)
            for mode in ("full", "content_only", "prior_only")
        }
        for mode, mode_output in mode_outputs.items():
            score_accumulator[mode].append(_weak_factor_score(mode_output, observations)["score"])
        full = mode_outputs["full"]
        flipped = model(images.flip(-1), prior_mode="full", return_masks=True)
        restored_flip_masks = torch.stack(
            [flip_helper.invert_factor_masks(value, flip_metadata, factor_dim=0) for value in flipped["factor_soft_masks"]]
        )
        flip_consistency.append(
            float((1.0 - (full["factor_soft_masks"] - restored_flip_masks).abs().mean()).detach().cpu())
        )
        score_accumulator["query_shuffle"].append(
            _weak_factor_score(_permute_factor_output(full, query_permutation), observations)["score"]
        )
        image_permutation = torch.roll(torch.arange(images.shape[0], device=device), shifts=1)
        shuffled = {
            name: full[name].index_select(0, image_permutation)
            for name in ("factor_presence_prob", "factor_visibility_prob", "factor_soft_masks")
        }
        score_accumulator["image_shuffle"].append(_weak_factor_score(shuffled, observations)["score"])
        prior_scales.append(full["prior_scale"].detach().float().cpu())
        dominance = full["measurement_stats"].get("dominant_prototype_rate")
        if dominance is not None:
            prototype_dominance.append(dominance.detach().float().cpu())
        effective = full["measurement_stats"].get("prototype_effective_count")
        if effective is not None:
            prototype_effective_count.append(effective.detach().float().cpu())
        dead = full["measurement_stats"].get("dead_prototype_count")
        if dead is not None:
            dead_prototype_count.append(dead.detach().float().cpu())
        for sample_index in range(images.shape[0]):
            if visual_cases < 16:
                factor_index = int(full["factor_presence_prob"][sample_index].argmax())
                name = model.schema_bundle["factors"][factor_index]["name"]
                case_name = f"{visual_cases:03d}_{name}.png"
                _save_overlay(
                    batch["image_path"][sample_index],
                    full["factor_soft_masks"][sample_index, factor_index],
                    output_dir / "factor_attention_overlays" / case_name,
                )
                _save_overlay(
                    batch["image_path"][sample_index],
                    mode_outputs["content_only"]["factor_soft_masks"][sample_index, factor_index],
                    output_dir / "factor_content_only" / case_name,
                )
                _save_overlay(
                    batch["image_path"][sample_index],
                    mode_outputs["prior_only"]["factor_soft_masks"][sample_index, factor_index],
                    output_dir / "factor_prior_only" / case_name,
                )
                _save_overlay(
                    batch["image_path"][sample_index],
                    full["factor_soft_masks"][sample_index, query_permutation[factor_index]],
                    output_dir / "factor_query_shuffle" / case_name,
                )
                _save_overlay(
                    batch["image_path"][sample_index],
                    shuffled["factor_soft_masks"][sample_index, factor_index],
                    output_dir / "factor_image_shuffle" / case_name,
                )
                _save_overlay(
                    batch["image_path"][sample_index],
                    restored_flip_masks[sample_index, factor_index],
                    output_dir / "left_right_flip" / case_name,
                )
                visual_cases += 1
            valid_factors = torch.nonzero(
                observations["geometry_mask_valid"][sample_index] > 0, as_tuple=False
            ).flatten()
            if valid_factors.numel() and geometry_cases < 16:
                geometry_factor = int(valid_factors[0])
                geometry_name = model.schema_bundle["factors"][geometry_factor]["name"]
                _save_overlay(
                    batch["image_path"][sample_index],
                    observations["geometry_mask"][sample_index, geometry_factor],
                    output_dir / "geometry_alignment" / f"{geometry_cases:03d}_{geometry_name}.png",
                )
                geometry_cases += 1
    means = {name: sum(values) / len(values) if values else 0.0 for name, values in score_accumulator.items()}
    full_score = means["full"]
    summary = {
        "sample_count": len(loader.dataset),
        "full_factor_metric": full_score,
        "content_only_factor_metric": means["content_only"],
        "prior_only_factor_metric": means["prior_only"],
        "query_shuffle_factor_metric": means["query_shuffle"],
        "image_shuffle_factor_metric": means["image_shuffle"],
        "content_only_retention": means["content_only"] / max(full_score, 1e-9),
        "query_shuffle_drop": full_score - means["query_shuffle"],
        "image_shuffle_drop": full_score - means["image_shuffle"],
        "prior_scale_max": float(torch.cat(prior_scales).max()) if prior_scales else 0.0,
        "dominant_prototype_rate_mean": float(torch.cat(prototype_dominance).mean()) if prototype_dominance else 0.0,
        "prototype_effective_count_mean": float(torch.cat(prototype_effective_count).mean()) if prototype_effective_count else 0.0,
        "dead_prototype_count_mean": float(torch.cat(dead_prototype_count).mean()) if dead_prototype_count else 0.0,
        "left_right_flip_consistency": sum(flip_consistency) / len(flip_consistency) if flip_consistency else 0.0,
        "visual_case_count": visual_cases,
        "geometry_case_count": geometry_cases,
        "geometry_is_forward_input": "geometry" in inspect.signature(model.forward).parameters,
        "split": "train_audit_only",
    }
    summary["pass"] = (
        summary["full_factor_metric"] > summary["prior_only_factor_metric"]
        and summary["content_only_retention"] >= 0.70
        and summary["query_shuffle_drop"] > 0
        and summary["image_shuffle_drop"] > 0
        and summary["prior_scale_max"] < 0.19
        and summary["dominant_prototype_rate_mean"] < 0.85
        and summary["prototype_effective_count_mean"] >= 1.5
        and summary["geometry_case_count"] > 0
        and summary["geometry_is_forward_input"] is False
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
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
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_mosaic_model_state_strict(model, checkpoint["model"])
    transform = AspectRatioLetterboxTransform(
        int(config["data"]["image_height"]),
        int(config["data"]["image_width"]),
        patch_size=int(config["data"]["patch_size"]),
    )
    dataset = BDDOIAMultiTaskDataset(
        config["data"]["data_root"], config["data"]["raw_root"], split="train",
        action_dim=4, reason_dim=21, load_image=True, transform=transform,
    )
    dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=args.num_workers > 0 and bool(config["data"].get("persistent_workers", True)),
        prefetch_factor=int(config["data"].get("prefetch_factor", 2)) if args.num_workers > 0 else None,
        collate_fn=_collate,
    )
    grounding_builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    grounding_index = BDD100KGroundingIndex(config["data"]["bdd100k_root"])
    print(json.dumps(run_visual_audit(model, loader, grounding_builder, grounding_index, device, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
