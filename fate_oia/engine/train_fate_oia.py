from __future__ import annotations

import ast
import argparse
import json
import math
import re
import socket
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
import vision_transformer as vits
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.datasets.bdd100k_grounding import load_bdd100k_objects
from fate_oia.grounding.losses import attention_grounding_bce, mask_iou, pointing_game_hit
from fate_oia.grounding.mask_builder import drivable_map_to_mask, objects_to_mask
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.token_provenance import keep_merge_tokens, recover_attribution
from fate_oia.transforms import AspectRatioLetterboxTransform, FixedSizeResizeTransform
from fate_oia.utils.lr_scaling import compute_lr_scaling


def build_transform(image_height: int, image_width: int, patch_size: int = 8, preserve_aspect_ratio: bool = True, return_meta: bool = True):
    if preserve_aspect_ratio:
        return AspectRatioLetterboxTransform(image_height=image_height, image_width=image_width, patch_size=patch_size, return_meta=return_meta)
    return FixedSizeResizeTransform(image_height=image_height, image_width=image_width, patch_size=patch_size, return_meta=return_meta)


def build_backbone(args, device: torch.device) -> tuple[nn.Module, int]:
    if args.arch not in vits.__dict__:
        raise ValueError(f"Unsupported SNNA ViT arch: {args.arch}")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, int(model.embed_dim)


@torch.no_grad()
def extract_tokens(backbone: nn.Module, images: torch.Tensor, n_last_blocks: int) -> torch.Tensor:
    layers = backbone.get_intermediate_layers(images, n_last_blocks)
    return layers[-1]


def labels_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([batch["action"].float(), batch["reason"].float()], dim=1)


def limited(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_loader(args, split: str, shuffle: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args.image_height, args.image_width, args.patch_size, args.preserve_aspect_ratio, return_meta=True),
    )
    max_samples = args.max_train_samples if split == "train" else args.max_val_samples
    ds = limited(ds, max_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def make_multilabel_criterion(args) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    return nn.BCEWithLogitsLoss()


def load_config_defaults(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8-sig")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = {}
        current_section = None
        for raw in text.splitlines():
            if not raw.strip() or raw.strip().startswith("#"):
                continue
            if not raw.startswith(" ") and raw.rstrip().endswith(":"):
                current_section = raw.strip()[:-1]
                loaded[current_section] = {}
                continue
            if current_section and ":" in raw:
                key, value = raw.strip().split(":", 1)
                value = value.strip()
                if value.lower() in {"true", "false"}:
                    parsed: Any = value.lower() == "true"
                else:
                    try:
                        parsed = int(value)
                    except ValueError:
                        try:
                            parsed = float(value)
                        except ValueError:
                            parsed = value.strip("\"'")
                loaded[current_section][key.strip()] = parsed
    flat: dict[str, Any] = {}
    if isinstance(loaded, dict):
        for value in loaded.values():
            if isinstance(value, dict):
                flat.update(value)
    return flat


def apply_config_defaults(args, config_defaults: dict[str, Any]) -> None:
    if not config_defaults:
        return
    cli_tokens = set(sys.argv[1:])
    for key, value in config_defaults.items():
        if not hasattr(args, key):
            continue
        if f"--{key}" in cli_tokens:
            continue
        setattr(args, key, value)
def load_grounding_cache(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fn = rec.get("file_name")
            if fn:
                out[str(fn)] = rec
    return out


def load_reason_grounding_rules(path: str, reason_dim: int) -> dict[int, set[str]]:
    """Load reason-index -> BDD100K category mappings.

    The config file is intentionally tiny YAML. PyYAML is used when available,
    with a conservative line parser fallback so training does not depend on a
    new package just for this mapping.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8-sig")
    # Some generated smoke configs may arrive with escaped newlines. Treat
    # those as real YAML line breaks rather than silently disabling grounding.
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    data: dict[str, Any] | None = None
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        data = loaded.get("reason_to_bdd100k_categories", loaded) if isinstance(loaded, dict) else {}
    except Exception:
        data = {}
        in_mapping = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("reason_to_bdd100k_categories"):
                in_mapping = True
                continue
            if not in_mapping:
                continue
            m = re.match(r"^(\d+)\s*:\s*(\[.*\])\s*$", stripped)
            if not m:
                continue
            try:
                data[m.group(1)] = ast.literal_eval(m.group(2))
            except Exception:
                continue
    rules: dict[int, set[str]] = {}
    if not isinstance(data, dict):
        return rules
    for key, value in data.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if idx < 0 or idx >= reason_dim:
            continue
        if isinstance(value, dict):
            value = value.get("target_categories") or value.get("categories") or value.get("bdd100k_categories") or []
        if isinstance(value, str):
            cats = {value}
        else:
            cats = {str(x) for x in value or [] if str(x)}
        if cats:
            rules[idx] = cats
    return rules


def compute_grounding_loss(
    label_attention: torch.Tensor | None,
    batch: dict[str, Any],
    grounding_cache: dict[str, dict[str, Any]],
    args,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if label_attention is None or not grounding_cache:
        return torch.zeros((), device=device), {}
    global_attn_maps = attention_to_patch_map(label_attention, slice(0, args.action_dim + args.reason_dim), args.image_height, args.image_width, args.patch_size)
    losses = []
    global_losses = []
    label_losses = []
    object_losses = []
    lane_losses = []
    drivable_losses = []
    stats: dict[str, float] = {
        "grounding_valid_count": 0.0,
        "grounding_global_count": 0.0,
        "grounding_label_count": 0.0,
        "grounding_object_count": 0.0,
        "grounding_lane_count": 0.0,
        "grounding_drivable_count": 0.0,
        "grounding_skipped_count": 0.0,
    }
    categories = set(x.strip() for x in args.grounding_categories.split(",") if x.strip()) if args.grounding_categories else None
    file_names = batch.get("file_name", [])
    if isinstance(file_names, str):
        file_names = [file_names]
    reasons = batch.get("reason")
    if isinstance(reasons, torch.Tensor):
        reasons = reasons.detach().cpu()
    reason_rules = getattr(args, "reason_grounding_rules_map", {}) or {}
    grounding_mode = getattr(args, "grounding_mode", "both")
    for i, fn in enumerate(file_names):
        rec = grounding_cache.get(str(fn))
        if not rec or not rec.get("label_json"):
            continue
        try:
            objects = load_bdd100k_objects(rec["label_json"])
        except Exception:
            continue
        output_size = global_attn_maps.shape[-2:]
        if grounding_mode in {"global", "both"}:
            try:
                target = objects_to_mask(objects, (args.grounding_image_width, args.grounding_image_height), output_size, categories=categories).to(device)
            except Exception:
                target = None
            if target is not None and float(target.sum().item()) > 0:
                loss_i = attention_grounding_bce(global_attn_maps[i], target)
                losses.append(loss_i)
                global_losses.append(loss_i)
                object_losses.append(loss_i)
                stats["grounding_valid_count"] += 1.0
                stats["grounding_global_count"] += 1.0
                stats["grounding_object_count"] += 1.0
                stats["global_count"] = stats.get("global_count", 0.0) + 1.0
                stats["pointing_game_hit_object_sum"] = stats.get("pointing_game_hit_object_sum", 0.0) + float(pointing_game_hit(global_attn_maps[i].detach(), target.detach()))
            else:
                stats["grounding_skipped_count"] += 1.0
            # Lane/drivable stats are split even if they are not separately
            # weighted yet; this keeps formal runs diagnosable.
            try:
                lane_target = objects_to_mask(
                    objects,
                    (args.grounding_image_width, args.grounding_image_height),
                    output_size,
                    categories=None,
                    include_box2d=False,
                    include_poly2d=True,
                    include_drivable=False,
                    include_lane=True,
                ).to(device)
                if float(lane_target.sum().item()) > 0:
                    loss_lane = attention_grounding_bce(global_attn_maps[i], lane_target)
                    lane_losses.append(loss_lane)
                    stats["grounding_lane_count"] += 1.0
                    stats["lane_iou_sum"] = stats.get("lane_iou_sum", 0.0) + float(mask_iou(global_attn_maps[i].detach(), lane_target.detach()))
            except Exception:
                pass
            try:
                drv_target = objects_to_mask(
                    objects,
                    (args.grounding_image_width, args.grounding_image_height),
                    output_size,
                    categories=None,
                    include_box2d=False,
                    include_poly2d=True,
                    include_drivable=True,
                    include_lane=False,
                ).to(device)
                drv_map = rec.get("drivable_map")
                if drv_map and Path(str(drv_map)).exists():
                    drv_target = torch.maximum(drv_target, drivable_map_to_mask(str(drv_map), output_size).to(device))
                if float(drv_target.sum().item()) > 0:
                    loss_drv = attention_grounding_bce(global_attn_maps[i], drv_target)
                    drivable_losses.append(loss_drv)
                    stats["grounding_drivable_count"] += 1.0
                    stats["drivable_iou_sum"] = stats.get("drivable_iou_sum", 0.0) + float(mask_iou(global_attn_maps[i].detach(), drv_target.detach()))
            except Exception:
                pass
        if grounding_mode in {"label", "both"} and reason_rules and reasons is not None:
            reason_vec = reasons[i] if i < len(reasons) else None
            if reason_vec is None:
                continue
            for reason_idx, reason_categories in reason_rules.items():
                if reason_idx >= len(reason_vec) or float(reason_vec[reason_idx]) <= 0:
                    continue
                label_idx = args.action_dim + int(reason_idx)
                reason_attn = attention_to_patch_map(label_attention, label_idx, args.image_height, args.image_width, args.patch_size)
                try:
                    target = objects_to_mask(objects, (args.grounding_image_width, args.grounding_image_height), reason_attn.shape[-2:], categories=reason_categories).to(device)
                except Exception:
                    continue
                if float(target.sum().item()) <= 0:
                    stats["grounding_skipped_count"] += 1.0
                    continue
                loss_i = attention_grounding_bce(reason_attn[i], target)
                losses.append(loss_i)
                label_losses.append(loss_i)
                key = f"reason_{reason_idx}_count"
                stats[key] = stats.get(key, 0.0) + 1.0
                stats[f"reason_{reason_idx}_loss_sum"] = stats.get(f"reason_{reason_idx}_loss_sum", 0.0) + float(loss_i.detach().item())
                stats["grounding_valid_count"] += 1.0
                stats["grounding_label_count"] += 1.0
    if not losses:
        return torch.zeros((), device=device), stats
    def _mean_value(items: list[torch.Tensor]) -> float:
        return float(torch.stack(items).mean().detach().item()) if items else 0.0

    stats["grounding_global_loss"] = _mean_value(global_losses)
    stats["grounding_label_loss"] = _mean_value(label_losses)
    stats["grounding_object_loss"] = _mean_value(object_losses)
    stats["grounding_lane_loss"] = _mean_value(lane_losses)
    stats["grounding_drivable_loss"] = _mean_value(drivable_losses)
    if stats.get("grounding_object_count", 0.0) > 0:
        stats["pointing_game_hit_object"] = stats.get("pointing_game_hit_object_sum", 0.0) / max(stats["grounding_object_count"], 1.0)
    if stats.get("grounding_lane_count", 0.0) > 0:
        stats["lane_iou"] = stats.get("lane_iou_sum", 0.0) / max(stats["grounding_lane_count"], 1.0)
    if stats.get("grounding_drivable_count", 0.0) > 0:
        stats["drivable_iou"] = stats.get("drivable_iou_sum", 0.0) / max(stats["grounding_drivable_count"], 1.0)
    return torch.stack(losses).mean(), stats



def scheduled_keep_ratio(args, epoch: int) -> float:
    if args.token_compression == "none":
        return 1.0
    if epoch < args.compression_start_epoch:
        return 1.0
    warm = max(int(args.compression_warmup_epochs), 0)
    if warm <= 0:
        return float(args.compression_keep_ratio_final)
    t = min(max(epoch - args.compression_start_epoch, 0), warm) / float(warm)
    return float(args.compression_keep_ratio_start + t * (args.compression_keep_ratio_final - args.compression_keep_ratio_start))


def compress_tokens(tokens: torch.Tensor, keep_ratio: float, num_summary_tokens: int, min_tokens: int, token_compression: str = "keep_merge") -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if token_compression == "none" or keep_ratio >= 0.999:
        return tokens, None, {"enabled": False, "original_tokens": int(tokens.shape[1]), "reduced_tokens": int(tokens.shape[1])}
    cls = tokens[:, :1]
    patch = tokens[:, 1:]
    scores = patch.norm(dim=-1)
    reduced_patch, patch_prov, stats = keep_merge_tokens(patch, scores=scores, keep_ratio=keep_ratio, num_summary_tokens=num_summary_tokens, min_tokens=min_tokens)
    reduced = torch.cat([cls, reduced_patch], dim=1)
    b, n, _ = tokens.shape
    r = reduced.shape[1]
    prov = torch.zeros(b, n, r, device=tokens.device, dtype=tokens.dtype)
    prov[:, 0, 0] = 1.0
    prov[:, 1:, 1:] = patch_prov
    stats = {**stats, "enabled": True, "original_tokens": int(n), "reduced_tokens": int(r)}
    return reduced, prov, stats


def recover_label_attention(attention: torch.Tensor | None, provenance: torch.Tensor | None, original_tokens: int) -> torch.Tensor | None:
    if attention is None:
        return None
    # label-query head returns [B,H,L,N]. Average heads -> [B,L,N].
    if attention.ndim == 4:
        attn = attention.mean(1)
    elif attention.ndim == 3:
        attn = attention
    else:
        raise ValueError(f"Unexpected attention shape {tuple(attention.shape)}")
    if provenance is None:
        return attn
    recovered = recover_attribution(attn.transpose(1, 2), provenance).transpose(1, 2)
    if recovered.shape[-1] != original_tokens:
        raise ValueError("Recovered attention length mismatch")
    return recovered


def attention_to_patch_map(label_attention: torch.Tensor, label_indices: slice | int | list[int], image_height: int, image_width: int, patch_size: int) -> torch.Tensor:
    # label_attention: [B,L,N_original]. Drop CLS and average selected labels.
    if isinstance(label_indices, int):
        patch_scores = label_attention[:, label_indices, 1:]
    else:
        selected = label_attention[:, label_indices, 1:]
        patch_scores = selected.mean(1)
    h = image_height // patch_size
    w = image_width // patch_size
    if patch_scores.shape[1] != h * w:
        side = int(math.sqrt(patch_scores.shape[1]))
        if side * side != patch_scores.shape[1]:
            raise ValueError(f"Cannot reshape {patch_scores.shape[1]} patch tokens into a map")
        h = w = side
    return patch_scores.reshape(patch_scores.shape[0], h, w)


def reason_to_action_consistency_loss(action_logits: torch.Tensor, r2a_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(r2a_logits, torch.sigmoid(action_logits).detach())


def action_branch_losses(
    out: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    loss_action_visual: float = 0.05,
    loss_r2a_gt: float = 0.10,
    loss_action_agree: float = 0.01,
    include_fused_branch_loss: bool = False,
    loss_action_fused_aux: float = 0.0,
) -> dict[str, torch.Tensor]:
    visual = out.get("action_visual_logits", out["action_logits"])
    reason = out.get("action_reason_logits", out.get("reason_to_action_logits", out["action_logits"]))
    fused = out.get("action_fused_logits", out["action_logits"])
    visual_loss = F.binary_cross_entropy_with_logits(visual, action_target)
    reason_loss = F.binary_cross_entropy_with_logits(reason, action_target)
    fused_loss = F.binary_cross_entropy_with_logits(fused, action_target)
    agree = F.mse_loss(torch.sigmoid(visual), torch.sigmoid(reason))
    fused_aux = float(loss_action_fused_aux) * fused_loss if include_fused_branch_loss else fused_loss.new_zeros(())
    total = (
        float(loss_action_visual) * visual_loss
        + float(loss_r2a_gt) * reason_loss
        + float(loss_action_agree) * agree
        + fused_aux
    )
    return {
        "action_total": total,
        "action_branch_total": total,
        "action_visual_loss": visual_loss,
        "action_reason_loss": reason_loss,
        "action_fused_loss_main_only": fused_loss,
        "action_fused_aux_loss": fused_aux,
        "action_agree_loss": agree,
    }


def _fill_masked_tokens(tokens: torch.Tensor, topk: torch.Tensor, mask_fill: str) -> torch.Tensor:
    masked = tokens.clone()
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
    if mask_fill == "zero":
        fill = torch.zeros(tokens.shape[0], topk.shape[1], tokens.shape[2], device=tokens.device, dtype=tokens.dtype)
    elif mask_fill == "mean":
        fill = tokens.mean(dim=1, keepdim=True).expand(-1, topk.shape[1], -1)
    elif mask_fill == "mask_token":
        fill = tokens[:, :1].detach().expand(-1, topk.shape[1], -1)
    else:
        raise ValueError(f"Unsupported counterfactual mask_fill: {mask_fill}")
    masked[batch_idx, topk] = fill
    return masked


def counterfactual_deletion_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    base_loss: torch.Tensor,
    action_dim: int,
    topk_ratio: float = 0.15,
    margin: float = 0.02,
    mask_fill: str = "mean",
) -> torch.Tensor:
    with torch.no_grad():
        out = model(tokens)
        attn = out.get("attention")
        if attn is None:
            return tokens.new_zeros(())
        attn_mean = attn.mean(1).mean(1) if attn.ndim == 4 else attn.mean(1)
        patch_scores = attn_mean[:, 1:] if attn_mean.shape[1] == tokens.shape[1] else attn_mean
        k = max(1, int(round(patch_scores.shape[1] * topk_ratio)))
        topk = torch.topk(patch_scores, k=k, dim=1).indices + 1
    masked = _fill_masked_tokens(tokens, topk, mask_fill)
    masked_out = model(masked)
    masked_logits = torch.cat([masked_out["action_logits"], masked_out["reason_logits"]], dim=1)
    masked_loss = F.binary_cross_entropy_with_logits(masked_logits, labels.float())
    return F.relu(margin + base_loss.detach() - masked_loss)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row)) + "\n")


def _first_image_meta(batch: dict[str, Any]) -> dict[str, Any]:
    meta = batch.get("image_meta") or {}
    if not isinstance(meta, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[0].detach().cpu().tolist() if value.ndim > 0 else value.item()
        elif isinstance(value, (list, tuple)) and value:
            first = value[0]
            out[key] = first.detach().cpu().tolist() if isinstance(first, torch.Tensor) else first
        else:
            out[key] = value
    return out


def build_run_manifest(args, output_dir: Path, train_count: int, val_count: int, *, is_smoke: bool) -> dict[str, Any]:
    return {
        "repo_name": "FATE-OIA",
        "command": " ".join(sys.argv),
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "dataset_root": str(args.data_root),
        "train_split_count": int(train_count),
        "val_split_count": int(val_count),
        "test_split_count": None,
        "config_resolved": vars(args),
        "checkpoint_input": str(args.pretrained_weights),
        "output_dir": str(output_dir),
        "is_smoke": bool(is_smoke),
        "max_train_samples": int(args.max_train_samples),
        "max_val_samples": int(args.max_val_samples),
        "num_gpus": int(getattr(args, "num_gpus", 1)),
        "per_gpu_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(getattr(args, "effective_batch_size", args.batch_size * max(1, args.gradient_accumulation_steps))),
        "reference_effective_batch": int(getattr(args, "reference_effective_batch", 32)),
        "base_lr_at_reference_batch": float(getattr(args, "base_head_lr_at_reference_batch", args.lr)),
        "lr_actual": float(args.lr),
        "backbone_lr": 0.0,
        "optimizer": "AdamW",
        "scheduler": "none",
        "mixed_precision": False,
        "loss_divided_by_accumulation": True,
        "token_compression": {
            "mode": args.token_compression,
            "compression_start_epoch": int(args.compression_start_epoch),
            "compression_warmup_epochs": int(args.compression_warmup_epochs),
            "compression_keep_ratio_start": float(args.compression_keep_ratio_start),
            "compression_keep_ratio_final": float(args.compression_keep_ratio_final),
            "token_score_mode": args.token_score_mode,
        },
        "grounding": {
            "mode": args.grounding_mode,
            "loss_grounding": float(args.loss_grounding),
            "grounding_cache_jsonl": args.grounding_cache_jsonl,
        },
        "counterfactual": {
            "loss_counterfactual": float(args.loss_counterfactual),
            "cf_mask_fill": args.cf_mask_fill,
            "counterfactual_topk_ratio": float(args.counterfactual_topk_ratio),
        },
    }


def _failure_cases(file_names: list[str], logits: torch.Tensor, labels: torch.Tensor, action_dim: int, limit: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if logits.numel() == 0 or labels.numel() == 0:
        return rows
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    for idx in range(min(limit, logits.shape[0])):
        gt_reason = labels[idx, action_dim:]
        pred_reason = pred[idx, action_dim:]
        false_pos = torch.where((pred_reason > 0) & (gt_reason <= 0))[0].tolist()
        false_neg = torch.where((pred_reason <= 0) & (gt_reason > 0))[0].tolist()
        if torch.equal(pred[idx], labels[idx]):
            continue
        rows.append(
            {
                "file_name": file_names[idx] if idx < len(file_names) else str(idx),
                "gt_action": labels[idx, :action_dim].tolist(),
                "pred_action_fused": pred[idx, :action_dim].tolist(),
                "gt_reason_positive": torch.where(gt_reason > 0)[0].tolist(),
                "pred_reason_positive": torch.where(pred_reason > 0)[0].tolist(),
                "top_false_positives": false_pos[:10],
                "top_false_negatives": false_neg[:10],
            }
        )
    return rows


def write_epoch_artifacts(output_dir: Path, epoch: int, train_stats: dict[str, Any], val_stats: dict[str, Any], run_manifest: dict[str, Any]) -> None:
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "epoch": epoch,
        "is_smoke": bool(run_manifest.get("is_smoke", False)),
        "train_loss": train_stats.get("loss", 0.0),
        "val_loss": val_stats.get("loss", 0.0),
        "val_metrics": val_stats.get("metrics", {}),
        "branch_metrics": val_stats.get("branch_metrics", {}),
    }
    fused = val_stats.get("branch_metrics", {}).get("action_fused", {})
    visual = val_stats.get("branch_metrics", {}).get("action_visual", {})
    reason_action = val_stats.get("branch_metrics", {}).get("action_reason", {})
    metrics.update(
        {
            "Act_mF1_visual": visual.get("Act_mF1"),
            "Act_oF1_visual": visual.get("Act_oF1"),
            "Act_mF1_reason_action": reason_action.get("Act_mF1"),
            "Act_oF1_reason_action": reason_action.get("Act_oF1"),
            "Act_mF1_fused": fused.get("Act_mF1", val_stats.get("metrics", {}).get("Act_mF1")),
            "Act_oF1_fused": fused.get("Act_oF1", val_stats.get("metrics", {}).get("Act_oF1")),
            "Exp_mF1": val_stats.get("metrics", {}).get("Exp_mF1"),
            "Exp_oF1": val_stats.get("metrics", {}).get("Exp_oF1"),
            "Exp_mAP": val_stats.get("metrics", {}).get("Exp_mAP"),
            "threshold_mode": val_stats.get("metrics", {}).get("threshold_mode"),
        }
    )
    _write_json(epoch_dir / "metrics_summary.json", metrics)
    _write_json(epoch_dir / "branch_metrics.json", val_stats.get("branch_metrics", {}))
    _write_jsonl(epoch_dir / "loss_components.jsonl", train_stats.get("loss_components", []) + val_stats.get("loss_components", []))
    labels = val_stats.get("labels", torch.empty(0, 0))
    logits = val_stats.get("logits", torch.empty(0, 0))
    config_resolved = run_manifest.get("config_resolved", {}) if isinstance(run_manifest, dict) else {}
    action_dim = int(config_resolved.get("action_dim", 4))
    torch.save(val_stats.get("visual_logits", torch.empty(0, 0)), epoch_dir / "logits_visual_action.pt")
    torch.save(val_stats.get("reason_action_logits", torch.empty(0, 0)), epoch_dir / "logits_reason_action.pt")
    torch.save(val_stats.get("fused_logits", torch.empty(0, 0)), epoch_dir / "logits_fused_action.pt")
    torch.save(logits[:, action_dim:] if logits.numel() else torch.empty(0, 0), epoch_dir / "logits_reason.pt")
    torch.save(labels[:, :action_dim] if labels.numel() else torch.empty(0, action_dim), epoch_dir / "labels_action.pt")
    torch.save(labels[:, action_dim:] if labels.numel() else torch.empty(0, 0), epoch_dir / "labels_reason.pt")
    file_names = val_stats.get("file_names", [])
    _write_json(epoch_dir / "file_names.json", file_names)
    _write_json(epoch_dir / "thresholds_fixed.json", {"threshold": 0.5})
    _write_json(epoch_dir / "thresholds_global.json", {"available": False, "reason": "computed by separate threshold tuner"})
    _write_json(epoch_dir / "thresholds_per_label.json", {"available": False, "reason": "computed by separate threshold tuner"})
    _write_jsonl(epoch_dir / "token_stats.jsonl", train_stats.get("token_stats", []) + val_stats.get("token_stats", []))
    _write_jsonl(epoch_dir / "grounding_stats.jsonl", train_stats.get("grounding_stats", []) + val_stats.get("grounding_stats", []))
    _write_jsonl(epoch_dir / "counterfactual_stats.jsonl", train_stats.get("counterfactual_stats", []) + val_stats.get("counterfactual_stats", []))
    _write_jsonl(epoch_dir / "failure_cases.jsonl", _failure_cases(file_names, logits, labels, action_dim))
    _write_json(epoch_dir / "run_manifest.json", run_manifest)


def run_epoch(args, backbone, model, loader, criterion, optimizer, device, train: bool, grounding_cache: dict[str, dict[str, Any]] | None = None, epoch: int = 0) -> dict[str, Any]:
    model.train(train)
    total_loss = 0.0
    count = 0
    logits_all: list[torch.Tensor] = []
    visual_logits_all: list[torch.Tensor] = []
    reason_action_logits_all: list[torch.Tensor] = []
    fused_logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    stats_rows: list[dict[str, Any]] = []
    loss_components: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    file_names_all: list[str] = []
    accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = labels_from_batch(batch).to(device, non_blocking=True)
        with torch.no_grad():
            original_tokens = extract_tokens(backbone, images, args.n_last_blocks)
        keep_ratio = scheduled_keep_ratio(args, epoch)
        tokens, provenance, token_stats = compress_tokens(original_tokens, keep_ratio, args.num_summary_tokens, args.min_tokens, args.token_compression)
        out = model(tokens)
        logits = torch.cat([out["action_fused_logits"], out["reason_logits"]], dim=1)
        main_loss = criterion(logits, labels)
        branch = action_branch_losses(
            out,
            labels[:, : args.action_dim],
            loss_action_visual=args.loss_action_visual,
            loss_r2a_gt=args.loss_r2a_gt,
            loss_action_agree=args.loss_action_agree,
            include_fused_branch_loss=args.include_fused_branch_loss,
            loss_action_fused_aux=args.loss_action_fused_aux,
        )
        r2a_loss = reason_to_action_consistency_loss(out["action_visual_logits"], out["action_reason_logits"])
        if args.r2a_consistency_mode == "detach_mimic":
            action_extra = args.loss_reason_to_action * r2a_loss
        elif args.r2a_consistency_mode == "none":
            action_extra = logits.new_zeros(())
        else:
            action_extra = branch["action_total"]
        loss = main_loss + action_extra
        cf_loss = original_tokens.new_zeros(())
        cf_stats = {"cf_loss": 0.0, "cf_topk_ratio": float(args.counterfactual_topk_ratio), "cf_mask_fill": args.cf_mask_fill, "cf_valid_count": 0}
        if args.loss_counterfactual > 0:
            cf_loss = counterfactual_deletion_loss(model, tokens, labels, main_loss, args.action_dim, args.counterfactual_topk_ratio, mask_fill=args.cf_mask_fill)
            loss = loss + args.loss_counterfactual * cf_loss
            cf_stats["cf_loss"] = float(cf_loss.detach().item())
            cf_stats["cf_valid_count"] = int(labels.shape[0])
        grounding_loss = original_tokens.new_zeros(())
        if args.loss_grounding > 0 and grounding_cache:
            recovered_attn = recover_label_attention(out.get("attention"), provenance, original_tokens.shape[1])
            grounding_loss, grounding_stats = compute_grounding_loss(recovered_attn, batch, grounding_cache, args, device)
            loss = loss + args.loss_grounding * grounding_loss
        else:
            grounding_stats = {}
        if train:
            (loss / float(accum)).backward()
            is_accum_boundary = ((step + 1) % accum == 0) or ((step + 1) == len(loader))
            if is_accum_boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        count += bs
        logits_all.append(logits.detach().cpu())
        visual_logits_all.append(out["action_visual_logits"].detach().cpu())
        reason_action_logits_all.append(out["action_reason_logits"].detach().cpu())
        fused_logits_all.append(out["action_fused_logits"].detach().cpu())
        labels_all.append(labels.detach().cpu())
        fn_batch = batch.get("file_name", [])
        if isinstance(fn_batch, str):
            file_names_all.append(fn_batch)
        else:
            file_names_all.extend([str(x) for x in fn_batch])
        if len(stats_rows) < args.max_saved_token_stats:
            meta0 = _first_image_meta(batch)
            stats_rows.append({
                **{k: (int(v) if isinstance(v, int) else v) for k, v in token_stats.items()},
                "compression_active": bool(token_stats.get("enabled", False)),
                "keep_ratio": float(keep_ratio),
                "summary_tokens": int(args.num_summary_tokens),
                "score_mode": str(args.token_score_mode),
                "score_source": "norm" if args.token_score_mode in {"norm", "hybrid"} else "fallback_norm",
                "image_height": int(args.image_height),
                "image_width": int(args.image_width),
                "patch_grid": meta0.get("patch_grid", [args.image_height // args.patch_size, args.image_width // args.patch_size]),
                "original_size": meta0.get("original_size"),
                "resized_size": meta0.get("resized_size"),
                "padding": meta0.get("padding"),
            })
        loss_components.append({
            "train": train,
            "epoch": epoch,
            "step": step,
            "loss": float(loss.detach().item()),
            "main_loss": float(main_loss.detach().item()),
            "r2a_loss": float(r2a_loss.detach().item()),
            "action_visual_loss": float(branch["action_visual_loss"].detach().item()),
            "action_reason_loss": float(branch["action_reason_loss"].detach().item()),
            "action_fused_loss_main_only": float(branch["action_fused_loss_main_only"].detach().item()),
            "action_fused_aux_loss": float(branch["action_fused_aux_loss"].detach().item()),
            "action_agree_loss": float(branch["action_agree_loss"].detach().item()),
            "action_branch_total": float(branch["action_branch_total"].detach().item()),
            "cf_loss": float(cf_loss.detach().item()),
            "grounding_loss": float(grounding_loss.detach().item()) if isinstance(grounding_loss, torch.Tensor) else 0.0,
            "total_loss": float(loss.detach().item()),
            "lr": float(getattr(args, "lr", 0.0)),
            "grad_norm": None,
            "effective_batch_size": int(getattr(args, "effective_batch_size", bs * accum)),
            "loss_divided_by_accumulation": True,
        })
        if grounding_stats:
            grounding_rows.append({"train": train, "epoch": epoch, "step": step, **grounding_stats})
        counterfactual_rows.append({"train": train, "epoch": epoch, "step": step, **cf_stats})
        if step % args.log_every == 0:
            print(json.dumps({
                "event": "fate_oia_batch",
                "train": train,
                "step": step,
                "loss": float(loss.item()),
                "main_loss": float(main_loss.item()),
                "r2a_loss": float(r2a_loss.item()),
                "action_branch_total": float(branch["action_branch_total"].item()),
                "action_visual_loss": float(branch["action_visual_loss"].item()),
                "action_reason_loss": float(branch["action_reason_loss"].item()),
                "action_fused_loss_main_only": float(branch["action_fused_loss_main_only"].item()),
                "action_fused_aux_loss": float(branch["action_fused_aux_loss"].item()),
                "action_agree_loss": float(branch["action_agree_loss"].item()),
                "cf_loss": float(cf_loss.item()),
                "cf_stats": cf_stats,
                "grounding_loss": float(grounding_loss.item()) if "grounding_loss" in locals() else 0.0,
                "grounding_stats": grounding_stats,
                "batch_size": bs,
                "token_stats": token_stats,
            }), flush=True)
    logits_tensor = torch.cat(logits_all, 0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    labels_tensor = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    metrics = evaluate_snna25(logits_tensor, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"]
    branch_metrics: dict[str, dict[str, float]] = {}
    if labels_tensor.numel() > 0:
        reason_part = logits_tensor[:, args.action_dim:]
        for name, action_logits in [
            ("action_visual", torch.cat(visual_logits_all, 0)),
            ("action_reason", torch.cat(reason_action_logits_all, 0)),
            ("action_fused", torch.cat(fused_logits_all, 0)),
        ]:
            branch_logits = torch.cat([action_logits, reason_part], dim=1)
            branch_metrics[name] = evaluate_snna25(branch_logits, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"]
    return {
        "loss": total_loss / max(count, 1),
        "count": count,
        "metrics": metrics,
        "branch_metrics": branch_metrics,
        "logits": logits_tensor,
        "visual_logits": torch.cat(visual_logits_all, 0) if visual_logits_all else torch.empty(0, args.action_dim),
        "reason_action_logits": torch.cat(reason_action_logits_all, 0) if reason_action_logits_all else torch.empty(0, args.action_dim),
        "fused_logits": torch.cat(fused_logits_all, 0) if fused_logits_all else torch.empty(0, args.action_dim),
        "labels": labels_tensor,
        "token_stats": stats_rows,
        "loss_components": loss_components,
        "grounding_stats": grounding_rows,
        "counterfactual_stats": counterfactual_rows,
        "file_names": file_names_all,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train full FATE-OIA token model with label-query, reason-to-action, optional compression and counterfactual loss.")
    ap.add_argument("--config", default="", help="Optional YAML config. CLI flags override matching config keys.")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4, choices=[4, 5])
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=224)
    ap.add_argument("--image_width", type=int, default=224)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--letterbox", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--auto_scale_lr", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reference_effective_batch", type=int, default=32)
    ap.add_argument("--base_head_lr_at_reference_batch", type=float, default=3e-4)
    ap.add_argument("--max_head_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_reason_to_action", type=float, default=0.1)
    ap.add_argument("--loss_action_visual", type=float, default=0.05)
    ap.add_argument("--loss_r2a_gt", type=float, default=0.10)
    ap.add_argument("--loss_action_agree", type=float, default=0.01)
    ap.add_argument("--include_fused_branch_loss", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--loss_action_fused_aux", type=float, default=0.0)
    ap.add_argument("--r2a_consistency_mode", choices=["none", "detach_mimic", "gt", "gt_and_agree"], default="gt_and_agree")
    ap.add_argument("--loss_counterfactual", type=float, default=0.0)
    ap.add_argument("--loss_grounding", type=float, default=0.0)
    ap.add_argument("--grounding_cache_jsonl", default="")
    ap.add_argument("--grounding_mode", choices=["global", "label", "both"], default="both")
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--grounding_categories", default="person,rider,bike,car,bus,truck,motor,traffic light,traffic sign,lane/crosswalk")
    ap.add_argument("--grounding_image_width", type=int, default=1280)
    ap.add_argument("--grounding_image_height", type=int, default=720)
    ap.add_argument("--counterfactual_topk_ratio", type=float, default=0.15)
    ap.add_argument("--cf_mask_fill", choices=["zero", "mean", "mask_token"], default="mean")
    ap.add_argument("--use_label_query", action="store_true", default=True)
    ap.add_argument("--use_reason_to_action", action="store_true", default=True)
    ap.add_argument("--token_compression", choices=["none", "keep_merge"], default="none")
    ap.add_argument("--token_keep_ratio", type=float, default=1.0)
    ap.add_argument("--compression_start_epoch", type=int, default=8)
    ap.add_argument("--compression_keep_ratio_start", type=float, default=0.85)
    ap.add_argument("--compression_keep_ratio_final", type=float, default=0.65)
    ap.add_argument("--compression_warmup_epochs", type=int, default=6)
    ap.add_argument("--token_score_mode", choices=["norm", "label_attention", "grounding_prior", "hybrid"], default="norm")
    ap.add_argument("--token_score_alpha_label", type=float, default=1.0)
    ap.add_argument("--token_score_beta_grounding", type=float, default=0.5)
    ap.add_argument("--token_score_gamma_uncertainty", type=float, default=0.2)
    ap.add_argument("--token_score_delta_norm", type=float, default=0.1)
    ap.add_argument("--num_summary_tokens", type=int, default=1)
    ap.add_argument("--min_tokens", type=int, default=16)
    ap.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="fixed")
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_saved_token_stats", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    apply_config_defaults(args, load_config_defaults(args.config))

    args.num_gpus = 1
    lr_info = compute_lr_scaling(
        per_gpu_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reference_effective_batch=args.reference_effective_batch,
        base_lr_at_reference_batch=args.base_head_lr_at_reference_batch,
        num_gpus=args.num_gpus,
        auto_scale_lr=args.auto_scale_lr,
        current_lr=args.lr,
        max_lr=args.max_head_lr,
    )
    args.effective_batch_size = lr_info.effective_batch_size
    if args.auto_scale_lr:
        args.lr = lr_info.lr_actual

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = FATEOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, use_label_query=args.use_label_query).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_multilabel_criterion(args)
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    grounding_cache = load_grounding_cache(args.grounding_cache_jsonl) if args.grounding_cache_jsonl else {}
    args.reason_grounding_rules_map = load_reason_grounding_rules(args.reason_grounding_rules, args.reason_dim)
    if args.loss_grounding > 0 and not grounding_cache:
        print(json.dumps({"event": "fate_oia_grounding_disabled", "reason": "empty_grounding_cache"}), flush=True)
    if args.loss_grounding > 0 and args.grounding_mode in {"label", "both"} and not args.reason_grounding_rules_map:
        print(json.dumps({"event": "fate_oia_label_grounding_disabled", "reason": "empty_reason_grounding_rules"}), flush=True)
    is_smoke = bool(args.max_train_samples or args.max_val_samples or args.epochs <= 1)
    manifest = build_run_manifest(args, out_dir, len(train_loader.dataset), len(val_loader.dataset), is_smoke=is_smoke)
    _write_json(out_dir / "run_manifest.json", manifest)
    _write_json(out_dir / "training_config_resolved.yaml", vars(args))
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, criterion, optimizer, device, True, grounding_cache, epoch)
        val_stats = run_epoch(args, backbone, model, val_loader, criterion, optimizer, device, False, grounding_cache, epoch)
        score = float(val_stats["metrics"].get("Act_mF1", 0.0)) + float(val_stats["metrics"].get("Exp_mF1", 0.0))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"], "val_metrics": val_stats["metrics"], "val_branch_metrics": val_stats.get("branch_metrics", {}), "selection_score": score}
        history.append(row)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        latest = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "dim": dim, "best_score": max(best, score)}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        torch.save(val_stats["logits"], out_dir / "val_logits_latest.pt")
        torch.save(val_stats["labels"], out_dir / "val_labels_latest.pt")
        (out_dir / "token_stats_latest.json").write_text(json.dumps({"train": train_stats["token_stats"], "val": val_stats["token_stats"]}, indent=2), encoding="utf-8")
        write_epoch_artifacts(out_dir, epoch, train_stats, val_stats, manifest)
        if score >= best:
            best = score
            torch.save(latest, out_dir / "checkpoint_best.pth")
            torch.save(val_stats["logits"], out_dir / "val_logits_best.pt")
            torch.save(val_stats["labels"], out_dir / "val_labels_best.pt")
        print(json.dumps({"event": "fate_oia_epoch", **row}), flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
